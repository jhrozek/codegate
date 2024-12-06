import asyncio
import ssl
import socket
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse
from .core.logging import logger
from .core.security import create_ssl_context
from .config.settings import settings
from .utils.proxy import get_target_url

# Increase buffer sizes
MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 64 * 1024  # 64KB

class ProxyProtocol(asyncio.Protocol):
    def __init__(self, loop):
        self.loop = loop
        self.transport: Optional[asyncio.Transport] = None
        self.target_transport: Optional[asyncio.Transport] = None
        self.peername: Optional[Tuple[str, int]] = None
        self.buffer = bytearray()
        self.target_host: Optional[str] = None
        self.target_port: Optional[int] = None
        self.handshake_done = False
        self.is_connect = False
        self.content_length = 0
        self.headers_parsed = False
        self.method = None
        self.path = None
        self.version = None
        self.headers = []
        self.target = None
        self.original_path = None

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        self.peername = transport.get_extra_info('peername')
        logger.info(f"Client connected from {self.peername}")

    def extract_path(self, full_path: str) -> str:
        """Extract the path portion from a full URL"""
        if full_path.startswith('http://') or full_path.startswith('https://'):
            parsed = urlparse(full_path)
            path = parsed.path
            if parsed.query:
                path = f"{path}?{parsed.query}"
            return path.lstrip('/')
        elif full_path.startswith('/'):
            return full_path.lstrip('/')
        return full_path

    def parse_headers(self) -> bool:
        """Parse HTTP headers from buffer"""
        try:
            if b'\r\n\r\n' not in self.buffer:
                return False

            headers_end = self.buffer.index(b'\r\n\r\n')
            headers = self.buffer[:headers_end].split(b'\r\n')
            
            # Parse request line
            request = headers[0].decode('utf-8')
            self.method, full_path, self.version = request.split(' ')
            
            # Store original path
            self.original_path = full_path
            
            # For CONNECT requests, store target and set path to empty
            if self.method == 'CONNECT':
                self.target = full_path
                self.path = ""
            else:
                # For non-CONNECT requests, extract the path portion
                self.path = self.extract_path(full_path)
            
            # Parse headers
            import re

            # Decode headers
            self.headers = [header.decode('utf-8') for header in headers[1:]]

            # Log the request details
            logger.info("=== Inbound Request ===")
            logger.info(f"Method: {self.method}")
            logger.info(f"Original Path: {self.original_path}")
            logger.info(f"Extracted Path: {self.path}")
            logger.info(f"Version: {self.version}")
            logger.info("Headers:")

            proxy_ep_value = None  # Initialize proxy-ep value

            # Process each header
            for header in self.headers:
                logger.info(f"  {header}")
                # Check for `authorization` header and extract `proxy-ep`
                if header.lower().startswith("authorization:"):
                    # Use regex to find proxy-ep in the header value
                    match = re.search(r"proxy-ep=([^;]+)", header)
                    if match:
                        proxy_ep_value = match.group(1)

            logger.info("=====================")

            # Log the extracted proxy-ep value
            if proxy_ep_value:
                logger.info(f"Extracted proxy-ep value: {proxy_ep_value}")
            else:
                logger.info("proxy-ep value not found.")


            return True
        except Exception as e:
            logger.error(f"Error parsing headers: {e}")
            return False

    async def handle_http_request(self):
        """Handle regular HTTP requests"""
        try:
            # Get target URL from path
            target_url = await get_target_url(self.path)
            if not target_url:
                self.send_error_response(404, b"Not Found")
                return
            logger.info(f"target URL {target_url}")
            # Parse target URL
            parsed_url = urlparse(target_url)
            logger.info(f"Parsed URL {parsed_url}")
            self.target_host = parsed_url.hostname
            self.target_port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)

            # Create connection to target
            target_protocol = ProxyTargetProtocol(self)
            await self.loop.create_connection(
                lambda: target_protocol,
                self.target_host,
                self.target_port,
                ssl=parsed_url.scheme == 'https'
            )

            # Add or update host header
            has_host = False
            new_headers = []
            for header in self.headers:
                if header.lower().startswith('host:'):
                    has_host = True
                    new_headers.append(f"Host: {self.target_host}")
                else:
                    new_headers.append(header)
            
            if not has_host:
                new_headers.append(f"Host: {self.target_host}")

            # Reconstruct request with path
            request_line = f"{self.method} /{self.path} {self.version}\r\n".encode()
            header_block = '\r\n'.join(new_headers).encode()
            headers = request_line + header_block + b'\r\n\r\n'
            
            # Send headers
            if self.target_transport:
                self.target_transport.write(headers)
                
                # Send body in chunks
                body_start = self.buffer.index(b'\r\n\r\n') + 4
                body = self.buffer[body_start:]
                
                # Send in chunks
                for i in range(0, len(body), CHUNK_SIZE):
                    chunk = body[i:i + CHUNK_SIZE]
                    self.target_transport.write(chunk)
            else:
                logger.error("Target transport not available")
                self.send_error_response(502, b"Failed to establish target connection")

        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
            self.send_error_response(502, str(e).encode())

    def data_received(self, data: bytes):
        try:
            if len(self.buffer) + len(data) > MAX_BUFFER_SIZE:
                logger.error("Request too large")
                self.send_error_response(413, b"Request body too large")
                return

            self.buffer.extend(data)

            if not self.headers_parsed:
                self.headers_parsed = self.parse_headers()
                if not self.headers_parsed:
                    return

                if self.method == 'CONNECT':
                    self.handle_connect()
                else:
                    # For non-CONNECT requests, handle immediately
                    asyncio.create_task(self.handle_http_request())
            elif self.target_transport and not self.target_transport.is_closing():
                # Forward data to target
                self.target_transport.write(data)

        except Exception as e:
            logger.error(f"Error in data_received: {e}")
            self.send_error_response(502, str(e).encode())

    def handle_connect(self):
        """Handle CONNECT requests"""
        try:
            path = unquote(self.target)  # Use the stored target instead of path
            if ':' in path:
                self.target_host, port = path.split(':')
                self.target_port = int(port)
                logger.info(f"CONNECT request to {self.target_host}:{self.target_port}")
                logger.info("Headers:")
                for header in self.headers:
                    logger.info(f"  {header}")
                
                # Connect to target
                self.is_connect = True
                asyncio.create_task(self.connect_to_target())
                self.handshake_done = True
            else:
                logger.error(f"Invalid CONNECT path: {path}")
                self.send_error_response(400, b"Invalid CONNECT path")
        except Exception as e:
            logger.error(f"Error handling CONNECT: {e}")
            self.send_error_response(502, str(e).encode())

    def send_error_response(self, status: int, message: bytes):
        """Send error response to client"""
        response = (
            f"HTTP/1.1 {status} {self.get_status_text(status)}\r\n"
            f"Content-Length: {len(message)}\r\n"
            f"Content-Type: text/plain\r\n"
            f"\r\n"
        ).encode() + message
        if self.transport and not self.transport.is_closing():
            self.transport.write(response)
            self.transport.close()

    def get_status_text(self, status: int) -> str:
        """Get status text for HTTP status code"""
        status_texts = {
            400: "Bad Request",
            404: "Not Found",
            413: "Request Entity Too Large",
            502: "Bad Gateway"
        }
        return status_texts.get(status, "Error")

    async def connect_to_target(self):
        """Connect to target server for CONNECT requests"""
        try:
            if not self.target_host or not self.target_port:
                raise ValueError("Target host and port not set")

            # Log the connection attempt
            logger.debug(f"Attempting to connect to {self.target_host}:{self.target_port}")

            # Resolve the target host to an IP address
            addrinfo = await self.loop.getaddrinfo(
                self.target_host,
                self.target_port,
                family=socket.AF_UNSPEC,  # Allow both IPv4 and IPv6
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP
            )
            
            if not addrinfo:
                raise ValueError(f"Could not resolve {self.target_host}")

            # Try each resolved address until one works
            last_error = None
            for family, type, proto, canonname, sockaddr in addrinfo:
                # split sockaddr by :
                sockaddr = sockaddr[0], sockaddr[1]
                
                try:
                    logger.debug(f"Trying to connect to {sockaddr}")

                    target_protocol = ProxyTargetProtocol(self)
                    transport, _ = await self.loop.create_connection(
                        lambda: target_protocol,
                        host=sockaddr[0],
                        port=sockaddr[1]
                    )
                    
                    # If we get here, the connection was successful
                    logger.debug(f"Successfully connected to {sockaddr}")
                    
                    # Send 200 Connection Established
                    if self.transport and not self.transport.is_closing():
                        self.transport.write(
                            b'HTTP/1.1 200 Connection Established\r\n'
                            b'Proxy-Agent: ProxyPilot\r\n'
                            b'Connection: keep-alive\r\n\r\n'
                        )
                    return
                except Exception as e:
                    last_error = e
                    logger.debug(f"Failed to connect to {sockaddr}: {e}")
                    continue

            # If we get here, none of the addresses worked
            if last_error:
                raise last_error
            else:
                raise ValueError(f"Could not connect to any resolved address for {self.target_host}")

        except Exception as e:
            logger.error(f"Failed to connect to target {self.target_host}:{self.target_port}: {e}")
            self.send_error_response(502, str(e).encode())

    def connection_lost(self, exc):
        logger.info(f"Client disconnected from {self.peername}")
        if self.target_transport and not self.target_transport.is_closing():
            self.target_transport.close()

class ProxyTargetProtocol(asyncio.Protocol):
    def __init__(self, proxy: ProxyProtocol):
        self.proxy = proxy
        self.transport: Optional[asyncio.Transport] = None

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        self.proxy.target_transport = transport

    def data_received(self, data: bytes):
        if self.proxy.transport and not self.proxy.transport.is_closing():
            self.proxy.transport.write(data)

    def connection_lost(self, exc):
        if self.proxy.transport and not self.proxy.transport.is_closing():
            self.proxy.transport.close()

async def create_proxy_server(host: str, port: int, ssl_context: Optional[ssl.SSLContext] = None):
    """Create and start the proxy server"""
    loop = asyncio.get_event_loop()

    def create_protocol():
        return ProxyProtocol(loop)

    server = await loop.create_server(
        create_protocol,
        host,
        port,
        ssl=ssl_context,
        reuse_port=True,
        start_serving=True
    )

    logger.info(f"Proxy server running on {host}:{port}")
    return server

async def run_proxy_server():
    """Run the proxy server"""
    try:
        # Create SSL context
        ssl_context = create_ssl_context()

        # Start proxy server
        server = await create_proxy_server(
            settings.HOST,
            settings.PORT,
            ssl_context
        )

        async with server:
            await server.serve_forever()

    except Exception as e:
        logger.error(f"Proxy server error: {e}")
        raise