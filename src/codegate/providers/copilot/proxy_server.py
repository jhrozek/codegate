import asyncio
import socket
import ssl
from typing import Dict, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import httpx
from fastapi import Request, Response, WebSocket

from ..config.settings import VALIDATED_ROUTES, settings
from ..core.logging import log_error, log_proxy_forward, logger
from .config.settings import settings
from .core.logging import logger
from .core.security import ca, create_ssl_context
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
        self.ssl_context = None
        self.decrypted_data = bytearray()

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        self.peername = transport.get_extra_info('peername')
        logger.info(f"Client connected from {self.peername}")

    def extract_path(self, full_path: str) -> str:
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
        try:
            if b'\r\n\r\n' not in self.buffer:
                return False

            headers_end = self.buffer.index(b'\r\n\r\n')
            headers = self.buffer[:headers_end].split(b'\r\n')
            
            request = headers[0].decode('utf-8')
            self.method, full_path, self.version = request.split(' ')
            
            self.original_path = full_path
            
            if self.method == 'CONNECT':
                self.target = full_path
                self.path = ""
            else:
                self.path = self.extract_path(full_path)
            
            import re
            self.headers = [header.decode('utf-8') for header in headers[1:]]

            logger.info("=== Inbound Request ===")
            logger.info(f"Method: {self.method}")
            logger.info(f"Original Path: {self.original_path}")
            logger.info(f"Extracted Path: {self.path}")
            logger.info(f"Version: {self.version}")
            logger.info("Headers:")

            proxy_ep_value = None

            for header in self.headers:
                logger.info(f"  {header}")
                if header.lower().startswith("authorization:"):
                    match = re.search(r"proxy-ep=([^;]+)", header)
                    if match:
                        proxy_ep_value = match.group(1)

            logger.info("=====================")

            if proxy_ep_value:
                logger.info(f"Extracted proxy-ep value: {proxy_ep_value}")
            else:
                logger.info("proxy-ep value not found.")

            return True
        except Exception as e:
            logger.error(f"Error parsing headers: {e}")
            return False

    def log_decrypted_data(self, data: bytes, direction: str):
        try:
            decoded = data.decode('utf-8')
            logger.info(f"=== Decrypted {direction} Data ===")
            logger.info(decoded)
            logger.info("=" * 40)
        except UnicodeDecodeError:
            logger.info(f"=== Decrypted {direction} Data (hex) ===")
            logger.info(data.hex())
            logger.info("=" * 40)

    async def handle_http_request(self):
        logger.debug(f"Method: {self.method}")
        try:
            target_url = await get_target_url(self.path)
            if not target_url:
                self.send_error_response(404, b"Not Found")
                return
            logger.info(f"target URL {target_url}")
            
            parsed_url = urlparse(target_url)
            logger.info(f"Parsed URL {parsed_url}")
            self.target_host = parsed_url.hostname
            self.target_port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)

            target_protocol = ProxyTargetProtocol(self)
            await self.loop.create_connection(
                lambda: target_protocol,
                self.target_host,
                self.target_port,
                ssl=parsed_url.scheme == 'https'
            )

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

            request_line = f"{self.method} /{self.path} {self.version}\r\n".encode()
            header_block = '\r\n'.join(new_headers).encode()
            headers = request_line + header_block + b'\r\n\r\n'
            
            if self.target_transport:
                self.log_decrypted_data(headers, "Request")
                self.target_transport.write(headers)
                
                body_start = self.buffer.index(b'\r\n\r\n') + 4
                body = self.buffer[body_start:]
                
                if body:
                    self.log_decrypted_data(body, "Request Body")
                
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
                    asyncio.create_task(self.handle_http_request())
            elif self.target_transport and not self.target_transport.is_closing():
                self.log_decrypted_data(data, "Client to Server")
                self.target_transport.write(data)

        except Exception as e:
            logger.error(f"Error in data_received: {e}")
            self.send_error_response(502, str(e).encode())

    def handle_connect(self):
        try:
            path = unquote(self.target)
            if ':' in path:
                self.target_host, port = path.split(':')
                self.target_port = int(port)
                logger.info(f"CONNECT request to {self.target_host}:{self.target_port}")
                logger.info("Headers:")
                for header in self.headers:
                    logger.info(f"  {header}")
                
                cert_path, key_path = ca.get_domain_certificate(self.target_host)
                
                self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                self.ssl_context.load_cert_chain(cert_path, key_path)
                self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
                
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
        status_texts = {
            400: "Bad Request",
            404: "Not Found",
            413: "Request Entity Too Large",
            502: "Bad Gateway"
        }
        return status_texts.get(status, "Error")

    async def connect_to_target(self):
        try:
            if not self.target_host or not self.target_port:
                raise ValueError("Target host and port not set")

            logger.debug(f"Attempting to connect to {self.target_host}:{self.target_port}")

            # Create SSL context for target connection
            target_ssl_context = ssl.create_default_context()
            # Don't verify certificates when connecting to target
            target_ssl_context.check_hostname = False
            target_ssl_context.verify_mode = ssl.CERT_NONE

            # Connect directly to target host
            target_protocol = ProxyTargetProtocol(self)
            transport, _ = await self.loop.create_connection(
                lambda: target_protocol,
                self.target_host,
                self.target_port,
                ssl=target_ssl_context
            )
            
            logger.debug(f"Successfully connected to {self.target_host}:{self.target_port}")
            
            # Send 200 Connection Established
            if self.transport and not self.transport.is_closing():
                self.transport.write(
                    b'HTTP/1.1 200 Connection Established\r\n'
                    b'Proxy-Agent: ProxyPilot\r\n'
                    b'Connection: keep-alive\r\n\r\n'
                )
                
                # Upgrade client connection to SSL
                transport = await self.loop.start_tls(
                    self.transport,
                    self,
                    self.ssl_context,
                    server_side=True
                )
                self.transport = transport

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
            self.proxy.log_decrypted_data(data, "Server to Client")
            self.proxy.transport.write(data)

    def connection_lost(self, exc):
        if self.proxy.transport and not self.proxy.transport.is_closing():
            self.proxy.transport.close()

async def create_proxy_server(host: str, port: int, ssl_context: Optional[ssl.SSLContext] = None):
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
    try:
        ssl_context = create_ssl_context()
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





async def get_target_url(path: str) -> Optional[str]:
    """Get target URL for the given path"""
    logger.debug(f"Attempting to get target URL for path: {path}")

    # Special handling for Copilot completions endpoint
    if '/v1/engines/copilot-codex/completions' in path:
        logger.debug("Using special case for completions endpoint")
        return 'https://proxy.enterprise.githubcopilot.com/v1/engines/copilot-codex/completions'
    
    for route in VALIDATED_ROUTES:
        if path == route.path:
            logger.debug(f"Found exact path match: {path} -> {route.target}")
            return str(route.target)

    # Then check for prefix match
    for route in VALIDATED_ROUTES:
        if path.startswith(route.path):
            # For prefix matches, keep the rest of the path
            remaining_path = path[len(route.path):]
            # Make sure we don't end up with double slashes
            if remaining_path and remaining_path.startswith('/'):
                remaining_path = remaining_path[1:]
            target = urljoin(str(route.target), remaining_path)
            logger.debug(f"Found prefix match: {path} -> {target} (using route {route.path} -> {route.target})")
            return target

    logger.warning(f"No route found for path: {path}")
    return None

def prepare_headers(request: Union[Request, WebSocket], target_url: str) -> Dict[str, str]:
    """Prepare headers for the proxy request"""
    headers = {}
    
    # Get headers from request
    if isinstance(request, Request):
        request_headers = request.headers
    else:  # WebSocket
        request_headers = request.headers
    
    # Copy preserved headers from the original request
    for header, value in request_headers.items():
        if header.lower() in [h.lower() for h in settings.PRESERVED_HEADERS]:
            headers[header] = value

    # Add endpoint-specific headers
    if isinstance(request, Request):
        path = urlparse(str(request.url)).path
        if path in settings.ENDPOINT_HEADERS:
            headers.update(settings.ENDPOINT_HEADERS[path])

    # Set the Host header to match the target
    target_parsed = urlparse(target_url)
    headers['Host'] = target_parsed.netloc

    # Remove any headers that shouldn't be forwarded
    for header in settings.REMOVED_HEADERS:
        headers.pop(header.lower(), None)

    # Log headers for debugging
    logger.debug(f"Prepared headers for {target_url}: {headers}")

    return headers

async def forward_request(
    request: Request,
    target_url: str,
    client: httpx.AsyncClient
) -> Tuple[Response, int]:
    """Forward the request to the target URL"""
    try:
        # Prepare headers
        headers = prepare_headers(request, target_url)

        # Get request body
        body = await request.body()

        logger.debug(f"Forwarding {request.method} request to {target_url}")
        logger.debug(f"Request headers: {headers}")
        if body:
            logger.debug(f"Request body length: {len(body)} bytes")

        # Forward the request
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            follow_redirects=True
        )

        logger.debug(f"Received response from {target_url}: status={response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")

        # Log the forwarded request
        log_proxy_forward(target_url, request.method, response.status_code)

        # Create FastAPI response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        ), response.status_code

    except httpx.RequestError as e:
        log_error("request_error", str(e), {"target_url": target_url})
        logger.error(f"Request error for {target_url}: {str(e)}")
        return Response(
            content=str(e).encode(),
            status_code=502,
            media_type="text/plain"
        ), 502

    except Exception as e:
        log_error("proxy_error", str(e), {"target_url": target_url})
        logger.error(f"Proxy error for {target_url}: {str(e)}")
        return Response(
            content=str(e).encode(),
            status_code=500,
            media_type="text/plain"
        ), 500

async def tunnel_websocket(websocket: WebSocket, target_host: str, target_port: int):
    """Create a tunnel between WebSocket and target server"""
    try:
        # Connect to target server
        reader, writer = await asyncio.open_connection(target_host, target_port)

        # Create bidirectional tunnel
        async def forward_ws_to_target():
            try:
                while True:
                    data = await websocket.receive_bytes()
                    writer.write(data)
                    await writer.drain()
            except Exception as e:
                logger.error(f"WS to target error: {e}")

        async def forward_target_to_ws():
            try:
                while True:
                    data = await reader.read(8192)
                    if not data:
                        break
                    await websocket.send_bytes(data)
            except Exception as e:
                logger.error(f"Target to WS error: {e}")

        # Run both forwarding tasks
        await asyncio.gather(
            forward_ws_to_target(),
            forward_target_to_ws(),
            return_exceptions=True
        )

    except Exception as e:
        log_error("tunnel_error", str(e))
        await websocket.close(code=1011, reason=str(e))
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

def create_error_response(status_code: int, message: str) -> Response:
    """Create an error response"""
    content = {
        "error": {
            "message": message,
            "type": "proxy_error",
            "code": status_code
        }
    }
    return Response(
        content=str(content).encode(),
        status_code=status_code,
        media_type="application/json"
    )
