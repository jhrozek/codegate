import asyncio
import ssl
import structlog
from typing import Optional, Tuple, Dict, Union
from urllib.parse import unquote, urlparse, urljoin
from fastapi import Request, Response, WebSocket
import httpx
from codegate.codegate_logging import setup_logging
from codegate.core.security import CertificateManager
from codegate.config import Config, VALIDATED_ROUTES

# Constants for buffer sizes
MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 64 * 1024  # 64KB
logger = structlog.get_logger("codegate")

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
        self.headers_parsed = False
        self.method = None
        self.path = None
        self.version = None
        self.headers = []
        self.original_path = None
        self.cfg = Config.load()

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        self.peername = transport.get_extra_info('peername')
        logger.info(f"Client connected from {self.peername}")

    def extract_path(self, full_path: str) -> str:
        """Extract the path portion from a full URL."""
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
        """Parse HTTP headers from buffer."""
        try:
            if b'\r\n\r\n' not in self.buffer:
                return False

            headers_end = self.buffer.index(b'\r\n\r\n')
            headers = self.buffer[:headers_end].split(b'\r\n')

            request_line = headers[0].decode('utf-8')
            self.method, full_path, self.version = request_line.split(' ')

            self.original_path = full_path
            self.path = self.extract_path(full_path) if self.method != 'CONNECT' else ""
            self.headers = [header.decode('utf-8') for header in headers[1:]]
            return True
        except Exception as e:
            logger.error(f"Error parsing headers: {e}")
            return False

    async def get_target_url(self, path: str) -> Optional[str]:
        """Resolve the target URL for the given path."""
        logger.debug(f"Resolving target URL for path: {path}")

        if '/v1/engines/copilot-codex/completions' in path:
            return 'https://proxy.individual.githubcopilot.com/v1/engines/copilot-codex/completions'

        for route in VALIDATED_ROUTES:
            if path == route.path or path.startswith(route.path):
                remaining_path = path[len(route.path):].lstrip('/')
                return urljoin(str(route.target), remaining_path)

        logger.warning(f"No route found for path: {path}")
        return None

    async def forward_request(self, request: Request, target_url: str, client: httpx.AsyncClient):
        """Forward the HTTP request to the target URL."""
        try:
            headers = self.prepare_headers(request, target_url)
            body = await request.body()
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                follow_redirects=True
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            ), response.status_code
        except Exception as e:
            logger.error(f"Error forwarding request: {e}")
            return Response(content=str(e).encode(), status_code=502), 502

    def prepare_headers(self, request: Union[Request, WebSocket], target_url: str) -> Dict[str, str]:
        """Prepare headers for the proxy request."""
        headers = {}
        request_headers = request.headers

        for header, value in request_headers.items():
            if header.lower() in [h.lower() for h in self.cfg.PRESERVED_HEADERS]:
                headers[header] = value

        target_parsed = urlparse(target_url)
        headers['Host'] = target_parsed.netloc

        for header in self.cfg.REMOVED_HEADERS:
            headers.pop(header.lower(), None)

        logger.debug(f"Prepared headers: {headers}")
        return headers

    def data_received(self, data: bytes):
        """Handle incoming data."""
        try:
            if len(self.buffer) + len(data) > MAX_BUFFER_SIZE:
                self.send_error_response(413, b"Request body too large")
                return

            self.buffer.extend(data)
            if not self.headers_parsed:
                self.headers_parsed = self.parse_headers()
                if self.headers_parsed:
                    asyncio.create_task(self.handle_http_request())
        except Exception as e:
            logger.error(f"Error in data_received: {e}")

    async def handle_http_request(self):
        """Handle a standard HTTP request."""
        target_url = await self.get_target_url(self.path)
        if not target_url:
            self.send_error_response(404, b"Not Found")
            return

        async with httpx.AsyncClient() as client:
            await self.forward_request(None, target_url, client)

    def send_error_response(self, status: int, message: bytes):
        """Send an error response to the client."""
        response = (
            f"HTTP/1.1 {status} {self.get_status_text(status)}\r\n"
            f"Content-Length: {len(message)}\r\n"
            f"Content-Type: text/plain\r\n\r\n"
        ).encode() + message
        if self.transport:
            self.transport.write(response)
            self.transport.close()

    def get_status_text(self, status: int) -> str:
        return {
            400: "Bad Request",
            404: "Not Found",
            413: "Request Entity Too Large",
            502: "Bad Gateway"
        }.get(status, "Error")


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
    cfg = Config.load()
    try:
        # Create certificate manager instance
        cert_manager = CertificateManager()

        # Ensure certificates exist
        cert_manager.ensure_certificates_exist()

        # Create SSL context using instance method
        ssl_context = cert_manager.create_ssl_context()

        server = await create_proxy_server(
            cfg.host,
            cfg.port,
            ssl_context
        )

        async with server:
            await server.serve_forever()

    except Exception as e:
        logger.error(f"Proxy server error: {e}")
        raise
