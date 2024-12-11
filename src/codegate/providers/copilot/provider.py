import asyncio
import json
import re
import ssl
from typing import Dict, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import httpx
import structlog
from fastapi import Request, Response, WebSocket

from codegate.config import Config

# from codegate.codegate_logging import log_error, log_proxy_forward, logger
from codegate.ca.codegate_ca import CertificateAuthority
from codegate.pipeline.base import InputPipelineInstance
from codegate.pipeline.factory import PipelineFactory
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.providers.copilot.mapping import VALIDATED_ROUTES
from codegate.providers.normalizer.completion import CompletionNormalizer

logger = structlog.get_logger("codegate")

# Increase buffer sizes
MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 64 * 1024  # 64KB


class CopilotProvider(asyncio.Protocol):
    def __init__(self, loop):
        logger.debug("Initializing CopilotProvider class: CopilotProvider")
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
        # Get the singleton instance of CertificateAuthority
        self.ca = CertificateAuthority.get_instance()
        self.pipeline_factory = PipelineFactory(SecretsManager())
        self.sessions = Dict[str, InputPipelineInstance]

    def _select_pipeline(self, headers: list[str]):
        for header in headers:
            if header.lower().startswith("user-agent:") and "GithubCopilot/" in header:
                logger.debug("Using FIM pipeline for Github Copilot")
                return self.pipeline_factory.create_fim_pipeline

        logger.debug("No pipeline selected")
        return None

    async def _body_through_pipeline(self, headers: list[str], body: bytes) -> bytes:
        logger.debug(f"Processing body through pipeline: {len(body)} bytes")
        pipeline_constructor = self._select_pipeline(headers)

        if pipeline_constructor is None:
            return body

        normalizer = None
        if pipeline_constructor == self.pipeline_factory.create_fim_pipeline:
            normalizer = CompletionNormalizer()
            json_body = json.loads(body)
            normalized_json_body = normalizer.normalize(json_body)
        else:
            return body

        pipeline = pipeline_constructor()
        result = await pipeline.process_request(
            request = normalized_json_body,
            provider = "github-copilot",
            prompt_id = "fake-prompt-id",
            model = "model_name",  # Replace with actual model if needed
            api_key = None,  # Extract from headers if needed
        )

        # Use the modified request body from the pipeline result
        if result.request:
            normalized_json_body = normalizer.denormalize(result.request)
            body = json.dumps(normalized_json_body).encode()

        logger.debug("Processing body through pipeline")
        return body

    def _data_through_pipeline(self, data: bytes) -> bytes:
        if b'\r\n\r\n' in data:
            # not tunneling
            return data

        headers_end = data.index(b'\r\n\r\n')
        headers = data[:headers_end].split(b'\r\n')

        request = headers[0].decode('utf-8')
        method, full_path, version = request.split(' ')
        body_start = data.index(b'\r\n\r\n') + 4
        body = data[body_start:]
        print("------ BEGIN META")
        print(f"Method: {method}, Full Path: {full_path}, Version: {version}")
        if method != 'POST' and full_path != "/chat/completions":
            return data

            pipeline = self.pipeline_factory.create_input_pipeline()
            pipeline_instance.
        print("------ END META")
        print("------ BEGIN DATA")
        print(body)
        print("------ END DATA")

    def _write_data_to_target(self, data: bytes):
        try:
        except ValueError:
             logger.error("Malformed data received")

        return self.target_transport.write(data)

    async def _write_headers_to_target(self, headers: list[str]):
        request_line = f"{self.method} /{self.path} {self.version}\r\n".encode()
        logger.debug(f"Request Line: {request_line}")
        header_block = '\r\n'.join(headers).encode()
        header_payload = request_line + header_block + b'\r\n\r\n'
        self.log_decrypted_data(header_block, "Request Headers")
        self.target_transport.write(header_payload)

    async def _write_body_to_target(self, headers: list[str], body: bytes):
        logger.debug(f"Writing body to target: {len(body)} bytes fn: _write_body_to_target")
        logger.debug(f"Using original {self.original_path} path: {self.path} method: {self.method}")
        logger.debug(f"Using headers: {headers}")

        print("------ ORIGINAL BODY")
        print(body)
        print("------ END ORIGINAL BODY")

        body = await self._body_through_pipeline(headers, body)

        for header in headers:
            if header.lower().startswith("content-length:"):
                print(f"------ original content-length header: {header}")
                headers.remove(header)
                break
        headers.append(f"Content-Length: {len(body)}")
        print(f"------ new content-length header: Content-Length: {len(body)}")

        print("------ MODIFIED BODY")
        print(body)
        print("------ END MODIFIED BODY")

        await self._write_headers_to_target(headers)
        for i in range(0, len(body), CHUNK_SIZE):
            chunk = body[i:i + CHUNK_SIZE]
            self.target_transport.write(chunk)

    def connection_made(self, transport: asyncio.Transport):
        logger.debug("Client connected fn: connection_made")
        self.transport = transport
        self.peername = transport.get_extra_info('peername')
        logger.debug(f"Client connected from {self.peername}")

    def extract_path(self, full_path: str) -> str:
        logger.debug(f"Extracting path from {full_path} fn: extract_path")
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
        logger.debug("Parsing headers fn: parse_headers")
        try:
            if b'\r\n\r\n' not in self.buffer:
                return False

            headers_end = self.buffer.index(b'\r\n\r\n')
            headers = self.buffer[:headers_end].split(b'\r\n')

            request = headers[0].decode('utf-8')
            self.method, full_path, self.version = request.split(' ')

            self.original_path = full_path

            if self.method == 'CONNECT':
                logger.debug(f"CONNECT request to {full_path}")
                self.target = full_path
                self.path = ""
            else:
                logger.debug(f"Request: {self.method} {full_path} {self.version}")
                self.path = self.extract_path(full_path)

            self.headers = [header.decode('utf-8') for header in headers[1:]]

            logger.debug("=" * 40)
            logger.debug("=== Inbound Request ===")
            logger.debug(f"Method: {self.method}")
            logger.debug(f"Original Path: {self.original_path}")
            logger.debug(f"Extracted Path: {self.path}")
            logger.debug(f"Version: {self.version}")
            logger.debug("Headers:")

            logger.debug("=" * 40)

            logger.debug("Searching for proxy-ep header value")
            proxy_ep_value = None

            for header in self.headers:
                logger.debug(f"  {header}")
                if header.lower().startswith("authorization:"):
                    match = re.search(r"proxy-ep=([^;]+)", header)
                    if match:
                        proxy_ep_value = match.group(1)

            if proxy_ep_value:
                logger.debug(f"Extracted proxy-ep value: {proxy_ep_value}")
            else:
                logger.debug("proxy-ep value not found.")
            logger.debug("=" * 40)

            return True
        except Exception as e:
            logger.error(f"Error parsing headers: {e}")
            return False

    def log_decrypted_data(self, data: bytes, direction: str):
        try:
            decoded = data.decode('utf-8')
            logger.debug(f"=== Decrypted {direction} Data ===")
            logger.debug(decoded)
            logger.debug("=" * 40)
        except UnicodeDecodeError:
            logger.debug(f"=== Decrypted {direction} Data (hex) ===")
            logger.debug(data.hex())
            logger.debug("=" * 40)

    async def handle_http_request(self):
        logger.debug("Handling HTTP request fn: handle_http_request")
        logger.debug("=" * 40)
        logger.debug(f"Method: {self.method}")
        logger.debug(f"Searched Path: {self.path} in target URL")
        try:
            target_url = await self.get_target_url(self.path)
            logger.debug(f"Target URL: {target_url}")
            if not target_url:
                self.send_error_response(404, b"Not Found")
                return
            logger.debug(f"target URL {target_url}")

            parsed_url = urlparse(target_url)
            logger.debug(f"Parsed URL {parsed_url}")
            self.target_host = parsed_url.hostname
            self.target_port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
            logger.debug("=" * 40)
            target_protocol = CopilotProxyTargetProtocol(self)
            logger.debug(f"Connecting to {self.target_host}:{self.target_port}")
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

            if self.target_transport:
                body_start = self.buffer.index(b'\r\n\r\n') + 4
                body = self.buffer[body_start:]

                if body:
                    self.log_decrypted_data(body, "Request Body")

                await self._write_body_to_target(new_headers, body)
            else:
                logger.debug("=" * 40)
                logger.error("Target transport not available")
                logger.debug("=" * 40)
                self.send_error_response(502, b"Failed to establish target connection")

        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
            self.send_error_response(502, str(e).encode())

    def data_received(self, data: bytes):
        logger.debug(f"Data received from {self.peername} fn: data_received")

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
                    logger.debug("Handling CONNECT request")
                    self.handle_connect()
                else:
                    logger.debug("Handling HTTP request")
                    asyncio.create_task(self.handle_http_request())
            elif self.target_transport and not self.target_transport.is_closing():
                self.log_decrypted_data(data, "Client to Server")
                self._write_data_to_target(data)

        except Exception as e:
            logger.error(f"Error in data_received: {e}")
            self.send_error_response(502, str(e).encode())

    def handle_connect(self):
        try:
            path = unquote(self.target)
            if ':' in path:
                self.target_host, port = path.split(':')
                self.target_port = int(port)
                logger.debug("=" * 40)
                logger.debug(f"CONNECT request to {self.target_host}:{self.target_port}")
                logger.debug("Headers:")
                for header in self.headers:
                    logger.debug(f"  {header}")

                logger.debug("=" * 40)
                cert_path, key_path = self.ca.get_domain_certificate(self.target_host)

                logger.debug(f"Setting up SSL context for {self.target_host}")
                logger.debug(f"Using certificate: {cert_path}")
                logger.debug(f"Using key: {key_path}")

                self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                self.ssl_context.load_cert_chain(cert_path, key_path)
                self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2

                self.is_connect = True
                logger.debug("CONNECT handshake complete")
                asyncio.create_task(self.connect_to_target())
                self.handshake_done = True
            else:
                logger.error(f"Invalid CONNECT path: {path}")
                self.send_error_response(400, b"Invalid CONNECT path")
        except Exception as e:
            logger.error(f"Error handling CONNECT: {e}")
            self.send_error_response(502, str(e).encode())

    def send_error_response(self, status: int, message: bytes):
        logger.debug(f"Sending error response: {status} {message} fn: send_error_response")
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
        logger.debug(f"Getting status text for {status} fn: get_status_text")
        status_texts = {
            400: "Bad Request",
            404: "Not Found",
            413: "Request Entity Too Large",
            502: "Bad Gateway"
        }
        return status_texts.get(status, "Error")

    async def connect_to_target(self):
        logger.debug(f"Connecting to target {self.target_host}:{self.target_port} fn: connect_to_target")
        try:
            if not self.target_host or not self.target_port:
                raise ValueError("Target host and port not set")

            logger.debug(f"Attempting to connect to {self.target_host}:{self.target_port}")

            # Create SSL context for target connection
            logger.debug("Creating SSL context for target connection")
            target_ssl_context = ssl.create_default_context()
            # Don't verify certificates when connecting to target
            target_ssl_context.check_hostname = False
            target_ssl_context.verify_mode = ssl.CERT_NONE

            # Connect directly to target host
            logger.debug(f"Connecting to {self.target_host}:{self.target_port}")
            target_protocol = CopilotProxyTargetProtocol(self)
            transport, _ = await self.loop.create_connection(
                lambda: target_protocol,
                self.target_host,
                self.target_port,
                ssl=target_ssl_context
            )

            logger.debug(f"Successfully connected to {self.target_host}:{self.target_port}")

            # Send 200 Connection Established
            if self.transport and not self.transport.is_closing():
                logger.debug("Sending 200 Connection Established response")
                self.transport.write(
                    b'HTTP/1.1 200 Connection Established\r\n'
                    b'Proxy-Agent: ProxyPilot\r\n'
                    b'Connection: keep-alive\r\n\r\n'
                )

                # Upgrade client connection to SSL
                logger.debug("Upgrading client connection to SSL")
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
        logger.debug(f"Connection lost from {self.peername} fn: connection_lost")
        logger.debug(f"Client disconnected from {self.peername}")
        if self.target_transport and not self.target_transport.is_closing():
            self.target_transport.close()

    @classmethod
    async def create_proxy_server(cls, host: str, port: int, ssl_context: Optional[ssl.SSLContext] = None):
        logger.debug(f"Creating proxy server on {host}:{port} fn: create_proxy_server")
        loop = asyncio.get_event_loop()

        def create_protocol():
            logger.debug("Creating protocol for proxy server fn: create_protocol")
            return cls(loop)

        logger.debug(f"Starting proxy server on {host}:{port}")
        server = await loop.create_server(
            create_protocol,
            host,
            port,
            ssl=ssl_context,
            reuse_port=True,
            start_serving=True
        )

        logger.debug(f"Proxy server running on {host}:{port}")
        return server

    @classmethod
    async def run_proxy_server(cls):
        logger.debug("Running proxy server fn: run_proxy_server")
        try:
            # Get the singleton instance of CertificateAuthority
            ca = CertificateAuthority.get_instance()
            logger.debug("Creating SSL context for proxy server")
            ssl_context = ca.create_ssl_context()
            server = await cls.create_proxy_server(
                Config.get_config().host,
                Config.get_config().proxy_port,
                ssl_context
            )
            logger.debug("Proxy server created")
            async with server:
                await server.serve_forever()

        except Exception as e:
            logger.error(f"Proxy server error: {e}")
            raise

    @classmethod
    async def get_target_url(cls, path: str) -> Optional[str]:
        """Get target URL for the given path"""
        logger.debug(f"Attempting to get target URL for path: {path} fn: get_target_url")

        logger.debug("=" * 40)
        logger.debug("Validated routes:")
        for route in VALIDATED_ROUTES:
            if path == route.path:
                logger.debug(f"  {route.path} -> {route.target}")
                logger.debug(f"Found exact path match: {path} -> {route.target}")
                return str(route.target)

        # Then check for prefix match
        for route in VALIDATED_ROUTES:
            if path.startswith(route.path):
                # For prefix matches, keep the rest of the path
                remaining_path = path[len(route.path):]
                logger.debug(f"Remaining path: {remaining_path}")
                # Make sure we don't end up with double slashes
                if remaining_path and remaining_path.startswith('/'):
                    remaining_path = remaining_path[1:]
                target = urljoin(str(route.target), remaining_path)
                logger.debug(f"Found prefix match: {path} -> {target} (using route {route.path} -> {route.target})")
                return target

        logger.warning(f"No route found for path: {path}")
        return None

    @classmethod
    def prepare_headers(cls, request: Union[Request, WebSocket], target_url: str) -> Dict[str, str]:
        """Prepare headers for the proxy request"""
        logger.debug(f"Preparing headers for {target_url}")
        headers = {}

        # Get headers from request
        logger.debug("Request headers:")
        if isinstance(request, Request):
            request_headers = request.headers
        else:  # WebSocket
            request_headers = request.headers

        # Copy preserved headers from the original request
        logger.debug("=" * 40)
        logger.debug("Preserved headers:")
        for header, value in request_headers.items():
            if header.lower() in [h.lower() for h in settings.PRESERVED_HEADERS]:
                headers[header] = value
        logger.debug("=" * 40)

        # Add endpoint-specific headers
        logger.debug("=" * 40)
        logger.debug("Endpoint headers:")
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
        logger.debug("=" * 40)

        return headers

    @classmethod
    async def forward_request(
        cls,
        request: Request,
        target_url: str,
        client: httpx.AsyncClient
    ) -> Tuple[Response, int]:
        """Forward the request to the target URL"""
        logger.debug(f"Forwarding request to {target_url} fn: forward_request")
        try:
            # Prepare headers
            headers = cls.prepare_headers(request, target_url)

            # Get request body
            body = await request.body()

            logger.debug(f"Forwarding {request.method} request to {target_url}")
            logger.debug(f"Request headers: {headers}")
            if body:
                logger.debug(f"Request body length: {len(body)} bytes")

            # Forward the request
            logger.debug(f"Sending request to {target_url}")
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
            logger.debug(f"Forwarded request to {target_url}: {response.status_code}")
            log_proxy_forward(target_url, request.method, response.status_code)

            # Create FastAPI response
            logger.debug(f"Creating FastAPI response for {target_url}")
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

    @classmethod
    async def tunnel_websocket(cls, websocket: WebSocket, target_host: str, target_port: int):
        """Create a tunnel between WebSocket and target server"""
        logger.debug(f"Creating WebSocket tunnel to {target_host}:{target_port}")
        try:
            # Connect to target server
            reader, writer = await asyncio.open_connection(target_host, target_port)

            # Create bidirectional tunnel
            logger.debug("Creating bidirectional tunnel")

            async def forward_ws_to_target():
                logger.debug("Forwarding WS to target")
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

    @classmethod
    def create_error_response(cls, status_code: int, message: str) -> Response:
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

class CopilotProxyTargetProtocol(asyncio.Protocol):
    def __init__(self, proxy: CopilotProvider):
        logger.debug("Initializing CopilotProxyTargetProtocol class: CopilotProxyTargetProtocol")
        self.proxy = proxy
        self.transport: Optional[asyncio.Transport] = None

    def connection_made(self, transport: asyncio.Transport):
        logger.debug(f"Connection made to target {self.proxy.target_host}:{self.proxy.target_port}")
        self.transport = transport
        self.proxy.target_transport = transport

    def data_received(self, data: bytes):
        logger.debug(f"Data received from target {self.proxy.target_host}:{self.proxy.target_port}")
        if self.proxy.transport and not self.proxy.transport.is_closing():
            self.proxy.log_decrypted_data(data, "Server to Client")
            self.proxy.transport.write(data)

    def connection_lost(self, exc):
        logger.debug(f"Connection lost from target {self.proxy.target_host}:{self.proxy.target_port}")
        if self.proxy.transport and not self.proxy.transport.is_closing():
            self.proxy.transport.close()
