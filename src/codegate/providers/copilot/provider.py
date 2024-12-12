import asyncio
import json
import re
import ssl
from typing import Dict, Optional, Tuple
from urllib.parse import unquote, urljoin, urlparse

import structlog
from litellm.types.utils import ModelResponse

from codegate.ca.codegate_ca import CertificateAuthority
from codegate.config import Config
from codegate.pipeline.base import PipelineContext
from codegate.pipeline.factory import PipelineFactory
from codegate.pipeline.output import OutputPipelineInstance
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.providers.copilot.mapping import VALIDATED_ROUTES
from codegate.providers.normalizer.completion import CompletionNormalizer

logger = structlog.get_logger("codegate")

# Increase buffer sizes
MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 64 * 1024  # 64KB


class SSEProcessor:
    def __init__(self):
        self.buffer = ""
        self.initial_chunk = True
        self.chunk_size = None  # Store the original chunk size
        self.size_written = False

    def process_chunk(self, chunk: bytes) -> list:
        # Skip any chunk size lines (hex number followed by \r\n)
        try:
            chunk_str = chunk.decode('utf-8')
            lines = chunk_str.split('\r\n')
            # Skip lines that look like chunk sizes (hex numbers)
            data_lines = [line for line in lines if not all(c in '0123456789abcdefABCDEF' for c in line.strip())]
            self.buffer += '\n'.join(data_lines)
        except UnicodeDecodeError:
            print(f"Failed to decode chunk")

        records = []
        while True:
            record_end = self.buffer.find('\n\n')
            if record_end == -1:
                print(f"REMAINING BUFFER {self.buffer}")
                break

            record = self.buffer[:record_end]
            self.buffer = self.buffer[record_end + 2:]

            if record.startswith('data: '):
                data_content = record[6:]
                if data_content.strip() == '[DONE]':
                    records.append({'type': 'done'})
                else:
                    try:
                        data = json.loads(data_content)
                        records.append({'type': 'data', 'content': data})
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON: {data_content}")

        return records

    def get_pending(self):
        """Return any pending incomplete data in the buffer"""
        return self.buffer

class SSEStreamIterator:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.done = False

    def add_record(self, record: dict):
        if not self.done:
            self.queue.put_nowait(record)
            if record['type'] == 'done':
                self.done = True

    async def __anext__(self):
        if self.done and self.queue.empty():
            raise StopAsyncIteration
        
        record = await self.queue.get()
        if record['type'] == 'done':
            return ModelResponse(type='done')
        return ModelResponse(**record['content'])

    def __aiter__(self):
        return self

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
        self.proxy_ep = None
        self.decrypted_data = bytearray()
        # Get the singleton instance of CertificateAuthority
        self.ca = CertificateAuthority.get_instance()
        self.pipeline_factory = PipelineFactory(SecretsManager())
        self.context_tracking: Optional[PipelineContext] = None

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
            model = "model_name",   # Replace with actual model if needed
            api_key = None,         # Extract from headers if needed
        )

        # Use the modified request body from the pipeline result
        if result.request:
            normalized_json_body = normalizer.denormalize(result.request)
            body = json.dumps(normalized_json_body).encode()

        logger.debug("Processing body through pipeline")
        return body

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
        self.peername = transport.get_extra_info("peername")
        logger.debug(f"Client connected from {self.peername}")

    def extract_path(self, full_path: str) -> str:
        logger.debug(f"Extracting path from {full_path} fn: extract_path")
        if full_path.startswith("http://") or full_path.startswith("https://"):
            parsed = urlparse(full_path)
            path = parsed.path
            if parsed.query:
                path = f"{path}?{parsed.query}"
            return path.lstrip("/")
        elif full_path.startswith("/"):
            return full_path.lstrip("/")
        return full_path

    def get_headers(self, buffer: bytes) -> Dict[str, str]:
        """Get request headers as a dictionary"""
        logger.debug("Getting headers as dictionary fn: get_headers")
        headers_dict = {}

        try:
            if b"\r\n\r\n" not in self.buffer:
                return {}

            headers_end = self.buffer.index(b"\r\n\r\n")
            headers = self.buffer[:headers_end].split(b"\r\n")[1:]  # Skip request line

            for header in headers:
                try:
                    name, value = header.decode("utf-8").split(":", 1)
                    headers_dict[name.strip().lower()] = value.strip()
                except ValueError:
                    continue

            return headers_dict
        except Exception as e:
            logger.error(f"Error getting headers: {e}")
            return {}

    def parse_headers(self) -> bool:
        logger.debug("Parsing headers fn: parse_headers")
        try:
            if b"\r\n\r\n" not in self.buffer:
                return False

            headers_end = self.buffer.index(b"\r\n\r\n")
            headers = self.buffer[:headers_end].split(b"\r\n")

            request = headers[0].decode("utf-8")
            self.method, full_path, self.version = request.split(" ")

            self.original_path = full_path

            if self.method == "CONNECT":
                logger.debug(f"CONNECT request to {full_path}")
                self.target = full_path
                self.path = ""
            else:
                logger.debug(f"Request: {self.method} {full_path} {self.version}")
                self.path = self.extract_path(full_path)

            self.headers = [header.decode("utf-8") for header in headers[1:]]
            return True
        except Exception as e:
            logger.error(f"Error parsing headers: {e}")
            return False

    def log_decrypted_data(self, data: bytes, direction: str):
        """
        Uncomment to log data from payload
        """
        try:
            # decoded = data.decode('utf-8')
            # logger.debug(f"=== Decrypted {direction} Data ===")
            # logger.debug(decoded)
            # logger.debug("=" * 40)
            pass
        except UnicodeDecodeError:
            # pass
            # logger.debug(f"=== Decrypted {direction} Data (hex) ===")
            # logger.debug(data.hex())
            # logger.debug("=" * 40)
            pass

    async def handle_http_request(self):
        logger.debug("Handling HTTP request fn: handle_http_request")
        logger.debug("=" * 40)
        logger.debug(f"Method: {self.method}")
        logger.debug(f"Searched Path: {self.path} in target URL")
        try:
            # Extract proxy endpoint from authorization header if present
            headers_dict = self.get_headers()
            auth_header = headers_dict.get("authorization", "")
            if auth_header:
                match = re.search(r"proxy-ep=([^;]+)", auth_header)
                if match:
                    self.proxy_ep = match.group(1)
                    logger.debug(f"Extracted proxy-ep value: {self.proxy_ep}")

                    # Check if the proxy_ep includes a scheme
                    parsed_proxy_ep = urlparse(self.proxy_ep)
                    if not parsed_proxy_ep.scheme:
                        # Default to https if no scheme is provided
                        self.proxy_ep = f"https://{self.proxy_ep}"
                        logger.debug(f"Added default scheme to proxy-ep: {self.proxy_ep}")

                    target_url = f"{self.proxy_ep}/{self.path}"
                else:
                    target_url = await self.get_target_url(self.path)
            else:
                target_url = await self.get_target_url(self.path)

            if not target_url:
                self.send_error_response(404, b"Not Found")
                return
            logger.debug(f"Target URL: {target_url}")

            parsed_url = urlparse(target_url)
            logger.debug(f"Parsed URL {parsed_url}")

            self.target_host = parsed_url.hostname
            self.target_port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)

            target_protocol = CopilotProxyTargetProtocol(self)
            logger.debug(f"Connecting to {self.target_host}:{self.target_port}")
            await self.loop.create_connection(
                lambda: target_protocol,
                self.target_host,
                self.target_port,
                ssl=parsed_url.scheme == "https",
            )

            has_host = False
            new_headers = []

            for header in self.headers:
                if header.lower().startswith("host:"):
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

    def _check_buffer_size(self, new_data: bytes) -> bool:
        """Check if adding new data would exceed the maximum buffer size"""
        return len(self.buffer) + len(new_data) <= MAX_BUFFER_SIZE

    def _handle_parsed_headers(self) -> None:
        """Handle the request based on parsed headers"""
        if self.method == "CONNECT":
            logger.debug("Handling CONNECT request")
            self.handle_connect()
        else:
            logger.debug("Handling HTTP request")
            asyncio.create_task(self.handle_http_request())

    async def _forward_data_through_pipeline(self, data: bytes):
        if b'\r\n\r\n' not in data:
            # not HTTP, pass as-is
            return data

        headers_end = data.index(b'\r\n\r\n')
        headers = data[:headers_end].split(b'\r\n')

        request = headers[0].decode('utf-8')
        method, full_path, version = request.split(' ')
        body_start = data.index(b'\r\n\r\n') + 4
        body = data[body_start:]

        if method != 'POST' and full_path != "/chat/completions":
            # not completion, forward as-is
            return data

        headers_dict = self.get_headers(data)
        request_id = headers_dict.get("x-request-id", "fake-prompt-id")

        pipeline = self.pipeline_factory.create_input_pipeline()
        json_body = json.loads(body)
        result = await pipeline.process_request(
            request = json_body,
            provider = "github-copilot",
            prompt_id = request_id,
            model = "model_name",   # Replace with actual model if needed
            api_key = None,
        )

        self.context_tracking = result.context

        # Use the modified request body from the pipeline result
        if result.request:
            body = json.dumps(result.request).encode()

        # Modify headers to update content length
        new_headers = []
        for header in headers[1:]:  # Skip the request line
            if not header.lower().startswith(b"content-length:"):
                new_headers.append(header)
        new_headers.append(f"Content-Length: {len(body)}".encode())

        # Reconstruct the full request
        request_line = headers[0]  # Original request line
        header_block = b'\r\n'.join([request_line] + new_headers)
        modified_request = header_block + b'\r\n\r\n' + body

        return modified_request

    async def _forward_data_to_target(self, data: bytes) -> None:
        """Forward data to target if connection is established"""
        if self.target_transport and not self.target_transport.is_closing():
            self.log_decrypted_data(data, "Client to Server")
            data = await self._forward_data_through_pipeline(data)
            self.target_transport.write(data)

    def data_received(self, data: bytes) -> None:
        """Handle received data from the client"""
        logger.debug(f"Data received from {self.peername} fn: data_received")

        try:
            # Check buffer size limit
            if not self._check_buffer_size(data):
                logger.error("Request exceeds maximum buffer size")
                self.send_error_response(413, b"Request body too large")
                return

            # Append new data to buffer
            self.buffer.extend(data)

            if not self.headers_parsed:
                # Try to parse headers
                self.headers_parsed = self.parse_headers()
                if not self.headers_parsed:
                    return

                # Handle the request based on parsed headers
                self._handle_parsed_headers()
            else:
                # Forward data to target if headers are already parsed
                asyncio.create_task(self._forward_data_to_target(data))

        except asyncio.CancelledError:
            logger.warning("Operation cancelled")
            raise
        except Exception as e:
            logger.error(f"Error processing received data: {e}")
            self.send_error_response(502, str(e).encode())

    def handle_connect(self):
        """
        This where requests are sent directly via the tunnel created during
        a CONNECT request. This is where the SSL context is created and the
        internal connection is made to the target host.

        We do not need to do a URL to mapping, as this passes through the
        tunnel with a FQDN already set by the source (client) request.
        """
        try:
            path = unquote(self.target)
            if ":" in path:
                self.target_host, port = path.split(":")
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
            502: "Bad Gateway",
        }
        return status_texts.get(status, "Error")

    async def connect_to_target(self):
        logger.debug(
            f"Connecting to target {self.target_host}:{self.target_port} fn: connect_to_target"
        )
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
                lambda: target_protocol, self.target_host, self.target_port, ssl=target_ssl_context
            )

            logger.debug(f"Successfully connected to {self.target_host}:{self.target_port}")

            # Send 200 Connection Established
            if self.transport and not self.transport.is_closing():
                logger.debug("Sending 200 Connection Established response")
                self.transport.write(
                    b"HTTP/1.1 200 Connection Established\r\n"
                    b"Proxy-Agent: ProxyPilot\r\n"
                    b"Connection: keep-alive\r\n\r\n"
                )

                # Upgrade client connection to SSL
                logger.debug("Upgrading client connection to SSL")
                transport = await self.loop.start_tls(
                    self.transport, self, self.ssl_context, server_side=True
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
    async def create_proxy_server(
        cls, host: str, port: int, ssl_context: Optional[ssl.SSLContext] = None
    ):
        logger.debug(f"Creating proxy server on {host}:{port} fn: create_proxy_server")
        loop = asyncio.get_event_loop()

        def create_protocol():
            logger.debug("Creating protocol for proxy server fn: create_protocol")
            return cls(loop)

        logger.debug(f"Starting proxy server on {host}:{port}")
        server = await loop.create_server(
            create_protocol, host, port, ssl=ssl_context, reuse_port=True, start_serving=True
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
                Config.get_config().host, Config.get_config().proxy_port, ssl_context
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
            # For prefix matches, keep the rest of the path
            remaining_path = path[len(route.path) :]
            logger.debug(f"Remaining path: {remaining_path}")
            # Make sure we don't end up with double slashes
            if remaining_path and remaining_path.startswith("/"):
                remaining_path = remaining_path[1:]
            target = urljoin(str(route.target), remaining_path)
            logger.debug(
                f"Found prefix match: {path} -> {target} "
                "(using route {route.path} -> {route.target})"
            )
            return target

        logger.warning(f"No route found for path: {path}")
        return None


class CopilotProxyTargetProtocol(asyncio.Protocol):
    def __init__(self, proxy: CopilotProvider):
        logger.debug("Initializing CopilotProxyTargetProtocol class: CopilotProxyTargetProtocol")
        self.proxy = proxy
        self.transport: Optional[asyncio.Transport] = None
        self.sse_processor = None
        self.headers_sent = False
        self.response_queue = asyncio.Queue()
        self.stream_task = None
        self.stream_iterator = None

        # Debug buffers
        self.raw_buffer = bytearray()
        self.processed_buffer = bytearray()
        self.output_pipeline_instance = None

    def _process_chunk(self, chunk: bytes):
        if not self.sse_processor:
            self._proxy_transport_write(chunk)
            return

        if not self.stream_iterator:
            self.stream_iterator = SSEStreamIterator()
            self.stream_task = asyncio.create_task(self._stream_processor())

        records = self.sse_processor.process_chunk(chunk)
        for record in records:
            self.stream_iterator.add_record(record)

    async def _stream_processor(self):
        async for processed_response in self.process_stream(self.stream_iterator):
            if processed_response.type == 'done':
                sse_data = b'data: [DONE]\n\n'
            else:
                sse_data = f"data: {json.dumps(processed_response.dict())}\n\n".encode('utf-8')
                
            chunk_size = hex(len(sse_data))[2:] + '\r\n'
            self._proxy_transport_write(chunk_size.encode())
            self._proxy_transport_write(sse_data)
            self._proxy_transport_write(b'\r\n')
            
            if processed_response.type == 'done':
                self._proxy_transport_write(b'0\r\n\r\n')

    def _proxy_transport_write(self, data: bytes):
        self.processed_buffer.extend(data)
        self.proxy.transport.write(data)
        print("------ WRITE DATA START")
        print(data)
        print("------ WRITE DATA END")

    def connection_made(self, transport: asyncio.Transport):
        logger.debug(f"Connection made to target {self.proxy.target_host}:{self.proxy.target_port}")
        self.transport = transport
        self.proxy.target_transport = transport

    def data_received(self, data: bytes):
        logger.debug(f"Data received from target {self.proxy.target_host}:{self.proxy.target_port}")

        self.raw_buffer.extend(data)  # Store original incoming data
        with open("raw_buffer.txt", "wb") as f:
            f.write(self.raw_buffer)

        if self.proxy.context_tracking is not None and self.sse_processor is None:
            logger.debug("Tracking context for pipeline processing")
            self.sse_processor = SSEProcessor()
            out_pipeline_processor = self.proxy.pipeline_factory.create_output_pipeline()
            self.output_pipeline_instance = OutputPipelineInstance(
                pipeline_steps=out_pipeline_processor.pipeline_steps,
                input_context=self.proxy.context_tracking,
            )

        if self.proxy.transport and not self.proxy.transport.is_closing():
            self.proxy.log_decrypted_data(data, "Server to Client")

            # DEBUGGING
            #self.proxy.transport.write(data)

            print("------ TARGET DATA START")
            print(data)
            print("------ TARGET DATA END")

            if not self.sse_processor:
                # Pass through non-SSE data unchanged
                logger.debug("No context tracked, pass through")
                print("------ WRITE DATA START")
                print(data)
                print("------ WRITE DATA END")
                self._proxy_transport_write(data)
                return

            # Check if this is the first chunk with headers
            if not self.headers_sent:
                header_end = data.find(b'\r\n\r\n')
                if header_end != -1:
                    self.headers_sent = True
                    # Send headers first
                    headers = data[:header_end + 4]
                    print("------ WRITE HEADERS START")
                    print(headers)
                    print("------ WRITE HEADERS END")
                    self._proxy_transport_write(headers)
                    logger.debug(f"Headers sent: {headers}")

                    data = data[header_end + 4:]
                    self.stream_task = asyncio.create_task(self._stream_processor())

            self._process_chunk(data)

    def connection_lost(self, exc):
        logger.debug(
            f"Connection lost from target {self.proxy.target_host}:{self.proxy.target_port}"
        )
        if self.proxy.transport and not self.proxy.transport.is_closing():
            self.proxy.transport.close()

        # TODO: clear the pipeline
