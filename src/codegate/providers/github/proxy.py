from typing import Optional, Dict, Tuple, Union
from urllib.parse import urljoin, urlparse
from fastapi import Request, Response, WebSocket
import httpx
import asyncio
from ..config.settings import settings, VALIDATED_ROUTES
from ..core.logging import log_proxy_forward, log_error, logger

async def get_target_url(path: str) -> Optional[str]:
    """Get target URL for the given path"""
    logger.debug(f"Attempting to get target URL for path: {path}")

    # Special handling for Copilot completions endpoint
    if '/v1/engines/copilot-codex/completions' in path:
        logger.debug("Using special case for completions endpoint")
        return 'https://proxy.individual.githubcopilot.com/v1/engines/copilot-codex/completions'
    
    # Check for exact path match first
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
