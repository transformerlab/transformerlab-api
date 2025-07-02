from fastapi import APIRouter, Request, Response, WebSocket, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import asyncio
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plugin_ui", tags=["plugin_ui"])

# Target server configuration
# Configuration that can be overridden via environment variables
TARGET_HOST = os.getenv("PLUGIN_UI_TARGET_HOST", "localhost")
TARGET_PORT = int(os.getenv("PLUGIN_UI_TARGET_PORT", "8339"))
TARGET_BASE_URL = f"http://{TARGET_HOST}:{TARGET_PORT}"

# HTTP client for proxying requests
client = httpx.AsyncClient(timeout=30.0)


@router.get("/", summary="Plugin UI Proxy Root")
async def plugin_ui_root():
    """fetch and return whatver is at localhost:8339/"""
    async with httpx.AsyncClient() as client:
        response = await client.get(TARGET_BASE_URL)
        return Response(
            content=response.content, status_code=response.status_code, media_type=response.headers.get("content-type")
        )


@router.api_route("/{name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_http_request(name: str, request: Request):
    """
    Proxy HTTP requests to the Gradio server.
    Maps /plugin_ui/<name> to localhost:8339/<name>
    """
    try:
        # Construct target URL
        target_url = f"{TARGET_BASE_URL}/{name}"
        print(f"Proxying request to: {target_url}")

        # Get query parameters
        query_params = str(request.query_params) if request.query_params else ""
        if query_params:
            target_url += f"?{query_params}"

        # Prepare headers (exclude host-specific headers)
        headers = dict(request.headers)
        headers_to_remove = ["host", "content-length", "transfer-encoding"]
        for header in headers_to_remove:
            headers.pop(header, None)

        # Add X-Forwarded headers for proper proxying
        headers["X-Forwarded-For"] = request.client.host if request.client else "unknown"
        headers["X-Forwarded-Proto"] = request.url.scheme
        headers["X-Forwarded-Host"] = request.headers.get("host", "unknown")

        # Get request body
        body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None

        # Make the proxied request
        response = await client.request(
            method=request.method, url=target_url, headers=headers, content=body, follow_redirects=False
        )

        # Prepare response headers
        response_headers = dict(response.headers)

        # Remove headers that should not be forwarded
        headers_to_remove = ["content-encoding", "transfer-encoding", "content-length"]
        for header in headers_to_remove:
            response_headers.pop(header, None)

        # Handle redirects by rewriting location headers
        if "location" in response_headers:
            location = response_headers["location"]
            if location.startswith(TARGET_BASE_URL):
                # Rewrite absolute URLs pointing to target server
                response_headers["location"] = location.replace(TARGET_BASE_URL, "/plugin_ui")
            elif location.startswith("/"):
                # Rewrite relative URLs
                response_headers["location"] = f"/plugin_ui{location}"

        # For HTML content, we might need to rewrite URLs in the content
        content_type = response_headers.get("content-type", "")
        if "text/html" in content_type:
            content = response.content.decode("utf-8")
            # Rewrite common Gradio URL patterns
            content = rewrite_gradio_urls(content)

            return Response(
                content=content, status_code=response.status_code, headers=response_headers, media_type=content_type
            )

        # For non-HTML content, stream the response
        return StreamingResponse(
            iter([response.content]),
            status_code=response.status_code,
            headers=response_headers,
            media_type=response_headers.get("content-type"),
        )

    except httpx.ConnectError:
        logger.error(f"Failed to connect to target server at {TARGET_BASE_URL}")
        raise HTTPException(status_code=502, detail="Bad Gateway: Unable to connect to target server")
    except httpx.TimeoutException:
        logger.error(f"Timeout when connecting to {target_url}")
        raise HTTPException(status_code=504, detail="Gateway Timeout")
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal proxy error")


@router.websocket("/{name:path}")
async def proxy_websocket(websocket: WebSocket, name: str):
    """
    Proxy WebSocket connections to the Gradio server.
    This is essential for Gradio's real-time functionality.
    """
    await websocket.accept()

    # Construct target WebSocket URL
    target_ws_url = f"ws://{TARGET_HOST}:{TARGET_PORT}/{name}"

    try:
        import websockets

        # Connect to target WebSocket
        async with websockets.connect(target_ws_url) as target_ws:
            # Create tasks for bidirectional data flow
            async def forward_to_target():
                try:
                    while True:
                        data = await websocket.receive()
                        if data["type"] == "websocket.disconnect":
                            break
                        elif data["type"] == "websocket.receive":
                            if "text" in data:
                                await target_ws.send(data["text"])
                            elif "bytes" in data:
                                await target_ws.send(data["bytes"])
                except Exception as e:
                    logger.error(f"Error forwarding to target: {e}")

            async def forward_to_client():
                try:
                    async for message in target_ws:
                        if isinstance(message, str):
                            await websocket.send_text(message)
                        elif isinstance(message, bytes):
                            await websocket.send_bytes(message)
                except Exception as e:
                    logger.error(f"Error forwarding to client: {e}")

            # Run both forwarding tasks concurrently
            await asyncio.gather(forward_to_target(), forward_to_client(), return_exceptions=True)

    except ImportError:
        logger.error("websockets library not available. WebSocket proxying disabled.")
        await websocket.close(code=1002, reason="WebSocket proxy not available")
    except Exception as e:
        logger.error(f"WebSocket proxy error: {str(e)}")
        await websocket.close(code=1011, reason="Internal server error")


def rewrite_gradio_urls(html_content: str) -> str:
    """
    Rewrite URLs in HTML content to point to the proxy instead of the original server.
    This handles common Gradio URL patterns.
    """
    import re

    # Rewrite absolute URLs pointing to the target server
    html_content = re.sub(rf"http://{TARGET_HOST}:{TARGET_PORT}/", "/plugin_ui/", html_content)

    # Rewrite relative URLs that might be used by Gradio
    # This pattern looks for common Gradio assets and API endpoints
    gradio_patterns = [
        (r'src="/', r'src="/plugin_ui/'),
        (r'href="/', r'href="/plugin_ui/'),
        (r'action="/', r'action="/plugin_ui/'),
        (r'url\("/', r'url("/plugin_ui/'),
        (r"url\('/'", r"url('/plugin_ui/"),
        # Gradio specific patterns
        (r'"/api/', r'"/plugin_ui/api/'),
        (r"'/api/", r"'/plugin_ui/api/"),
        (r'"/queue/', r'"/plugin_ui/queue/'),
        (r"'/queue/", r"'/plugin_ui/queue/"),
        (r'"/file/', r'"/plugin_ui/file/'),
        (r"'/file/", r"'/plugin_ui/file/"),
    ]

    for pattern, replacement in gradio_patterns:
        html_content = re.sub(pattern, replacement, html_content)

    return html_content


# @router.on_event("shutdown")
# async def shutdown_event():
#     """Clean up the HTTP client on shutdown."""
#     await client.aclose()


# Health check endpoint
@router.get("/health")
async def health_check():
    """Check if the proxy and target server are healthy."""
    try:
        response = await client.get(f"{TARGET_BASE_URL}/health", timeout=5.0)
        return {"status": "healthy", "target_server": "connected", "target_status": response.status_code}
    except Exception as e:
        return {"status": "unhealthy", "target_server": "disconnected", "error": str(e)}
