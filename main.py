import base64
import io
import uvicorn
from typing import Any
from fastapi import FastAPI, Request
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from PIL import Image as PILImage, ImageOps

# 1. Initialize FastAPI and MCP Server
app = FastAPI()
mcp_server = Server("remote-image-filter")

# 2. Define the SSE Transport (Global variable to hold the connection)
sse_transport = None

@app.get("/sse")
async def handle_sse(request: Request):
    """Client connects here to start the session."""
    import mcp.server.sse
    
    # Create a new transport for this connection
    transport = mcp.server.sse.SseServerTransport("/messages")
    
    # Start the server session using this transport
    # Note: In a real production app, you'd manage multiple sessions.
    # For this scaffold, we attach the server to this transport.
    
    async def run_session():
        await mcp_server.run(
            transport.read_incoming(),
            transport.write_outgoing(),
            mcp_server.create_initialization_options()
        )

    # Use Starlette/FastAPI's EventSourceResponse (or equivalent wrapper)
    # The mcp library provides a helper for this usually, but here is the manual hookup:
    return await transport.connect_sse(request, run_session)

@app.post("/messages")
async def handle_messages(request: Request):
    """Client sends requests (like 'call_tool') here."""
    # This endpoint receives the JSON-RPC messages and pipes them to the active transport
    # In a full implementation, you map this to the specific session ID.
    # For this simple scaffold, we assume the transport created in /sse handles it.
    
    # Note: The 'mcp' SDK's SseServerTransport handles the logic, 
    # we just need to ensure the transport object is accessible or the library handles the routing.
    # A cleaner pattern often uses the `starlette-sse` or specific MCP integration.
    pass 
    # *Simplified implementation below using the high-level MCP-FastAPI pattern*

# --- REWRITE: The Cleanest "Remote" Implementation using Starlette/FastAPI ---
# The logic above is complex to wire manually. 
# Below is the production-ready way using the 'mcp' SDK's built-in SSE support.

from starlette.applications import Starlette
from starlette.routing import Route
from mcp.server.sse import SseServerTransport

# Redefine server with the tools
@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="process_image",
            description="Applies grayscale, invert, or rotate to a base64 image.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {"type": "string", "description": "Base64 encoded image"},
                    "operation": {"type": "string", "enum": ["grayscale", "invert", "rotate"]}
                },
                "required": ["image_base64"]
            }
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    if name != "process_image":
        raise ValueError(f"Unknown tool: {name}")

    image_base64 = arguments["image_base64"]
    operation = arguments.get("operation", "grayscale")

    # --- Image Logic (Same as before) ---
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        pil_img = PILImage.open(io.BytesIO(image_data))
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

    if operation == "grayscale":
        result_img = ImageOps.grayscale(pil_img)
    elif operation == "invert":
        result_img = ImageOps.invert(pil_img.convert("RGB"))
    elif operation == "rotate":
        result_img = pil_img.rotate(90, expand=True)
    else:
        result_img = pil_img

    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return [ImageContent(type="image", data=img_str, mimeType="image/png")]

# --- The Server Entry Point ---
sse = SseServerTransport("/messages")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(
            streams[0], streams[1], mcp_server.create_initialization_options()
        )

async def handle_messages(request):
    await sse.handle_post_message(request.scope, request.receive, request._send)

starlette_app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages", endpoint=handle_messages, methods=["POST"]),
    ]
)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=8080)