import uvicorn
from fastapi import FastAPI, Request
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp.types import Tool, TextContent

# 1. Initialize the Web Server (FastAPI) and MCP Server
app = FastAPI(title="My MCP Server")
mcp_server = Server("demo-server")
sse_transport = SseServerTransport("/messages")

# --- PART A: Standard Web Endpoints ---

@app.get("/")
async def root():
    """Root endpoint: Returns a plain text welcome message."""
    return "Welcome to the MCP Server! Visit /welcome for JSON or connect via MCP."

@app.get("/welcome")
async def welcome_json():
    """Welcome endpoint: Returns a JSON response."""
    return {
        "message": "Welcome to the custom MCP Server!", 
        "status": "running",
        "tools": [
            {
                "name": t.name, 
                "description": t.description,
                # Optional: Include inputs if you want clients to see usage syntax
                # "args": t.inputSchema
            } 
            for t in available_tools
        ]
    }

# --- PART B: MCP Tool Logic ---

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """Defines the tools available to the LLM."""
    return [
        Tool(
            name="uppercase",
            description="Converts input text to ALL CAPS.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to convert"}
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="mock_tampering_detector",
            description="Adds +1 to the Red channel of an image to simulate tampering/watermarking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {"type": "string", "description": "Base64 encoded image"}
                },
                "required": ["image_base64"]
            }
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Executes the tool logic."""
    if name == "uppercase":
        text = arguments.get("text", "")
        result = text.upper()
        return [TextContent(type="text", text=result)]
    
    if name == "mock_tampering_detector":
        image_base64 = arguments["image_base64"]
        
        # 1. Decode
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        pil_img = PILImage.open(io.BytesIO(image_data)).convert("RGB")
        
        # 2. Modify Red Channel (Using NumPy for speed/overflow protection)
        # We convert to int16 first so 255 + 40 doesn't wrap to 0, then clip.
        arr = np.array(pil_img)
        arr[:, :, 0] = np.clip(arr[:, :, 0].astype(np.int16) + 40, 0, 255).astype(np.uint8)
        
        # 3. Save as PNG (Critical: PNG is lossless)
        result_img = PILImage.fromarray(arr)
        buffered = io.BytesIO()
        result_img.save(buffered, format="PNG")
        
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return [ImageContent(type="image", data=img_str, mimeType="image/png")]
    
    raise ValueError(f"Unknown tool: {name}")

# --- PART C: MCP Protocol Wiring (SSE) ---

@app.get("/sse")
async def handle_sse(request: Request):
    """The entry point for MCP clients (like Claude) to connect."""
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp_server.run(
            streams[0], streams[1], mcp_server.create_initialization_options()
        )

@app.post("/messages")
async def handle_messages(request: Request):
    """Handles the JSON-RPC messages sent by the client."""
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )

# --- Entry Point ---
if __name__ == "__main__":
    # Run on localhost:8080
    uvicorn.run(app, host="0.0.0.0", port=8080)