import uvicorn
from fastapi import FastAPI, Request
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp.types import Tool, TextContent

import sys
from collections import deque

# We use a deque with maxlen to keep only the last 1000 lines
log_buffer = deque(maxlen=1000)

class StreamTee:
    """Intercepts writes and sends them to both the original stream and our buffer."""
    def __init__(self, original_stream):
        self.original = original_stream

    def write(self, message):
        self.original.write(message)  # Print to actual console
        if message.strip():           # Don't save empty newlines
            log_buffer.append(message)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        """
        Delegates all other attributes (isatty, fileno, encoding, etc.) 
        to the original stream.
        """
        return getattr(self.original, name)

# Redirect stdout and stderr immediately
sys.stdout = StreamTee(sys.stdout)
sys.stderr = StreamTee(sys.stderr)

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
    """Returns a welcome message that lists all available tools dynamically."""
    available_tools = await list_tools()
    
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

@app.get("/log")
async def view_logs():
    """Returns the captured server logs."""
    return "\n".join(log_buffer)

@app.get("/log/json")
async def view_logs():
    """Returns the captured server logs."""
    return {
        "status": "logging_active",
        "entry_count": len(log_buffer),
        "logs": list(log_buffer)
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

import base64
import io
from PIL import Image as PILImage

class Base64ImageContext:
    def __init__(self, base64_string: str, format: str = "PNG"):
        """
        Args:
            base64_string: The input image string (with or without header).
            format: Output format for re-encoding (default "PNG" for lossless).
        """
        self.input_str = base64_string
        self.format = format
        self.image = None  # The PIL Image object available in the 'with' block
        self.output_base64 = None  # The final result string after the block

    def __enter__(self):
        # 1. Decode Logic
        b64_data = self.input_str
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
            
        image_data = base64.b64decode(b64_data)
        self.image = PILImage.open(io.BytesIO(image_data)).convert("RGB")
        
        # Return self so user can access ctx.image
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # If an error occurred inside the block, do not try to encode
        if exc_type:
            return False

        # 2. Encode Logic (using the modified self.image)
        buffered = io.BytesIO()
        self.image.save(buffered, format=self.format)
        self.output_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Executes the tool logic."""
    if name == "uppercase":
        text = arguments.get("text", "")
        result = text.upper()
        return [TextContent(type="text", text=result)]
    
    if name == "mock_tampering_detector":
        image_base64 = arguments["image_base64"]
        
        with Base64ImageContext(raw_b64) as ctx:
            # 'ctx.image' is a standard PIL object ready for use
            
            # --- Your Custom Logic (Clean & Readable) ---
            arr = np.array(ctx.image)
            
            # Add 40 to Red channel (with overflow protection)
            arr[:, :, 0] = np.clip(arr[:, :, 0].astype(np.int16) + 40, 0, 255).astype(np.uint8)
            
            # Update the context's image with your result
            ctx.image = PILImage.fromarray(arr)
            # --------------------------------------------
        
        return [ImageContent(type="image", data=ctx.output_base64, mimeType="image/png")]
    
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