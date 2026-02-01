import uvicorn
from fastapi import FastAPI, Request
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent

import sys
from collections import deque

import base64
import io
from PIL import Image as PILImage
import numpy as np
import fitz  # PyMuPDF

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
async def view_logs_text():
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


def apply_mock_tampering_mask(image_np, alpha=0.4):
    """
    Takes an RGB numpy array and overlays a random red 'segmentation' mask.
    
    Args:
        image_np: numpy array of shape (H, W, 3)
        alpha: Transparency of the red mask (0.0 to 1.0)
    """
    h, w, _ = image_np.shape
    
    # 1. Create a blank boolean mask
    mask = np.zeros((h, w), dtype=bool)
    
    # 2. Generate a random "tampered" region
    # We'll pick a random center and size for a 'blob'
    num_blobs = np.random.randint(1, 4)
    for _ in range(num_blobs):
        # Randomly choose a rect to represent a word or line being 'tampered'
        bh = np.random.randint(20, h // 5)
        bw = np.random.randint(50, w // 3)
        y = np.random.randint(0, h - bh)
        x = np.random.randint(0, w - bw)
        mask[y:y+bh, x:x+bw] = True

    # 3. Create an RGBA version of the original image
    # Convert to float32 (0.0-1.0) for easier math
    img_float = image_np.astype(np.float32) / 255.0
    
    # 4. Prepare the Red Mask Overlay
    # We blend: (1 - alpha) * original + (alpha) * pure_red
    red_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # Apply the blend only where mask is True
    result = img_float.copy()
    result[mask] = (1 - alpha) * img_float[mask] + alpha * red_color
    
    # 5. Convert back to uint8 (0-255)
    return (result * 255).astype(np.uint8)

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
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    """Executes the tool logic."""
    if name == "uppercase":
        text = arguments.get("text", "")
        result = text.upper()
        return [TextContent(type="text", text=result)]
    
    if name == "mock_tampering_detector":
        source_base64 = arguments["image_base64"]
        
        # 1. Decode header to check file type
        header = ""
        b64_data = source_base64
        if "," in source_base64:
            header, b64_data = source_base64.split(",", 1)
        
        raw_bytes = base64.b64decode(b64_data)
        
        # 2. Check if it's a PDF (PDF files start with %PDF)
        is_pdf = raw_bytes.startswith(b"%PDF")
        
        results = []
        
        if is_pdf:
            # --- PDF PROCESSING PATH ---
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            for page in doc:
                # Render page to a high-res image (RGB)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for clarity
                img_pil = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Apply mask using NumPy
                arr = np.array(img_pil)
                # ... (reuse your tampering logic here) ...
                processed_arr = apply_mock_tampering_mask(arr) # Your helper function
                
                # Encode back to Base64
                out_img = PILImage.fromarray(processed_arr)
                buf = io.BytesIO()
                out_img.save(buf, format="PNG")
                results.append(ImageContent(
                    type="image", 
                    data=base64.b64encode(buf.getvalue()).decode(), 
                    mimeType="image/png"
                ))
            doc.close()
        else:
            # --- SINGLE IMAGE PATH ---
            with Base64ImageContext(source_base64) as ctx:
                arr = np.array(ctx.image)
                processed_arr = apply_mock_tampering_mask(arr)
                ctx.image = PILImage.fromarray(processed_arr)
            results.append(ImageContent(type="image", data=ctx.output_base64, mimeType="image/png"))

        return results

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