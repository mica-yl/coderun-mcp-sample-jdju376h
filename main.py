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
    return {"message": "Welcome to the custom MCP Server!", "status": "running"}

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
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Executes the tool logic."""
    if name == "uppercase":
        text = arguments.get("text", "")
        result = text.upper()
        return [TextContent(type="text", text=result)]
    
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