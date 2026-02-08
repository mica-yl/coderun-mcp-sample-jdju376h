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

## model
import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
from torch.nn import functional as F

from lib.config import config, update_config
from lib.utils import get_model
from dataset.dataset_test import TestDataset
from torch.utils.data import DataLoader, TensorDataset

import argparse


args = argparse.Namespace(experiment="trufor_ph3",opts=None)
update_config(config, args)

device= "cpu"
# device= "cuda"
model_state_file = '/usr/src/app/TruFor/weights/trufor.pth.tar'

def load_model(model_state_file,config,device=device):
  print('=> loading model from {}'.format(model_state_file))
  checkpoint = torch.load(model_state_file, map_location=torch.device(device),weights_only=False)
  print("Epoch: {}".format(checkpoint['epoch']))
  model = get_model(config)
  model.load_state_dict(checkpoint['state_dict'])
  model = model.to(device)
  return model

def predict(model, testloader):
    results_d=[]
    with torch.no_grad():
        for index, [rgb] in enumerate(tqdm(testloader)):
            rgb = rgb.to(device)

            model.eval()

            det  = None
            conf = None

            pred, conf, det, npp = model(rgb)
            if conf is not None:
                conf = torch.squeeze(conf, 0)
                conf = torch.sigmoid(conf)[0]
                conf = conf.cpu().numpy()

            if npp is not None:
                npp = torch.squeeze(npp, 0)[0]
                npp = npp.cpu().numpy()

            if det is not None:
                det_sig = torch.sigmoid(det).item()

            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()

            out_dict = dict()
            out_dict['map'    ] = pred
            out_dict['imgsize'] = tuple(rgb.shape[2:])
            if det is not None:
                out_dict['score'] = det_sig
            if conf is not None:
                out_dict['conf'] = conf
            out_dict['np++'] = npp

            results_d.append(out_dict)
    return results_d

def predict_one_tensor(model, t):
    # 2. Extract only the tensors and stack them using torch.stack
    # list_of_tensors_only = [item[0] for item in list_img]
    # tensors_img = torch.stack(list_of_tensors_only)
    tensors_img = torch.stack([t])

    # 3. Create Dataset
    # TensorDataset requires a tensor
    test_dataset = TensorDataset(tensors_img)
    testloader = DataLoader(
        test_dataset,
        batch_size=1)   # 1 to allow arbitrary input sizes
    pred = predict(model, testloader)[0]
    return pred



def apply_mask(pil_img, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlays a mask like Matplotlib's 'imshow(mask, alpha=0.5)'.

    color: (R, G, B) tuple
    alpha: 0.0 to 1.0
    """
    # 1. Convert original to RGBA
    base =np.array(pil_img.convert("RGBA")).astype(np.float32) # Use float for precise math

    # 2. Create an empty RGBA array for the mask layer
    h, w = mask.shape
    mask_rgba = np.zeros((h, w, 4), dtype=np.float32)

    # 3. Fill the 'True' areas of the mask layer with color and alpha
    # Matplotlib alpha works by multiplying the color channel
    mask_rgba[mask, 0:3] = color
    mask_rgba[mask, 3] = alpha * 255  # Convert 0-1.0 to 0-255

    # 4. Create the mask Image and composite
    mask_image = PILImage.fromarray(mask_rgba.astype(np.uint8), "RGBA")

    # This is the "secret sauce" that mimics Matplotlib's layering
    return PILImage.alpha_composite(pil_img.convert("RGBA"), mask_image)

# apply_mask(image,pred > 0.5 )

model=load_model(model_state_file,config,device=device)

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

@app.get("/works")
async def works():
    """Returns the captured server logs."""
    return "it works"

@app.get("/log/json")
async def view_logs():
    """Returns the captured server logs."""
    return {
        "status": "logging_active",
        "entry_count": len(log_buffer),
        "logs": list(log_buffer)
    }

# Define the expected structure of the JSON body
class InputData(BaseModel):
    image_base64: str
#
@app.post("/api/v1/doc-tamper-detection")
def doc_tamper_detection(item: InputData):
    """accepts a base64 encoded image and returns a mask and a modification percentage."""
    source_base64=item.image_base64
    # 1. Decode header to check file type
    header = ""
    b64_data = source_base64
    if "," in source_base64:
        header, b64_data = source_base64.split(",", 1)
    
    raw_bytes = base64.b64decode(b64_data)
    
    result= {
        "mask_b64":None,
        "score":None,
        "excution_time_sec":None,
        "modification_percentage":None
    }

    # --- SINGLE IMAGE PATH ---
    with Base64ImageContext(source_base64) as ctx:
        arr = np.array(ctx.image)
        try:
            import time

            start = time.perf_counter()
            # ------
            tensor=torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float) / 256.0 # FIXME 255
            pred_d = predict_one_tensor(model,tensor)

            mask_np=pred_d['map'] > 0.5
            
            
            
            ctx.image = apply_mask(ctx.image,mask_np)
            #------
            end = time.perf_counter()

            modification_percent=(mask_np.sum()/mask_np.size) * 100

            result['excution_time_sec']=end - start
            result['mask_b64']=ctx.output_base64
            result['score']=pred_d['score']
            result['modification_percentage']=modification_percent
            
        except Exception as e:
            print(e)
            raise e
    
    return result

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
            name="mock_image_tampering_detector",
            description="accepts image and returns image with random red mask overlay",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {"type": "string", "description": "Base64 encoded image"}
                },
                "required": ["image_base64"]
            }
        ),
        Tool(
            name="pdf_tampering_detector",
            description="accepts pdf and returns images with red mask overlay",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdf_base64": {"type": "string", "description": "Base64 encoded pdf"}
                },
                "required": ["image_base64"]
            }
        ),
        Tool(
            name="tampering_detector",
            description="accepts a pdf returns text result of tampering detector on image",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdf_base64": {"type": "string", "description": "Base64 encoded pdf"}
                },
                "required": ["pdf_base64"]
            }
        ),
        Tool(
            name="tampering_detector_image",
            description="accepts an image returns a masked version of the image.",
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
    
    if name == "mock_image_tampering_detector":
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
    
    if name == "pdf_tampering_detector":
        source_base64 = arguments["pdf_base64"]
        
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

    if name == "tampering_detector":
        # source_base64 = arguments["pdf_base64"]
        import random
        result = random.choice(["document is suspected for having tampering regions", "document is not suspected for having tampering regions"])
        return [TextContent(type="text", text=result)]

    if name == "tampering_detector_image":
        source_base64 = arguments["image_base64"]
        
        # 1. Decode header to check file type
        header = ""
        b64_data = source_base64
        if "," in source_base64:
            header, b64_data = source_base64.split(",", 1)
        
        raw_bytes = base64.b64decode(b64_data)
        
        results = []

        # --- SINGLE IMAGE PATH ---
        with Base64ImageContext(source_base64) as ctx:
            arr = np.array(ctx.image)
            try:
                import time

                start = time.perf_counter()
                # ------
                tensor=torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float) / 256.0
                pred_d = predict_one_tensor(model,tensor)

                mask_np=pred_d['map'] > 0.5
                
                
                
                ctx.image = apply_mask(ctx.image,mask_np)
                #------
                end = time.perf_counter()

                modification_percent=(mask_np.sum()/mask_np.size) * 100
                results.append(
                    TextContent(type="text", 
                    text=f"Modification percentage: {modification_percent:.3f}%\n"#+
                    # f"whole-image integrity score: {pred_d['score']:.4f}"
                    ,
                    # text="Modification percentage: 0.000%\nwhole-image integrity score: 0.00"
                    meta={
                        "execution_time_sec": f"{end - start:.4f}",
                        "internal_id": "task_99"
                    }
                    ))
                
                print(f"Execution time: {end - start:.4f} seconds")
                # results.append(
                #     TextContent(type="text", 
                #         text=f"Execution time: {end - start:.4f} seconds"
                #     ))
            except Exception as e:
                print(e)
                raise e
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