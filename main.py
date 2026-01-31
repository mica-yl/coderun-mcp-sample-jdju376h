import base64
import io
from typing import Annotated
from mcp.server.fastmcp import FastMCP
from mcp.types import Image
from PIL import Image as PILImage, ImageOps

# 1. Initialize the FastMCP server
mcp = FastMCP("ImageFilterServer")

@mcp.tool()
def process_image(
    image_base64: str, 
    operation: str = "grayscale"
) -> Image:
    """
    Takes a base64 encoded image, applies a transformation, and returns the new image.
    
    Args:
        image_base64: The raw image data encoded as a base64 string.
        operation: The operation to perform ('grayscale', 'invert', 'rotate').
    """
    
    # --- Step A: Decode Input ---
    try:
        # Remove header if present (e.g., "data:image/png;base64,...")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
            
        image_data = base64.b64decode(image_base64)
        pil_img = PILImage.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

    # --- Step B: Image Processing Logic ---
    # (This is where you would plug in your mask function from before)
    if operation == "grayscale":
        result_img = ImageOps.grayscale(pil_img)
    elif operation == "invert":
        # Invert requires RGB mode usually
        if pil_img.mode == 'RGBA':
            r,g,b,a = pil_img.split()
            rgb_image = PILImage.merge('RGB', (r,g,b))
            inverted = ImageOps.invert(rgb_image)
            r2,g2,b2 = inverted.split()
            result_img = PILImage.merge('RGBA', (r2,g2,b2,a))
        else:
            result_img = ImageOps.invert(pil_img.convert("RGB"))
    elif operation == "rotate":
        result_img = pil_img.rotate(90, expand=True)
    else:
        result_img = pil_img  # No-op

    # --- Step C: Encode Output ---
    buffered = io.BytesIO()
    # Save as PNG to preserve transparency/quality
    result_img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    encoded_str = base64.b64encode(img_bytes).decode("utf-8")

    # --- Step D: Return MCP Image Type ---
    # FastMCP automatically wraps this in the correct protocol response
    return Image(
        data=encoded_str,
        mimeType="image/png"
    )

if __name__ == "__main__":
    # Runs the server over stdio (standard input/output) by default
    mcp.run()