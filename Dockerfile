FROM michaelsamir13/model-base:v0.1

RUN pwd

# MCP
RUN pip install mcp fastapi uvicorn pillow numpy pymupdf


COPY main.py ./
# pytorch cpu

# TruFor 


CMD [ "python", "./main.py" ]