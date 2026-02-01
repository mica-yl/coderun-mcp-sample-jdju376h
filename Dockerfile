FROM python:3

WORKDIR /usr/src/app

COPY main.py ./
RUN pip install mcp fastapi uvicorn pillow numpy

# pytorch cpu
# TruFor 


CMD [ "python", "./main.py" ]