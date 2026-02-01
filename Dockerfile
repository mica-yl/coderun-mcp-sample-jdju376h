FROM python:3

WORKDIR /usr/src/app

COPY main.py ./
RUN pip install mcp fastapi uvicorn pillow

CMD [ "python", "./main.py" ]