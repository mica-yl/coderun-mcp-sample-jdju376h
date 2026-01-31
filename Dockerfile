FROM python:3

WORKDIR /usr/src/app

COPY main.py ./
RUN pip install mcp pillow

RUN ls

CMD [ "python", "./main.py" ]