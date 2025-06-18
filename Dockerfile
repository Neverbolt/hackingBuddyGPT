FROM python:3.10-slim

WORKDIR /app
COPY . /app/
RUN python3 -m pip install -e .

ENTRYPOINT ["wintermute"]
