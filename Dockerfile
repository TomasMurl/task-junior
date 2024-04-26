FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY SimpleServer.py SimpleServer.py

CMD ["python", "SimpleServer.py"]