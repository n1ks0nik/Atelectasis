FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install aiokafka python-multipart
CMD ["python", "app/main.py"]
