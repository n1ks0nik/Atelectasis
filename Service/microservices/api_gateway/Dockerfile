FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn aiokafka python-multipart
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
