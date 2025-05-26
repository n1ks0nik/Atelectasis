import os
import uvicorn
from fastapi import FastAPI, UploadFile
from aiokafka import AIOKafkaProducer
import asyncio

app = FastAPI()
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_RAW = os.getenv("TOPIC_RAW", "raw-images")

producer: AIOKafkaProducer

@app.on_event("startup")
async def start_kafka():
    global producer
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
    await producer.start()

@app.on_event("shutdown")
async def stop_kafka():
    await producer.stop()

@app.post("/upload/")
async def upload(file: UploadFile):
    data = await file.read()
    await producer.send_and_wait(TOPIC_RAW, data)
    return {"status": "sent to raw-images"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
