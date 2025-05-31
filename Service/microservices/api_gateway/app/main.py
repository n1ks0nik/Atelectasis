import os
import uvicorn
from fastapi import FastAPI, UploadFile
from aiokafka import AIOKafkaProducer
from pathlib import Path
from dotenv import load_dotenv

app = FastAPI()

# .env
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent
env_path = parent_dir / '.env'
load_dotenv(dotenv_path=env_path)
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
