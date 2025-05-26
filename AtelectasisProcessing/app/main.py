import os
import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, GroupCoordinatorNotAvailableError
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_RAW    = os.getenv("TOPIC_RAW", "raw-images")
TOPIC_PROC   = os.getenv("TOPIC_PROC", "inference-results")
GROUP_ID     = "proc-service-group"


async def start_with_retries(start_coro, retries=10, delay=3):
    for i in range(retries):
        try:
            return await start_coro()
        except (KafkaConnectionError, GroupCoordinatorNotAvailableError) as e:
            print(f"⚠️ Kafka not ready ({e}), retry {i+1}/{retries}")
            await asyncio.sleep(delay)
    raise RuntimeError("❌ Cannot connect to Kafka after retries")


async def ensure_topics(bootstrap, topics):
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await start_with_retries(admin.start)   # та же retry-обёртка, что и для продюсера
    existing = await admin.list_topics()
    to_create = [
        NewTopic(name=t, num_partitions=1, replication_factor=1)
        for t in topics if t not in existing
    ]
    if to_create:
        await admin.create_topics(to_create)
        print("✅ Topics created:", [t.name for t in to_create])
    await admin.close()


async def process_loop():
    await ensure_topics(KAFKA_BOOTSTRAP, [TOPIC_RAW, TOPIC_PROC])

    await asyncio.sleep(30)

    consumer = AIOKafkaConsumer(
        TOPIC_RAW,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID
    )
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
    await start_with_retries(consumer.start)
    await start_with_retries(producer.start)

    try:
        while True:
            try:
                msg = await consumer.getone()
            except GroupCoordinatorNotAvailableError:
                print("⚠️ GroupCoordinator not ready, retrying in 3s…")
                await asyncio.sleep(3)
                continue

            raw = msg.value
            result = b'{"processed": true, "size": %d}' % len(raw)
            await producer.send_and_wait(TOPIC_PROC, result)

    finally:
        await consumer.stop()
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(process_loop())
