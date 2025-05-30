import os
import asyncio
import logging
import json
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, GroupCoordinatorNotAvailableError, KafkaError
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_RAW = os.getenv("TOPIC_RAW", "raw-images")
TOPIC_PROC = os.getenv("TOPIC_PROC", "inference-results")
GROUP_ID = "proc-service-group"


async def wait_for_kafka_ready(bootstrap_servers, max_retries=15, delay=5):
    """Простая проверка готовности Kafka через создание продюсера"""

    for attempt in range(max_retries):
        try:
            # Пробуем создать и запустить продюсер
            producer = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                request_timeout_ms=5000,
                connections_max_idle_ms=10000
            )

            await producer.start()
            logger.info(f"✅ Kafka is ready! Connection successful")
            await producer.stop()
            return True

        except Exception as e:
            logger.warning(f"⚠️ Kafka not ready (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            await asyncio.sleep(delay)

    raise RuntimeError("❌ Kafka not ready after maximum retries")


async def start_with_retries(component, max_retries=10, delay=3):
    """Запуск компонента с повторными попытками"""
    for attempt in range(max_retries):
        try:
            await component.start()
            logger.info(f"✅ Component started successfully")
            return
        except (KafkaConnectionError, GroupCoordinatorNotAvailableError, KafkaError) as e:
            logger.warning(f"⚠️ Failed to start component (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(delay)

    raise RuntimeError("❌ Cannot start component after retries")


async def ensure_topics(bootstrap, topics):
    """Создание топиков если их нет"""
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await start_with_retries(admin, max_retries=5)

    try:
        existing = await admin.list_topics()
        to_create = [
            NewTopic(name=topic, num_partitions=1, replication_factor=1)
            for topic in topics if topic not in existing
        ]

        if to_create:
            await admin.create_topics(to_create)
            logger.info(f"✅ Topics created: {[t.name for t in to_create]}")
        else:
            logger.info("✅ All topics already exist")
    finally:
        await admin.close()


async def process_loop():
    """Основной цикл обработки"""
    logger.info("🚀 Starting processing service...")

    # Ждем готовности Kafka
    await wait_for_kafka_ready(KAFKA_BOOTSTRAP)

    # Создаем топики
    await ensure_topics(KAFKA_BOOTSTRAP, [TOPIC_RAW, TOPIC_PROC])

    # Дополнительная пауза для стабилизации
    logger.info("⏳ Waiting for system stabilization...")
    await asyncio.sleep(10)

    # Создаем consumer и producer
    consumer = AIOKafkaConsumer(
        TOPIC_RAW,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        auto_commit_interval_ms=1000,
        consumer_timeout_ms=1000
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        acks='all'
    )

    # Запускаем с повторными попытками
    await start_with_retries(consumer)
    await start_with_retries(producer)

    logger.info("✅ Processing service ready, waiting for messages...")

    try:
        while True:
            try:
                # Получаем сообщения с таймаутом
                msg_batch = await consumer.getmany(timeout_ms=1000, max_records=10)

                if not msg_batch:
                    continue

                for topic_partition, messages in msg_batch.items():
                    for msg in messages:
                        logger.info(f"📨 Processing message from {topic_partition}, offset: {msg.offset}")

                        # Обрабатываем сообщение
                        raw_data = msg.value
                        result_dict = {
                            "processed": True,
                            "size": len(raw_data),
                            "timestamp": asyncio.get_event_loop().time(),
                            "message_offset": msg.offset,
                            "topic": msg.topic,
                            "partition": msg.partition
                        }
                        result = json.dumps(result_dict).encode('utf-8')

                        # Отправляем результат
                        await producer.send_and_wait(TOPIC_PROC, result)
                        logger.info(f"✅ Message processed and sent to {TOPIC_PROC}")

            except GroupCoordinatorNotAvailableError as e:
                logger.warning(f"⚠️ GroupCoordinator not available: {e}, retrying...")
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logger.error(f"❌ Error in processing loop: {e}")
                await asyncio.sleep(5)
                continue

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down processing service...")
    finally:
        await consumer.stop()
        await producer.stop()
        logger.info("✅ Processing service stopped")


if __name__ == "__main__":
    try:
        asyncio.run(process_loop())
    except KeyboardInterrupt:
        logger.info("👋 Processing service terminated by user")