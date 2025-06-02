import os
import asyncio
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, GroupCoordinatorNotAvailableError, KafkaError
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

# Импорты из модулей сервиса
from pipeline import AtelectasisPipeline
from dicom_handler import DicomHandler
from detector import AtelectasisDetector

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_RAW = os.getenv("TOPIC_RAW", "raw-images")
TOPIC_PROC = os.getenv("TOPIC_PROC", "inference-results")
GROUP_ID = "ai-processor-group"
MODEL_PATH = os.getenv("MODEL_PATH", "./model/best_deit_scm_model.pth")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

# Создаем необходимые директории
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "json_reports"), exist_ok=True)

# Глобальные объекты для обработки
pipeline: AtelectasisPipeline = None
detector: AtelectasisDetector = None


async def wait_for_kafka_ready(bootstrap_servers, max_retries=15, delay=5):
    """Проверка готовности Kafka"""
    for attempt in range(max_retries):
        try:
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


def initialize_ai_components():
    """Инициализация компонентов ИИ"""
    global pipeline, detector

    logger.info("🤖 Initializing AI components...")

    # Проверяем наличие модели
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Инициализируем pipeline
    pipeline = AtelectasisPipeline(MODEL_PATH, OUTPUT_DIR)
    detector = AtelectasisDetector(MODEL_PATH)

    logger.info("✅ AI components initialized successfully")


async def process_dicom_message(message_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обработка DICOM файла из сообщения Kafka
    """
    study_id = message_data.get("study_id")
    file_path = message_data.get("file_path")
    timestamp = message_data.get("timestamp")

    logger.info(f"🔍 Processing DICOM file: {file_path}")

    start_time = time.time()

    try:
        # Обрабатываем файл через pipeline
        result = pipeline.process_dicom(file_path)

        if result["status"] == "success":
            # Добавляем метаинформацию
            processing_time = time.time() - start_time

            # Формируем результат для Kafka
            kafka_result = {
                "study_id": study_id,
                "status": "completed",
                "processing_time": processing_time,
                "timestamp_received": timestamp,
                "timestamp_processed": datetime.now().isoformat(),
                "results": result["results"],
                "json_report_path": result.get("json_report"),
                "original_dicom_path": file_path,
                "error": None
            }

            # Проверяем время обработки (требование ≤5 сек)
            if processing_time > 5.0:
                logger.warning(f"⚠️ Processing time exceeded 5 seconds: {processing_time:.2f}s")
            else:
                logger.info(f"✅ Processing completed in {processing_time:.2f}s")

            return kafka_result

        else:
            # Обработка ошибок
            return {
                "study_id": study_id,
                "status": "error",
                "processing_time": time.time() - start_time,
                "timestamp_received": timestamp,
                "timestamp_processed": datetime.now().isoformat(),
                "results": None,
                "original_dicom_path": file_path,
                "error": result.get("error", "Unknown error during processing")
            }

    except Exception as e:
        logger.error(f"❌ Error processing DICOM: {str(e)}")
        return {
            "study_id": study_id,
            "status": "error",
            "processing_time": time.time() - start_time,
            "timestamp_received": timestamp,
            "timestamp_processed": datetime.now().isoformat(),
            "results": None,
            "error": str(e)
        }


async def process_loop():
    """Основной цикл обработки"""
    logger.info("🚀 Starting AI Processing Service...")

    # Инициализируем компоненты ИИ
    initialize_ai_components()

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
        consumer_timeout_ms=1000,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all'
    )

    # Запускаем с повторными попытками
    await start_with_retries(consumer)
    await start_with_retries(producer)

    logger.info("✅ AI Processing Service ready, waiting for messages...")

    try:
        while True:
            try:
                # Получаем сообщения с таймаутом
                msg_batch = await consumer.getmany(timeout_ms=1000, max_records=10)

                if not msg_batch:
                    continue

                for topic_partition, messages in msg_batch.items():
                    for msg in messages:
                        logger.info(f"📨 Received message from {topic_partition}, offset: {msg.offset}")

                        try:
                            # Обрабатываем сообщение
                            message_data = msg.value

                            # Валидация сообщения
                            if not isinstance(message_data, dict) or "file_path" not in message_data:
                                logger.error(f"❌ Invalid message format: {message_data}")
                                continue

                            # Обрабатываем DICOM
                            result = await process_dicom_message(message_data)

                            # Отправляем результат
                            await producer.send_and_wait(TOPIC_PROC, result)
                            logger.info(f"✅ Result sent to {TOPIC_PROC} for study_id: {result['study_id']}")

                        except Exception as e:
                            logger.error(f"❌ Error processing message: {e}")
                            # Отправляем сообщение об ошибке
                            error_result = {
                                "study_id": message_data.get("study_id", "unknown"),
                                "status": "error",
                                "error": str(e),
                                "timestamp_processed": datetime.now().isoformat()
                            }
                            await producer.send_and_wait(TOPIC_PROC, error_result)

            except GroupCoordinatorNotAvailableError as e:
                logger.warning(f"⚠️ GroupCoordinator not available: {e}, retrying...")
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logger.error(f"❌ Error in processing loop: {e}")
                await asyncio.sleep(5)
                continue

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down AI Processing Service...")
    finally:
        await consumer.stop()
        await producer.stop()
        logger.info("✅ AI Processing Service stopped")


if __name__ == "__main__":
    try:
        asyncio.run(process_loop())
    except KeyboardInterrupt:
        logger.info("👋 AI Processing Service terminated by user")