import os
import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError, GroupCoordinatorNotAvailableError, KafkaError
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from shared.database import db_manager

# Импорты из модулей сервиса
from report_generator import DicomSRGenerator

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_RAW = os.getenv("TOPIC_RAW", "raw-images")
TOPIC_PROC = os.getenv("TOPIC_PROC", "inference-results")
GROUP_ID = "storage-service-group"
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
REPORTS_DIR = os.getenv("REPORTS_DIR", "./reports")

# Создаем необходимые директории
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "json"), exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "dicom_sr"), exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "json_api"), exist_ok=True)

# Глобальный генератор отчетов
report_generator: DicomSRGenerator = None


async def wait_for_kafka_ready(bootstrap_servers, max_retries=15, delay=5):
    """Проверка готовности Kafka"""
    from aiokafka import AIOKafkaProducer

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


async def connect_to_database():
    """Подключение к БД с повторными попытками"""
    max_retries = 10
    for attempt in range(max_retries):
        try:
            await db_manager.connect()
            logger.info("✅ Connected to PostgreSQL")
            return
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to DB (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(5)

    raise RuntimeError("❌ Cannot connect to database")


def initialize_storage_components():
    """Инициализация компонентов хранилища"""
    global report_generator

    logger.info("💾 Initializing storage components...")
    report_generator = DicomSRGenerator()

    asyncio.create_task(connect_to_database())

    logger.info("✅ Storage components initialized successfully")


async def process_and_store_result(result_data: Dict[str, Any]):
    """
    Обработка и сохранение результата анализа
    """
    study_id = result_data.get("study_id")
    status = result_data.get("status")
    original_dicom_path = result_data.get("original_dicom_path")

    logger.info(f"💾 Processing result for study_id: {study_id}, status: {status}")

    try:
        await db_manager.save_analysis_result(study_id, result_data)

        if status == "completed" and result_data.get("results"):
            results = result_data["results"]
            report_paths = {}

            # Сохраняем основной JSON отчет
            json_report_path = os.path.join(REPORTS_DIR, "json", f"{study_id}_report.json")
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logger.info(f"✅ JSON report saved: {json_report_path}")
            report_paths['json_report'] = json_report_path

            # Генерируем DICOM SR
            if original_dicom_path and os.path.exists(original_dicom_path):
                try:
                    dicom_files = report_generator.generate_complete_report(
                        json_report_path,
                        original_dicom_path,
                        os.path.join(REPORTS_DIR, "dicom_sr"),
                        study_id
                    )

                    if dicom_files:
                        logger.info(f"✅ Complete DICOM report generated: {len(dicom_files)} files")
                        report_paths['dicom_series'] = os.path.join(REPORTS_DIR, "dicom_sr", study_id)

                        for file_path in dicom_files:
                            if 'annotated' in file_path:
                                report_paths['dicom_annotated'] = file_path
                            elif 'sr' in file_path:
                                report_paths['dicom_sr'] = file_path

                except Exception as e:
                    logger.error(f"❌ Failed to generate DICOM report: {e}")
                    import traceback
                    traceback.print_exc()

            else:
                logger.warning(f"⚠️ Original DICOM not found at: {original_dicom_path}")

            # Генерируем JSON отчет для API
            try:
                api_json_path = os.path.join(REPORTS_DIR, "json_api", f"{study_id}_api.json")
                success = report_generator.generate_json_report(json_report_path, api_json_path)
                if success:
                    logger.info(f"✅ API JSON report generated: {api_json_path}")
                    report_paths['api_json'] = api_json_path
            except Exception as e:
                logger.error(f"❌ Failed to generate API JSON: {e}")

            # Сохраняем пути к файлам в БД
            await db_manager.save_report_paths(study_id, report_paths)

            # Обновляем статус на completed
            await db_manager.update_study_status(study_id, 'completed')


        elif status == "error":
            # Обновляем статус на error
            await db_manager.update_study_status(study_id, 'error')

        logger.info(f"✅ Result successfully stored in database for study_id: {study_id}")

    except Exception as e:
        logger.error(f"❌ Failed to store result for study_id {study_id}: {e}")
        # Обновляем статус на storage_error
        try:
            await db_manager.update_study_status(study_id, 'storage_error')
        except:
            pass

    finally:
        # Удаляем временный файл после всей обработки
        if original_dicom_path and os.path.exists(original_dicom_path):
            try:
                os.remove(original_dicom_path)
                logger.info(f"🗑️ Temporary DICOM file removed: {original_dicom_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove temporary file: {e}")


async def store_loop():
    """Основной цикл хранения результатов"""
    logger.info("🚀 Starting Storage Service...")

    # Инициализируем компоненты
    initialize_storage_components()

    # Ждем готовности Kafka
    await wait_for_kafka_ready(KAFKA_BOOTSTRAP)

    # Создаем топики
    await ensure_topics(KAFKA_BOOTSTRAP, [TOPIC_RAW, TOPIC_PROC])

    # Дополнительная пауза для стабилизации
    logger.info("⏳ Waiting for system stabilization...")
    await asyncio.sleep(10)

    # Создаем consumer
    consumer = AIOKafkaConsumer(
        TOPIC_PROC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        auto_commit_interval_ms=1000,
        consumer_timeout_ms=1000,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    # Запускаем с повторными попытками
    await start_with_retries(consumer)

    logger.info("✅ Storage Service ready, waiting for results...")

    try:
        while True:
            try:
                # Получаем сообщения с таймаутом
                msg_batch = await consumer.getmany(timeout_ms=1000, max_records=10)

                if not msg_batch:
                    continue

                for topic_partition, messages in msg_batch.items():
                    for msg in messages:
                        logger.info(f"📨 Received result from {topic_partition}, offset: {msg.offset}")

                        try:
                            result_data = msg.value

                            # Валидация данных
                            if not isinstance(result_data, dict) or "study_id" not in result_data:
                                logger.error(f"❌ Invalid result format: {result_data}")
                                continue

                            # Обрабатываем и сохраняем результат
                            await process_and_store_result(result_data)

                        except Exception as e:
                            logger.error(f"❌ Error processing result: {e}")
                            import traceback
                            traceback.print_exc()

            except GroupCoordinatorNotAvailableError as e:
                logger.warning(f"⚠️ GroupCoordinator not available: {e}, retrying...")
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logger.error(f"❌ Error in storage loop: {e}")
                await asyncio.sleep(5)
                continue

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Storage Service...")
    finally:
        await consumer.stop()
        await db_manager.disconnect()
        logger.info("✅ Storage Service stopped")


if __name__ == "__main__":
    try:
        asyncio.run(store_loop())
    except KeyboardInterrupt:
        logger.info("👋 Storage Service terminated by user")