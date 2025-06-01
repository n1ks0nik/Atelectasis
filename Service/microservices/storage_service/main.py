#TODO: переделать хранение в словаре на нормальную бд

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
import pydicom

# Импорты из модулей сервиса
from report_generator import DicomSRGenerator, generate_dicom_sr_from_json, generate_json_api_report

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
env_path = parent_dir / '.env'
load_dotenv(dotenv_path=env_path)

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

# Временное хранилище результатов (в реальной системе - база данных)
results_storage: Dict[str, Any] = {}


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


def initialize_storage_components():
    """Инициализация компонентов хранилища"""
    global report_generator

    logger.info("💾 Initializing storage components...")
    report_generator = DicomSRGenerator()
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
        # Сохраняем результат в хранилище
        results_storage[study_id] = result_data

        if status == "completed" and result_data.get("results"):
            results = result_data["results"]

            # Сохраняем основной JSON отчет
            json_report_path = os.path.join(REPORTS_DIR, "json", f"{study_id}_report.json")
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logger.info(f"✅ JSON report saved: {json_report_path}")

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
                        result_data["dicom_files"] = dicom_files
                        result_data["dicom_series_path"] = os.path.join(REPORTS_DIR, "dicom_sr", study_id)
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
                    result_data["api_json_path"] = api_json_path
            except Exception as e:
                logger.error(f"❌ Failed to generate API JSON: {e}")

            # Обновляем хранилище с путями к отчетам
            results_storage[study_id] = result_data

            # Логируем статистику
            if results.get("status") == "atelectasis_only":
                logger.info(f"🔴 Atelectasis detected! Probability: {results.get('atelectasis_probability', 0):.2%}")
            elif results.get("status") == "normal":
                logger.info(
                    f"🟢 Normal result. Atelectasis probability: {results.get('atelectasis_probability', 0):.2%}")
            elif results.get("status") == "other_pathologies":
                logger.info(
                    f"🟡 Other pathologies detected. Atelectasis probability: {results.get('atelectasis_probability', 0):.2%}"
                )
        elif status == "error":
            # Сохраняем информацию об ошибке
            error_info = {
                "study_id": study_id,
                "status": "error",
                "error": result_data.get("error"),
                "timestamp": result_data.get("timestamp_processed"),
                "processing_time": result_data.get("processing_time")
            }

            error_path = os.path.join(REPORTS_DIR, "json", f"{study_id}_error.json")
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, ensure_ascii=False, indent=4)

            logger.error(f"❌ Error result saved: {error_path}")

        # Добавляем метаданные о сохранении
        result_data["stored_at"] = datetime.now().isoformat()
        results_storage[study_id] = result_data

        logger.info(f"✅ Result successfully stored for study_id: {study_id}")

    except Exception as e:
        logger.error(f"❌ Failed to store result for study_id {study_id}: {e}")
        # Сохраняем минимальную информацию об ошибке
        results_storage[study_id] = {
            "study_id": study_id,
            "status": "storage_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

    if original_dicom_path and os.path.exists(original_dicom_path):
        try:
            os.remove(original_dicom_path)
            logger.info(f"🗑️ Temporary DICOM file removed: {original_dicom_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to remove temporary file: {e}")


def get_result_by_study_id(study_id: str) -> Optional[Dict[str, Any]]:
    """
    Получение результата по study_id
    В реальной системе должно быть подключение к БД
    """
    return results_storage.get(study_id)


async def cleanup_old_results():
    """
    Периодическая очистка старых результатов
    """
    while True:
        try:
            await asyncio.sleep(3600)  # Каждый час

            current_time = datetime.now()
            to_remove = []

            for study_id, result in results_storage.items():
                # Удаляем результаты старше 24 часов
                if "stored_at" in result:
                    stored_time = datetime.fromisoformat(result["stored_at"])
                    if (current_time - stored_time).total_seconds() > 86400:  # 24 часа
                        to_remove.append(study_id)

            for study_id in to_remove:
                del results_storage[study_id]
                logger.info(f"🗑️ Removed old result: {study_id}")

            if to_remove:
                logger.info(f"✅ Cleaned up {len(to_remove)} old results")

        except Exception as e:
            logger.error(f"❌ Error in cleanup task: {e}")


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

    # Запускаем задачу очистки в фоне
    cleanup_task = asyncio.create_task(cleanup_old_results())

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

                            # Логируем статистику
                            total_results = len(results_storage)
                            completed = sum(1 for r in results_storage.values() if r.get("status") == "completed")
                            errors = sum(1 for r in results_storage.values() if r.get("status") == "error")

                            logger.info(
                                f"📊 Storage stats - Total: {total_results}, Completed: {completed}, Errors: {errors}")

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
        cleanup_task.cancel()
        await consumer.stop()
        logger.info("✅ Storage Service stopped")


# API endpoints для получения результатов (для интеграции с API Gateway)
async def get_study_result(study_id: str) -> Optional[Dict[str, Any]]:
    """
    Получить результат исследования по ID
    Эта функция может быть вызвана через gRPC или REST API
    """
    result = get_result_by_study_id(study_id)

    if not result:
        return None

    # Формируем ответ для API
    response = {
        "study_id": study_id,
        "status": result.get("status"),
        "processing_time": result.get("processing_time"),
        "timestamp_processed": result.get("timestamp_processed")
    }

    if result.get("status") == "completed":
        # Добавляем результаты анализа
        if result.get("results"):
            response["results"] = result["results"]

        # Добавляем пути к отчетам
        if result.get("api_json_path") and os.path.exists(result["api_json_path"]):
            with open(result["api_json_path"], 'r', encoding='utf-8') as f:
                response["detailed_report"] = json.load(f)

        if result.get("dicom_sr_path"):
            response["dicom_sr_available"] = True
            response["dicom_sr_path"] = result["dicom_sr_path"]

    elif result.get("status") == "error":
        response["error"] = result.get("error")

    return response


async def get_study_statistics() -> Dict[str, Any]:
    """
    Получить статистику по всем исследованиям
    """
    total = len(results_storage)
    completed = sum(1 for r in results_storage.values() if r.get("status") == "completed")
    errors = sum(1 for r in results_storage.values() if r.get("status") == "error")
    processing = sum(1 for r in results_storage.values() if r.get("status") == "processing")

    # Статистика по патологиям
    atelectasis_count = 0
    normal_count = 0
    other_pathologies_count = 0

    for result in results_storage.values():
        if result.get("status") == "completed" and result.get("results"):
            results = result["results"]
            if results.get("status") == "atelectasis_only":
                atelectasis_count += 1
            elif results.get("status") == "normal":
                normal_count += 1
            elif results.get("status") == "other_pathologies":
                other_pathologies_count += 1

    # Средняя вероятность ателектаза
    atelectasis_probs = []
    for result in results_storage.values():
        if result.get("status") == "completed" and result.get("results"):
            prob = result["results"].get("atelectasis_probability")
            if prob is not None:
                atelectasis_probs.append(prob)

    avg_atelectasis_prob = sum(atelectasis_probs) / len(atelectasis_probs) if atelectasis_probs else 0

    # Среднее время обработки
    processing_times = []
    for result in results_storage.values():
        if result.get("processing_time") is not None:
            processing_times.append(result["processing_time"])

    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

    return {
        "total_studies": total,
        "completed": completed,
        "errors": errors,
        "processing": processing,
        "pathology_statistics": {
            "atelectasis": atelectasis_count,
            "normal": normal_count,
            "other_pathologies": other_pathologies_count
        },
        "average_atelectasis_probability": avg_atelectasis_prob,
        "average_processing_time": avg_processing_time,
        "last_update": datetime.now().isoformat()
    }


# ТОЧКА ВХОДА - ЭТО БЫЛО ПРОПУЩЕНО!
if __name__ == "__main__":
    try:
        asyncio.run(store_loop())
    except KeyboardInterrupt:
        logger.info("👋 Storage Service terminated by user")