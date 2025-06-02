import os, sys
import asyncio
import logging
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import uuid

import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError
from dotenv import load_dotenv
import pydicom

# Импорты из модулей сервиса
from validators import DicomValidator
from dicom_utils import create_dicom_from_png

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from  shared.database import db_manager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_RAW = os.getenv("TOPIC_RAW", "raw-images")
TOPIC_RESULTS = os.getenv("TOPIC_PROC", "inference-results")
UPLOAD_DIR = "./uploads"
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "104857600"))  # 100MB

# Создаем директорию для загрузок
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Atelectasis Detection API", version="1.0.0")
security = HTTPBearer()

# Глобальный producer для Kafka
producer: Optional[AIOKafkaProducer] = None


# Модели данных
class ProcessingRequest(BaseModel):
    study_id: str
    file_path: str
    timestamp: str
    metadata: dict


class ProcessingResponse(BaseModel):
    study_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    kafka_connected: bool
    timestamp: str


# Функции для работы с Kafka
async def wait_for_kafka_ready(bootstrap_servers, max_retries=15, delay=5):
    """Проверка готовности Kafka"""
    for attempt in range(max_retries):
        try:
            test_producer = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                request_timeout_ms=5000,
                connections_max_idle_ms=10000
            )
            await test_producer.start()
            logger.info("✅ Kafka is ready!")
            await test_producer.stop()
            return True
        except Exception as e:
            logger.warning(f"⚠️ Kafka not ready (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(delay)

    raise RuntimeError("❌ Kafka not ready after maximum retries")


# События приложения
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global producer

    logger.info("🚀 Starting API Gateway Service...")

    # Подключаемся к БД
    try:
        await db_manager.connect()
        logger.info("✅ Connected to PostgreSQL")
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        raise

    # Ждем готовности Kafka
    try:
        await wait_for_kafka_ready(KAFKA_BOOTSTRAP)

        # Создаем producer
        producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retry_backoff_ms=2000,
            request_timeout_ms=30000
        )
        await producer.start()
        logger.info("✅ Kafka producer started successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Kafka: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке"""
    global producer

    # Отключаемся от БД
    await db_manager.disconnect()
    logger.info("✅ Disconnected from PostgreSQL")

    if producer:
        await producer.stop()
        logger.info("✅ Kafka producer stopped")


# Функция проверки JWT токена (заглушка)
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Проверка JWT токена"""
    # TODO: Реализовать проверку JWT
    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса"""
    kafka_connected = producer is not None and not producer._closed

    return HealthResponse(
        status="healthy" if kafka_connected else "degraded",
        kafka_connected=kafka_connected,
        timestamp=datetime.now().isoformat()
    )


@app.post("/analyze", response_model=ProcessingResponse)
async def analyze_dicom(
        file: UploadFile,
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Загрузка и анализ DICOM файла
    """
    # Проверяем размер файла
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error_code": 413, "message": "Файл слишком большой", "severity": "critical"}
        )

    # Генерируем уникальный ID для исследования
    study_id = str(uuid.uuid4())
    temp_file_path = None

    try:
        # Сохраняем файл временно
        temp_file_path = os.path.join(UPLOAD_DIR, f"{study_id}_{file.filename}")

        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Валидируем DICOM
        is_valid, error = DicomValidator.validate_dicom_file(temp_file_path)
        if not is_valid:
            raise HTTPException(
                status_code=error["error_code"],
                detail=error
            )

        # Создаем сообщение для Kafka
        kafka_message = {
            "study_id": study_id,
            "file_path": temp_file_path,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "user_token": credentials.credentials,
            "metadata": {
                "content_type": file.content_type,
                "file_size": file.size
            }
        }

        await db_manager.create_study(
            study_id=study_id,
            study_instance_uid=pydicom.dcmread(temp_file_path).StudyInstanceUID,
            user_token=credentials.credentials,
            filename=file.filename,
            file_size=file.size
        )

        # Отправляем в Kafka
        await producer.send_and_wait(TOPIC_RAW, kafka_message)

        logger.info(f"✅ File {file.filename} sent to processing, study_id: {study_id}")

        return ProcessingResponse(
            study_id=study_id,
            status="processing",
            message="Файл успешно загружен и отправлен на обработку"
        )

    except HTTPException:
        # Удаляем временный файл при ошибке валидации
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise

    except KafkaConnectionError as e:
        # Удаляем временный файл при ошибке Kafka
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        logger.error(f"❌ Kafka error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": 503, "message": "Сервис временно недоступен", "severity": "critical"}
        )

    except Exception as e:
        # Удаляем временный файл при любой другой ошибке
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": 500, "message": "Внутренняя ошибка сервера", "severity": "critical"}
        )


@app.get("/result/{study_id}")
async def get_result(
        study_id: str,
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Получение результата анализа по study_id
    """
    # Получаем из БД
    study = await db_manager.get_study(study_id)

    if not study:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study not found"
        )

    # Формируем ответ
    response = {
        "study_id": study_id,
        "status": study['status'],
        "created_at": study['created_at'].isoformat() if study['created_at'] else None,
        "updated_at": study['updated_at'].isoformat() if study['updated_at'] else None
    }

    if study['status'] == 'completed':
        # Добавляем результаты анализа
        if study['result_status']:
            response['results'] = {
                "status": study['result_status'],
                "atelectasis_probability": float(study['atelectasis_probability']) if study[
                    'atelectasis_probability'] else None,
                "processing_time": float(study['processing_time']) if study['processing_time'] else None,
                "conclusion": study['conclusion'],
                "location": study['location_description']
            }

            if study['bbox_xmin'] is not None:
                response['results']['bbox'] = [
                    study['bbox_xmin'],
                    study['bbox_ymin'],
                    study['bbox_xmax'],
                    study['bbox_ymax']
                ]

        # Добавляем пути к файлам
        report_paths = await db_manager.get_report_paths(study_id)
        if report_paths:
            response['reports'] = report_paths

    elif study['status'] == 'error':
        response['error'] = study['error_message']

    return response


@app.get("/statistics")
async def get_statistics(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Получение общей статистики системы
    """
    stats = await db_manager.get_statistics()
    return stats


@app.post("/test/create_dicom")
async def create_test_dicom(
        png_file: UploadFile,
        add_patient_info: bool = True
):
    """
    Тестовый endpoint для создания DICOM из PNG (только для разработки)
    """
    if not png_file.filename.endswith('.png'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Требуется PNG файл"
        )

    # Сохраняем PNG временно
    temp_png = os.path.join(UPLOAD_DIR, f"temp_{png_file.filename}")
    temp_dcm = os.path.join(UPLOAD_DIR, f"test_{png_file.filename.replace('.png', '.dcm')}")

    try:
        with open(temp_png, "wb") as buffer:
            content = await png_file.read()
            buffer.write(content)

        # Конвертируем в DICOM
        create_dicom_from_png(temp_png, temp_dcm, add_patient_info)

        return {
            "status": "success",
            "dicom_path": temp_dcm,
            "message": "DICOM файл успешно создан"
        }

    finally:
        # Удаляем временный PNG
        if os.path.exists(temp_png):
            os.remove(temp_png)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )