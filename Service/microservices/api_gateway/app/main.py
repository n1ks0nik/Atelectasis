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
import zipfile
import io

import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, Depends, status, Request, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
from shared.database import db_manager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_RAW = os.getenv("TOPIC_RAW", "raw-images")
TOPIC_RESULTS = os.getenv("TOPIC_PROC", "inference-results")
UPLOAD_DIR = "./uploads"
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "104857600"))  # 100MB

# Создаем директории
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(title="Atelectasis Detection API", version="1.0.0")
security = HTTPBearer()

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Глобальный producer для Kafka
producer: Optional[AIOKafkaProducer] = None

# Простая система аутентификации
VALID_TOKENS = {
    "demo_token_123": {"username": "demo_user", "role": "user"},
    "admin_token_456": {"username": "admin", "role": "admin"},
    "test_token_789": {"username": "test_user", "role": "user"}
}


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


class LoginRequest(BaseModel):
    token: str


# Функции аутентификации
def verify_token_simple(token: str):
    """Простая проверка токена"""
    return VALID_TOKENS.get(token)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Проверка JWT токена"""
    token = credentials.credentials
    user_data = verify_token_simple(token)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return {"token": token, **user_data}


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


# ========== WEB INTERFACE ROUTES ==========

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Главная страница - дашборд"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Страница загрузки файлов"""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    """Страница результатов"""
    return templates.TemplateResponse("results.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Страница входа"""
    return templates.TemplateResponse("login.html", {"request": request})


# ========== API ROUTES ==========

@app.post("/api/login")
async def login(login_data: LoginRequest):
    """API для входа в систему"""
    user_data = verify_token_simple(login_data.token)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    return {
        "status": "success",
        "user": user_data,
        "token": login_data.token
    }


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
        user_data: dict = Depends(verify_token)
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
            "user_token": user_data["token"],
            "metadata": {
                "content_type": file.content_type,
                "file_size": file.size
            }
        }

        # Получаем Study Instance UID из DICOM
        dicom_data = pydicom.dcmread(temp_file_path)
        study_instance_uid = dicom_data.StudyInstanceUID

        # Проверяем, существует ли уже такое исследование
        existing_study = await db_manager.get_existing_study(study_instance_uid)

        if existing_study:
            logger.info(f"⚠️ Study with UID {study_instance_uid} already exists: {existing_study['study_id']}")
            # Создаем уникальный Study Instance UID для новой записи
            unique_study_uid = f"{study_instance_uid}.{study_id}"
            logger.info(f"✅ Creating new study with modified UID: {unique_study_uid}")
        else:
            unique_study_uid = study_instance_uid

        # Создаем запись в БД
        try:
            await db_manager.create_study(
                study_id=study_id,
                study_instance_uid=unique_study_uid,
                user_token=user_data["token"],
                filename=file.filename,
                file_size=file.size
            )
        except Exception as db_error:
            # Если все еще возникает ошибка дублирования (редкий случай)
            if "duplicate key value violates unique constraint" in str(db_error):
                logger.warning(f"⚠️ Unexpected duplicate error, using timestamp suffix")
                timestamp_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
                fallback_uid = f"{study_instance_uid}.{timestamp_suffix}"
                await db_manager.create_study(
                    study_id=study_id,
                    study_instance_uid=fallback_uid,
                    user_token=user_data["token"],
                    filename=file.filename,
                    file_size=file.size
                )
                logger.info(f"✅ Created study with fallback UID: {fallback_uid}")
            else:
                raise db_error

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
        user_data: dict = Depends(verify_token)
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
        "updated_at": study['updated_at'].isoformat() if study['updated_at'] else None,
        "filename": study['filename']
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


@app.get("/api/studies")
async def get_studies(
        limit: int = 50,
        offset: int = 0,
        user_data: dict = Depends(verify_token)
):
    """
    Получение списка исследований пользователя
    """
    # Простой запрос всех исследований (в реальной системе нужна фильтрация по пользователю)
    async with db_manager.acquire() as conn:
        rows = await conn.fetch("""
            SELECT s.study_id, s.filename, s.status, s.created_at, s.updated_at,
                   ar.atelectasis_probability, ar.conclusion
            FROM studies s
            LEFT JOIN analysis_results ar ON s.study_id = ar.study_id
            ORDER BY s.created_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)

        studies = []
        for row in rows:
            study = dict(row)
            if study['created_at']:
                study['created_at'] = study['created_at'].isoformat()
            if study['updated_at']:
                study['updated_at'] = study['updated_at'].isoformat()
            studies.append(study)

        return {"studies": studies}


@app.get("/statistics")
async def get_statistics(
        user_data: dict = Depends(verify_token)
):
    """
    Получение общей статистики системы
    """
    stats = await db_manager.get_statistics()
    return stats


@app.get("/download/reports/{study_id}")
async def download_reports(
        study_id: str,
        user_data: dict = Depends(verify_token)
):
    """
    Скачивание отчетов в виде ZIP архива
    """
    # Получаем пути к файлам
    report_paths = await db_manager.get_report_paths(study_id)

    logger.info(f"📁 Download request for study_id: {study_id}")
    logger.info(f"📁 Found report paths: {report_paths}")

    if not report_paths:
        logger.warning(f"❌ No report paths found for study_id: {study_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reports not found"
        )

    # Создаем ZIP архив в памяти
    zip_buffer = io.BytesIO()
    files_added = 0

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_type, file_path in report_paths.items():
            logger.info(f"📄 Processing: {file_type} -> {file_path}")

            if file_type == 'dicom_series':
                # Это путь к папке - добавляем всё содержимое
                if os.path.exists(file_path) and os.path.isdir(file_path):
                    # Добавляем все файлы из папки dicom_series
                    for root, dirs, files in os.walk(file_path):
                        for file in files:
                            file_full_path = os.path.join(root, file)
                            # Относительный путь от базовой папки
                            relative_path = os.path.relpath(file_full_path, os.path.dirname(file_path))

                            logger.info(f"✅ Adding to ZIP: {relative_path}")
                            zip_file.write(file_full_path, relative_path)
                            files_added += 1
                else:
                    logger.warning(f"⚠️ DICOM series directory not found: {file_path}")

            elif file_type in ['json_report', 'api_json']:
                # Только JSON файлы добавляем в корень архива
                if os.path.exists(file_path):
                    try:
                        # Определяем имя файла в корне архива
                        if file_type == 'json_report':
                            archive_name = f"{study_id}_report.json"
                        elif file_type == 'api_json':
                            archive_name = f"{study_id}_api_report.json"

                        file_size = os.path.getsize(file_path)
                        logger.info(f"✅ Adding JSON file {archive_name} (size: {file_size} bytes)")

                        zip_file.write(file_path, archive_name)
                        files_added += 1

                    except Exception as e:
                        logger.error(f"❌ Error adding file {file_path}: {e}")
                else:
                    logger.warning(f"⚠️ JSON file not found: {file_path}")

            # Пропускаем dicom_sr и dicom_annotated, так как они уже включены в dicom_series
            elif file_type in ['dicom_sr', 'dicom_annotated']:
                logger.info(f"⏭️ Skipping {file_type} (included in dicom_series folder)")
                continue

            else:
                # Для любых других типов файлов
                if os.path.exists(file_path):
                    try:
                        archive_name = f"{study_id}_{file_type}"
                        file_size = os.path.getsize(file_path)
                        logger.info(f"✅ Adding other file {archive_name} (size: {file_size} bytes)")

                        zip_file.write(file_path, archive_name)
                        files_added += 1

                    except Exception as e:
                        logger.error(f"❌ Error adding file {file_path}: {e}")
                else:
                    logger.warning(f"⚠️ File not found: {file_path}")

    if files_added == 0:
        logger.warning(f"❌ No files were added to ZIP for study_id: {study_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No report files found on disk"
        )

    zip_buffer.seek(0)
    archive_size = len(zip_buffer.getvalue())
    logger.info(f"📦 Created ZIP archive with {files_added} files, size: {archive_size} bytes")

    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={study_id}_reports.zip"}
    )


@app.get("/download/dicom/{study_id}/{file_type}")
async def download_dicom_file(
        study_id: str,
        file_type: str,  # 'sr' или 'annotated'
        user_data: dict = Depends(verify_token)
):
    """
    Скачивание отдельного DICOM файла
    """
    report_paths = await db_manager.get_report_paths(study_id)

    if file_type == 'sr' and 'dicom_sr' in report_paths:
        file_path = report_paths['dicom_sr']
        filename = f"{study_id}_structured_report.dcm"
    elif file_type == 'annotated' and 'dicom_annotated' in report_paths:
        file_path = report_paths['dicom_annotated']
        filename = f"{study_id}_annotated_image.dcm"
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="DICOM file not found"
        )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk"
        )

    return FileResponse(
        file_path,
        media_type="application/dicom",
        filename=filename
    )


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