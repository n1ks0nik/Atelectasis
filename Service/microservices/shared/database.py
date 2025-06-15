import os
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
import logging

import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Конфигурация БД
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'user': os.getenv('POSTGRES_USER', 'user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'secretpassword'),
    'database': os.getenv('POSTGRES_DB', 'atelectasis_db'),
    'min_size': 10,
    'max_size': 20
}

logger.info("DB Config: ", DB_CONFIG)


class DatabaseManager:
    def __init__(self):
        self.pool: Optional[Pool] = None

    async def connect(self):
        """Создание пула соединений"""
        self.pool = await asyncpg.create_pool(**DB_CONFIG)

    async def disconnect(self):
        """Закрытие пула соединений"""
        if self.pool:
            await self.pool.close()

    @asynccontextmanager
    async def acquire(self):
        """Контекстный менеджер для получения соединения"""
        async with self.pool.acquire() as connection:
            yield connection

    # Методы для работы с исследованиями
    async def create_study(self, study_id: str, study_instance_uid: str,
                           user_token: str, filename: str, file_size: int) -> str:
        """Создание нового исследования"""
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO studies (study_id, study_instance_uid, user_token, 
                                   filename, file_size, status)
                VALUES ($1, $2, $3, $4, $5, 'processing')
            """, study_id, study_instance_uid, user_token, filename, file_size)
        return study_id

    async def check_study_exists(self, study_instance_uid: str) -> bool:
        """Проверка существования исследования по Study Instance UID"""
        async with self.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM studies WHERE study_instance_uid = $1
            """, study_instance_uid)
            return result > 0

    async def get_existing_study(self, study_instance_uid: str) -> Optional[Dict[str, Any]]:
        """Получение существующего исследования по Study Instance UID"""
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT study_id, filename, status, created_at 
                FROM studies 
                WHERE study_instance_uid = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            """, study_instance_uid)

            if row:
                return dict(row)
            return None

    async def update_study_status(self, study_id: str, status: str):
        """Обновление статуса исследования"""
        async with self.acquire() as conn:
            await conn.execute("""
                UPDATE studies 
                SET status = $2, updated_at = CURRENT_TIMESTAMP
                WHERE study_id = $1
            """, study_id, status)

    async def get_study(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Получение информации об исследовании"""
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT s.*, 
                       ar.status as result_status,
                       ar.atelectasis_probability,
                       ar.processing_time,
                       ar.conclusion,
                       ar.error_message,
                       ar.bbox_xmin, ar.bbox_ymin, ar.bbox_xmax, ar.bbox_ymax,
                       ar.location_description
                FROM studies s
                LEFT JOIN analysis_results ar ON s.study_id = ar.study_id
                WHERE s.study_id = $1
            """, study_id)

            if row:
                return dict(row)
            return None

    # Методы для сохранения результатов анализа
    async def save_analysis_result(self, study_id: str, result_data: Dict[str, Any]):
        """Сохранение результатов анализа"""
        async with self.acquire() as conn:
            async with conn.transaction():
                # Обновляем статус исследования
                await conn.execute("""
                    UPDATE studies 
                    SET status = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE study_id = $1
                """, study_id, result_data['status'])

                # Сохраняем результаты анализа
                results = result_data.get('results', {})
                bbox = results.get('bbox', [])

                await conn.execute("""
                    INSERT INTO analysis_results 
                    (study_id, status, atelectasis_probability, processing_time,
                     bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
                     location_description, conclusion, error_message)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (study_id) 
                    DO UPDATE SET
                        status = EXCLUDED.status,
                        atelectasis_probability = EXCLUDED.atelectasis_probability,
                        processing_time = EXCLUDED.processing_time,
                        bbox_xmin = EXCLUDED.bbox_xmin,
                        bbox_ymin = EXCLUDED.bbox_ymin,
                        bbox_xmax = EXCLUDED.bbox_xmax,
                        bbox_ymax = EXCLUDED.bbox_ymax,
                        location_description = EXCLUDED.location_description,
                        conclusion = EXCLUDED.conclusion,
                        error_message = EXCLUDED.error_message
                """,
                                   study_id,
                                   results.get('status', result_data['status']),
                                   results.get('atelectasis_probability'),
                                   result_data.get('processing_time'),
                                   bbox[0] if len(bbox) > 0 else None,
                                   bbox[1] if len(bbox) > 1 else None,
                                   bbox[2] if len(bbox) > 2 else None,
                                   bbox[3] if len(bbox) > 3 else None,
                                   results.get('location'),
                                   results.get('conclusion'),
                                   result_data.get('error')
                                   )

                # Сохраняем информацию о других патологиях
                if results.get('other_pathologies_prob', 0) >= 0.3:
                    await conn.execute("""
                        INSERT INTO other_pathologies 
                        (study_id, other_pathologies_probability)
                        VALUES ($1, $2)
                    """, study_id, results.get('other_pathologies_prob'))

    async def save_report_paths(self, study_id: str, report_paths: Dict[str, str]):
        """Сохранение путей к файлам отчетов"""
        async with self.acquire() as conn:
            for file_type, file_path in report_paths.items():
                if file_path:
                    await conn.execute("""
                        INSERT INTO report_files (study_id, file_type, file_path)
                        VALUES ($1, $2, $3)
                    """, study_id, file_type, file_path)

    async def get_report_paths(self, study_id: str) -> Dict[str, str]:
        """Получение путей к файлам отчетов"""
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT file_type, file_path 
                FROM report_files 
                WHERE study_id = $1
            """, study_id)

            return {row['file_type']: row['file_path'] for row in rows}

    async def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по всем исследованиям"""
        async with self.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_studies,
                    COUNT(CASE WHEN s.status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN s.status = 'error' THEN 1 END) as errors,
                    COUNT(CASE WHEN s.status = 'processing' THEN 1 END) as processing,
                    AVG(ar.processing_time) as avg_processing_time,
                    AVG(ar.atelectasis_probability) as avg_atelectasis_probability
                FROM studies s
                LEFT JOIN analysis_results ar ON s.study_id = ar.study_id
            """)

            pathology_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(CASE WHEN ar.status = 'atelectasis_only' THEN 1 END) as atelectasis_count,
                    COUNT(CASE WHEN ar.status = 'normal' THEN 1 END) as normal_count,
                    COUNT(CASE WHEN ar.status = 'other_pathologies' THEN 1 END) as other_pathologies_count
                FROM analysis_results ar
            """)

            return {
                **dict(stats),
                **dict(pathology_stats),
                'last_update': datetime.now().isoformat()
            }


# Глобальный экземпляр менеджера БД
db_manager = DatabaseManager()