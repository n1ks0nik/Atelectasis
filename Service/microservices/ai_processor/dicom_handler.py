import pydicom
import numpy as np
import cv2
from datetime import datetime
import os
import json


class DicomHandler:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or os.getenv('LOG_DIR', './logs')
        self.anonymization_log = os.path.join(self.log_dir, "anonymization.log")
        os.makedirs(self.log_dir, exist_ok=True)

    def anonymize_dicom(self, ds: pydicom.Dataset, preserve_study_info=True) -> tuple:
        """
        Анонимизирует DICOM файл и возвращает (анонимизированный dataset, удаленные данные)
        """
        removed_data = {}

        # Теги для удаления согласно требованиям
        sensitive_tags = [
            (0x0010, 0x0010),  # PatientName
            (0x0010, 0x0030),  # PatientBirthDate
            (0x0010, 0x0040),  # PatientSex
            (0x0008, 0x0080),  # InstitutionName
        ]

        for tag in sensitive_tags:
            if tag in ds:
                tag_name = pydicom.datadict.keyword_for_tag(tag)
                removed_data[tag_name] = str(ds[tag].value)
                del ds[tag]

        # Логирование анонимизации
        if preserve_study_info and hasattr(ds, 'StudyInstanceUID'):
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "study_instance_uid": str(ds.StudyInstanceUID),
                "removed_fields": list(removed_data.keys())
            }

            with open(self.anonymization_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return ds, removed_data

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Применяет CLAHE для улучшения контраста"""
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_uint8)

    def preprocess_dicom(self, dicom_path: str, target_size=(224, 224)) -> tuple:
        """
        Извлекает и предобрабатывает изображение из DICOM
        Возвращает (обработанное изображение, метаданные)
        """
        ds = pydicom.dcmread(dicom_path)

        # Сохраняем важные метаданные перед анонимизацией
        metadata = {
            "StudyInstanceUID": str(ds.StudyInstanceUID) if hasattr(ds, 'StudyInstanceUID') else None,
            "SeriesInstanceUID": str(ds.SeriesInstanceUID) if hasattr(ds, 'SeriesInstanceUID') else None,
            "SOPInstanceUID": str(ds.SOPInstanceUID) if hasattr(ds, 'SOPInstanceUID') else None,
            "Modality": str(ds.Modality) if hasattr(ds, 'Modality') else None,
            "ViewPosition": str(ds.ViewPosition) if hasattr(ds, 'ViewPosition') else None,
        }

        # Анонимизация
        ds, removed_data = self.anonymize_dicom(ds)
        metadata["anonymized_data"] = removed_data

        # Извлечение изображения
        img = ds.pixel_array.astype(np.float32)

        # Применение RescaleSlope и RescaleIntercept если есть
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept

        # Нормализация к диапазону 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0

        # CLAHE + resize
        img = cv2.resize(img, target_size)

        # Преобразование к 3 каналам для нейросети
        img_rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)

        return img_rgb, metadata