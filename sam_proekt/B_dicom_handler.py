import pydicom
import numpy as np
import cv2
from datetime import datetime


# Анонимизация DICOM
def anonymize_dicom(ds: pydicom.Dataset) -> pydicom.Dataset:
    sensitive_tags = ['PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex']
    for tag in sensitive_tags:
        if tag in ds:
            del ds[tag]
    return ds


# CLAHE-контраст
def apply_clahe(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# Извлечение и предобработка
def preprocess_dicom(dicom_path: str) -> np.ndarray:
    ds = pydicom.dcmread(dicom_path)

    # Логирование анонимизации
    log_entry = f"{datetime.now()} | StudyInstanceUID: {ds.StudyInstanceUID} | Anonymized\n"
    with open(r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_dicom\anon.log", "a") as f:
        f.write(log_entry)

    ds = anonymize_dicom(ds)

    # Извлечение изображения
    img = ds.pixel_array.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img *= 255.0

    # CLAHE + resize
    img = apply_clahe(img)
    img = cv2.resize(img, (224, 224))

    # Преобразование к 3 каналам
    img = np.stack([img] * 3, axis=-1)
    img = img.astype(np.uint8)

    return img
