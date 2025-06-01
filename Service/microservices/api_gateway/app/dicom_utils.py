from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import cv2
import datetime
import numpy as np
import os


def create_dicom_from_png(png_path, output_dcm_path, add_patient_info=True):
    """
    Создает DICOM файл из PNG изображения с необходимыми тегами
    Эта функция используется только для тестирования
    """
    # Загружаем PNG (грейскейл)
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {png_path}")

    img = cv2.resize(img, (1024, 1024))

    # Создаем file_meta
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.1'  # Digital X-Ray Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Создаем DICOM Dataset
    ds = FileDataset(output_dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Заполняем обязательные теги
    if add_patient_info:
        ds.PatientName = "Test^Patient^Name"
        ds.PatientID = "123456789"
        ds.PatientBirthDate = "19800101"
        ds.PatientSex = "M"
        ds.InstitutionName = "Test Hospital"

    ds.Modality = "DX"  # Digital Radiography
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # Временные метки
    dt = datetime.datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S.%f')[:-3]
    ds.SeriesDate = ds.StudyDate
    ds.SeriesTime = ds.StudyTime
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime

    # Дополнительные теги
    ds.StudyDescription = "Chest X-Ray"
    ds.SeriesDescription = "PA View"
    ds.ViewPosition = "PA"
    ds.BodyPartExamined = "CHEST"
    ds.PatientPosition = "PA"

    # Параметры изображения
    ds.Rows, ds.Columns = img.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1

    # Конвертируем изображение в 16-битное
    img_16bit = (img.astype(np.float32) / 255.0 * 4095).astype(np.uint16)
    ds.PixelData = img_16bit.tobytes()

    # Сохраняем
    ds.save_as(output_dcm_path)
    print(f"Создан тестовый DICOM: {output_dcm_path}")
    return ds