from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import cv2
import datetime

def create_dicom_from_png(png_path, output_dcm_path):
    # Загружаем PNG (грейскейл)
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1024, 1024))  # или размер, подходящий под твой случай

    # Создаем file_meta
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    # Важный тег, без которого ошибка
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Создаем DICOM Dataset
    ds = FileDataset(output_dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Заполняем обязательные теги
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.Modality = "DX"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
    ds.ViewPosition = "PA"

    ds.Rows, ds.Columns = img.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = img.tobytes()

    # Сохраняем
    ds.save_as(output_dcm_path)
    print(f"Создан тестовый DICOM: {output_dcm_path}")

# Пример вызова
create_dicom_from_png(r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\new_classes\Atelectasis\00003426_005.png",
                      r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_dicom\pre_dicom\test_atelectasis_2.dcm")
