import pydicom
import os


class DicomValidator:
    """
    Валидатор для DICOM файлов согласно требованиям
    """

    REQUIRED_TAGS = [
        'PatientID',
        'StudyInstanceUID',
        'SeriesInstanceUID',
        'Modality',
        'SOPInstanceUID',
        'StudyDate',
        'ViewPosition'
    ]

    ALLOWED_MODALITIES = ['DX', 'CR']  # Digital X-Ray и Computed Radiography
    MIN_RESOLUTION = 512
    MAX_SERIES_IMAGES = 100

    @staticmethod
    def validate_dicom_file(file_path):
        """
        Валидирует DICOM файл согласно требованиям
        Возвращает (is_valid, error_message)
        """
        try:
            # Проверяем, что файл существует
            if not os.path.exists(file_path):
                return False, {"error_code": 403, "message": "Файл недоступен для чтения.", "severity": "critical"}

            # Пытаемся прочитать DICOM
            try:
                ds = pydicom.dcmread(file_path)
            except:
                return False, {"error_code": 400, "message": "Неверный формат файла. Требуется DICOM.",
                               "severity": "critical"}

            # Проверяем обязательные теги
            missing_tags = []
            for tag in DicomValidator.REQUIRED_TAGS:
                if not hasattr(ds, tag):
                    missing_tags.append(tag)

            if missing_tags:
                return False, {
                    "error_code": 400,
                    "message": f"Отсутствуют метаданные DICOM: {', '.join(missing_tags)}.",
                    "severity": "critical"
                }

            # Проверяем модальность
            if ds.Modality not in DicomValidator.ALLOWED_MODALITIES:
                return False, {
                    "error_code": 422,
                    "message": "Неверная анатомическая область. Требуется рентген грудной клетки.",
                    "severity": "critical"
                }

            # Проверяем разрешение
            if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                if ds.Rows < DicomValidator.MIN_RESOLUTION or ds.Columns < DicomValidator.MIN_RESOLUTION:
                    return False, {
                        "error_code": 422,
                        "message": f"Низкое качество изображения. Минимальное разрешение {DicomValidator.MIN_RESOLUTION}x{DicomValidator.MIN_RESOLUTION} px.",
                        "severity": "warning"
                    }

            # Проверяем, что это изображение грудной клетки
            # TODO: проверить есть ли такие теги в реальных диком и убрать проверку если нет
            if hasattr(ds, 'BodyPartExamined'):
                if ds.BodyPartExamined.upper() not in ['CHEST', 'THORAX']:
                    return False, {
                        "error_code": 422,
                        "message": "Неверная анатомическая область. Требуется рентген грудной клетки.",
                        "severity": "critical"
                    }

            return True, None

        except Exception as e:
            return False, {
                "error_code": 500,
                "message": f"Ошибка при валидации файла: {str(e)}",
                "severity": "critical"
            }