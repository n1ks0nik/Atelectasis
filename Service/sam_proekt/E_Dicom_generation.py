import os
import json
from datetime import datetime
from pydicom import dcmread, Dataset
from pydicom.uid import generate_uid
from pydicom.dataset import FileDataset
import pydicom.uid


class DicomSRGenerator:
    def __init__(self):
        self.manufacturer = "AtelectasisAI"
        self.software_version = "1.0"
        self.observer_type = "AI"

    def validate_required_fields(self, report_data):
        """
        Проверяет наличие всех обязательных полей согласно требованиям
        """
        required_fields = {
            'atelectasis_probability': 'Вероятность ателектаза (0-1)',
            'bbox': 'Локализация (xmin, ymin, xmax, ymax)',
            'conclusion': 'Текстовое заключение'
        }

        missing_fields = []

        for field, description in required_fields.items():
            if field not in report_data or report_data[field] is None:
                missing_fields.append(f"{field} ({description})")

        # Специальная проверка для bbox
        if 'bbox' in report_data and report_data['bbox']:
            if not isinstance(report_data['bbox'], list) or len(report_data['bbox']) != 4:
                missing_fields.append("bbox (должен содержать 4 координаты: xmin, ymin, xmax, ymax)")

        # Проверка вероятности
        if 'atelectasis_probability' in report_data:
            prob = report_data['atelectasis_probability']
            if not isinstance(prob, (int, float)) or not (0 <= prob <= 1):
                missing_fields.append("atelectasis_probability (должна быть числом от 0 до 1)")

        return missing_fields

    def create_basic_sr_dataset(self, original_ds, report_data):
        """
        Создает базовый DICOM SR dataset с обязательными полями согласно требованиям
        """
        # Проверяем обязательные поля
        missing_fields = self.validate_required_fields(report_data)
        if missing_fields:
            raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing_fields)}")

        # File meta information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.88.22'  # Enhanced SR
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Main dataset
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Устанавливаем кодировку для поддержки русского языка
        ds.SpecificCharacterSet = 'ISO_IR 192'  # UTF-8

        # Patient Module - анонимные данные, если не найдены в оригинале
        ds.PatientName = getattr(original_ds, 'PatientName', "Anonymous^Patient")
        ds.PatientID = getattr(original_ds, 'PatientID', "ANON_ID_001")

        if hasattr(original_ds, 'PatientBirthDate'):
            ds.PatientBirthDate = original_ds.PatientBirthDate
        if hasattr(original_ds, 'PatientSex'):
            ds.PatientSex = original_ds.PatientSex

        # General Study Module
        ds.StudyInstanceUID = getattr(original_ds, 'StudyInstanceUID', generate_uid())
        ds.StudyDate = getattr(original_ds, 'StudyDate', datetime.now().strftime('%Y%m%d'))
        ds.StudyTime = getattr(original_ds, 'StudyTime', datetime.now().strftime('%H%M%S'))
        ds.ReferringPhysicianName = ""
        ds.StudyID = getattr(original_ds, 'StudyID', "1")
        ds.AccessionNumber = getattr(original_ds, 'AccessionNumber', "")

        # SR Document Series Module
        ds.Modality = "SR"  # ОБЯЗАТЕЛЬНОЕ: Structured Report
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 999
        ds.SeriesDate = datetime.now().strftime('%Y%m%d')
        ds.SeriesTime = datetime.now().strftime('%H%M%S')
        ds.SeriesDescription = "AI Atelectasis Analysis Report"

        # General Equipment Module
        ds.Manufacturer = self.manufacturer
        ds.ManufacturerModelName = "AI Atelectasis Detector"
        ds.SoftwareVersions = self.software_version
        ds.DeviceSerialNumber = "AI-DETECT-001"

        # SR Document General Module
        ds.ContentDate = datetime.now().strftime('%Y%m%d')
        ds.ContentTime = datetime.now().strftime('%H%M%S.%f')[:-3]
        ds.InstanceNumber = 1
        ds.CompletionFlag = "COMPLETE"
        ds.VerificationFlag = "UNVERIFIED"

        # Document Title
        ds.ConceptNameCodeSequence = [self._create_code("18748-4", "LN", "Diagnostic imaging report")]

        # SOP Common Module
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

        # ОБЯЗАТЕЛЬНОЕ ПОЛЕ: ObserverType
        ds.ObserverType = self.observer_type

        # Referenced Image (ссылка на оригинальное изображение)
        if hasattr(original_ds, 'SOPInstanceUID'):
            self._add_referenced_image(ds, original_ds)

        return ds

    def _add_referenced_image(self, ds, original_ds):
        """
        Добавляет ссылку на оригинальное изображение
        """
        ref_image = Dataset()
        ref_image.ReferencedSOPClassUID = getattr(original_ds, 'SOPClassUID', '1.2.840.10008.5.1.4.1.1.1.1')
        ref_image.ReferencedSOPInstanceUID = original_ds.SOPInstanceUID

        ds.CurrentRequestedProcedureEvidenceSequence = [Dataset()]
        ds.CurrentRequestedProcedureEvidenceSequence[0].StudyInstanceUID = ds.StudyInstanceUID
        ds.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence = [Dataset()]
        ds.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence[0].SeriesInstanceUID = getattr(
            original_ds, 'SeriesInstanceUID', generate_uid()
        )
        ds.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence[0].ReferencedImageSequence = [
            ref_image]

    def _create_code(self, code_value, coding_scheme, code_meaning):
        """
        Создает стандартный код для DICOM SR
        """
        code_item = Dataset()
        code_item.CodeValue = code_value
        code_item.CodingSchemeDesignator = coding_scheme
        code_item.CodeMeaning = code_meaning
        return code_item

    def _create_text_content(self, relationship_type, concept_code, text_value):
        """
        Создает текстовый элемент контента с проверкой длины
        """
        content_item = Dataset()
        content_item.RelationshipType = relationship_type
        content_item.ValueType = "TEXT"
        content_item.ConceptNameCodeSequence = [concept_code]

        # DICOM ограничение для TextValue (максимум 1024 символа)
        if len(text_value) > 1024:
            text_value = text_value[:1021] + "..."

        content_item.TextValue = text_value
        return content_item

    def _create_num_content(self, relationship_type, concept_code, numeric_value, unit_code=None):
        """
        Создает числовой элемент контента с проверкой формата
        """
        content_item = Dataset()
        content_item.RelationshipType = relationship_type
        content_item.ValueType = "NUM"
        content_item.ConceptNameCodeSequence = [concept_code]

        # Создаем последовательность измеренных значений
        measured_value = Dataset()

        # Форматируем числовое значение для DS VR (максимум 16 символов)
        if isinstance(numeric_value, float):
            numeric_str = f"{numeric_value:.6f}".rstrip('0').rstrip('.')
        else:
            numeric_str = str(numeric_value)

        if len(numeric_str) > 16:
            numeric_str = f"{float(numeric_value):.4f}"

        measured_value.NumericValue = numeric_str

        if unit_code:
            measured_value.MeasurementUnitsCodeSequence = [unit_code]

        content_item.MeasuredValueSequence = [measured_value]
        return content_item

    def _create_spatial_coordinates(self, bbox, reference_uid=None):
        """
        ОБЯЗАТЕЛЬНОЕ ПОЛЕ: Создает пространственные координаты для bbox в DICOM-координатах
        Формат: [xmin, ymin, xmax, ymax] в пикселях
        """
        if not bbox or len(bbox) != 4:
            raise ValueError("BoundingBox должен содержать 4 координаты: [xmin, ymin, xmax, ymax]")

        # Проверяем, что координаты валидны
        x_min, y_min, x_max, y_max = bbox
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"Некорректные координаты bbox: xmin({x_min}) >= xmax({x_max}) или ymin({y_min}) >= ymax({y_max})")

        coord_item = Dataset()
        coord_item.RelationshipType = "CONTAINS"
        coord_item.ValueType = "SCOORD"
        coord_item.ConceptNameCodeSequence = [self._create_code("111030", "DCM", "Image Region")]

        # Графический тип - прямоугольник (замкнутый полигон)
        coord_item.GraphicType = "POLYLINE"

        # Координаты прямоугольника в DICOM координатах (порядок: x1,y1, x2,y1, x2,y2, x1,y2, x1,y1)
        coord_item.GraphicData = [
            float(x_min), float(y_min),  # Левый верхний угол
            float(x_max), float(y_min),  # Правый верхний угол
            float(x_max), float(y_max),  # Правый нижний угол
            float(x_min), float(y_max),  # Левый нижний угол
            float(x_min), float(y_min)  # Замыкание контура
        ]

        # Добавляем ссылку на изображение
        if reference_uid:
            coord_item.ReferencedImageSequence = [Dataset()]
            coord_item.ReferencedImageSequence[0].ReferencedSOPInstanceUID = reference_uid

        return coord_item

    def _create_bbox_text_description(self, bbox):
        """
        Создает текстовое описание координат bounding box
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        description = f"Локализация в DICOM-координатах: xmin={x_min}, ymin={y_min}, xmax={x_max}, ymax={y_max} (размер: {width}x{height} пикселей)"
        return description

    def add_content_sequence(self, ds, report_data, original_ds):
        """
        Добавляет содержимое отчета в DICOM SR со всеми обязательными полями
        """
        content_sequence = []

        # Создаем корневой контейнер
        root_container = Dataset()
        root_container.RelationshipType = "CONTAINS"
        root_container.ValueType = "CONTAINER"
        root_container.ConceptNameCodeSequence = [self._create_code("18748-4", "LN", "Diagnostic imaging report")]
        root_container.ContinuityOfContent = "SEPARATE"
        root_container.ContentSequence = []

        # 1. ОБЯЗАТЕЛЬНОЕ ПОЛЕ: Modality (источник изображения)
        original_modality = getattr(original_ds, 'Modality', 'DX')  # По умолчанию DX для рентгена
        modality_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121139", "DCM", "Modality"),
            original_modality
        )
        root_container.ContentSequence.append(modality_item)

        # 2. ОБЯЗАТЕЛЬНОЕ ПОЛЕ: ObserverType
        observer_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121005", "DCM", "Observer Type"),
            self.observer_type
        )
        root_container.ContentSequence.append(observer_item)

        # 3. ОБЯЗАТЕЛЬНОЕ ПОЛЕ: Findings (текстовое заключение)
        findings_text = report_data['conclusion']

        # Добавляем обязательное предупреждение
        warning_text = "Заключение сгенерировано ИИ. Требуется подтверждение врача."
        if not findings_text.endswith('.'):
            findings_text += '.'
        findings_text += f" {warning_text}"

        findings_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121071", "DCM", "Finding"),
            findings_text
        )
        root_container.ContentSequence.append(findings_item)

        # 4. ОБЯЗАТЕЛЬНОЕ ПОЛЕ: Probability (вероятность ателектаза 0-1)
        probability_item = self._create_num_content(
            "CONTAINS",
            self._create_code("113011", "DCM", "Atelectasis Probability"),
            report_data['atelectasis_probability'],
            self._create_code("1", "UCUM", "Unity")  # Безразмерная величина
        )
        root_container.ContentSequence.append(probability_item)

        # 5. ОБЯЗАТЕЛЬНОЕ ПОЛЕ: BoundingBox (локализация в DICOM-координатах)
        if report_data['bbox'] and len(report_data['bbox']) == 4:
            # Текстовое описание координат
            bbox_description = self._create_bbox_text_description(report_data['bbox'])
            bbox_text_item = self._create_text_content(
                "CONTAINS",
                self._create_code("111001", "DCM", "Bounding Box Coordinates"),
                bbox_description
            )
            root_container.ContentSequence.append(bbox_text_item)

            # Пространственные координаты
            reference_uid = getattr(original_ds, 'SOPInstanceUID', None)
            bbox_spatial_item = self._create_spatial_coordinates(report_data['bbox'], reference_uid)
            root_container.ContentSequence.append(bbox_spatial_item)

        # 6. Дополнительные поля (не обязательные, но полезные)

        # Статус анализа
        status_item = self._create_text_content(
            "CONTAINS",
            self._create_code("33999-4", "LN", "Status"),
            report_data.get('status', 'completed')
        )
        root_container.ContentSequence.append(status_item)

        # Локализация в анатомических терминах (если есть)
        if report_data.get('location'):
            location_item = self._create_text_content(
                "CONTAINS",
                self._create_code("363698007", "SCT", "Finding site"),
                report_data['location']
            )
            root_container.ContentSequence.append(location_item)

        # Техническая информация о системе ИИ
        tech_container = Dataset()
        tech_container.RelationshipType = "CONTAINS"
        tech_container.ValueType = "CONTAINER"
        tech_container.ConceptNameCodeSequence = [self._create_code("113876", "DCM", "Device")]
        tech_container.ContinuityOfContent = "SEPARATE"
        tech_container.ContentSequence = []

        # Информация о системе
        system_item = self._create_text_content(
            "CONTAINS",
            self._create_code("113878", "DCM", "Device Serial Number"),
            f"{self.manufacturer} {self.software_version}"
        )
        tech_container.ContentSequence.append(system_item)

        # Время анализа
        analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_item = self._create_text_content(
            "CONTAINS",
            self._create_code("111526", "DCM", "DateTime Started"),
            analysis_time
        )
        tech_container.ContentSequence.append(time_item)

        root_container.ContentSequence.append(tech_container)

        # Добавляем корневой контейнер в последовательность контента
        content_sequence.append(root_container)
        ds.ContentSequence = content_sequence

        return ds

    def generate_sr_from_json(self, json_path, original_dicom_path, output_path):
        """
        Генерирует DICOM SR из JSON отчета с соблюдением всех обязательных полей
        """
        try:
            # Загружаем JSON отчет
            with open(json_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            print("=== ПРОВЕРКА ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ ===")

            # Проверяем обязательные поля
            missing_fields = self.validate_required_fields(report_data)
            if missing_fields:
                raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing_fields)}")

            print("✓ Все обязательные поля присутствуют")
            print(f"✓ Вероятность ателектаза: {report_data['atelectasis_probability']:.3f}")
            print(f"✓ Координаты bbox: {report_data['bbox']}")
            print(f"✓ Заключение: {report_data['conclusion'][:100]}...")

            # Загружаем оригинальный DICOM
            original_ds = dcmread(original_dicom_path)
            print(f"✓ Оригинальный DICOM загружен: {getattr(original_ds, 'Modality', 'N/A')}")

            # Создаем базовый SR dataset
            sr_ds = self.create_basic_sr_dataset(original_ds, report_data)

            # Добавляем содержимое с обязательными полями
            sr_ds = self.add_content_sequence(sr_ds, report_data, original_ds)

            # Сохраняем
            sr_ds.save_as(output_path)
            print(f"✓ DICOM SR успешно создан: {output_path}")

            # Выводим сводку по обязательным полям
            print("\n=== СВОДКА ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ В DICOM SR ===")
            print(f"• Modality: {getattr(original_ds, 'Modality', 'DX')}")
            print(f"• ObserverType: {sr_ds.ObserverType}")
            print(f"• Findings: {report_data['conclusion'][:80]}...")
            print(f"• Probability: {report_data['atelectasis_probability']:.6f}")
            if report_data['bbox']:
                x_min, y_min, x_max, y_max = report_data['bbox']
                print(f"• BoundingBox: xmin={x_min}, ymin={y_min}, xmax={x_max}, ymax={y_max}")
            print(f"• Warning: Заключение сгенерировано ИИ. Требуется подтверждение врача.")
            print("==========================================\n")

            return True

        except Exception as e:
            print(f"❌ Ошибка при создании DICOM SR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def generate_json_report(self, json_input_path, json_output_path):
        """
        Генерирует стандартизированный JSON отчет для API с обязательными полями
        """
        try:
            # Загружаем исходный JSON
            with open(json_input_path, 'r', encoding='utf-8') as f:
                source_data = json.load(f)

            # Проверяем обязательные поля
            missing_fields = self.validate_required_fields(source_data)
            if missing_fields:
                raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing_fields)}")

            # Создаем стандартизированный отчет
            standardized_report = {
                # ОБЯЗАТЕЛЬНЫЕ ПОЛЯ согласно требованиям
                "atelectasis_probability": source_data['atelectasis_probability'],
                "localization": {
                    "xmin": source_data['bbox'][0],
                    "ymin": source_data['bbox'][1],
                    "xmax": source_data['bbox'][2],
                    "ymax": source_data['bbox'][3],
                    "coordinate_system": "DICOM"
                },
                "conclusion": source_data['conclusion'],
                "warning": "Заключение сгенерировано ИИ. Требуется подтверждение врача.",

                # Дополнительные поля
                "status": source_data.get('status', 'completed'),
                "location_description": source_data.get('location', ''),
                "other_pathologies": source_data.get('other_pathologies', []),
                "confidence_levels": source_data.get('confidence_levels', []),

                # Метаданные
                "metadata": {
                    "observer_type": self.observer_type,
                    "software_version": self.software_version,
                    "manufacturer": self.manufacturer,
                    "generation_timestamp": datetime.now().isoformat(),
                    "original_metadata": source_data.get('metadata', {})
                },

                # API-специфичные поля
                "api_version": "1.0",
                "report_type": "atelectasis_detection",
                "format": "JSON"
            }

            # Сохраняем стандартизированный отчет
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(standardized_report, f, ensure_ascii=False, indent=4)

            print(f"✓ JSON отчет для API создан: {json_output_path}")
            return True

        except Exception as e:
            print(f"❌ Ошибка при создании JSON отчета: {str(e)}")
            return False


def generate_dicom_sr_from_json(json_path, original_dicom_path, output_path):
    """
    Функция для обратной совместимости с основным пайплайном
    """
    generator = DicomSRGenerator()
    return generator.generate_sr_from_json(json_path, original_dicom_path, output_path)


def generate_json_api_report(json_input_path, json_output_path):
    """
    Генерирует стандартизированный JSON отчет для API
    """
    generator = DicomSRGenerator()
    return generator.generate_json_report(json_input_path, json_output_path)


# Для автономного тестирования
if __name__ == "__main__":
    # Пути к файлам
    json_report_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_json\post_json\test_atelectasis_5.json"
    original_dicom_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_dicom\pre_dicom\test_atelectasis_5.dcm"
    output_sr_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\results\dicom_sr\test_atelectasis_5_sr.dcm"
    output_json_api_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\results\json_api\test_atelectasis_5_api.json"

    # Создаем директории если не существуют
    os.makedirs(os.path.dirname(output_sr_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_api_path), exist_ok=True)

    generator = DicomSRGenerator()

    print("=== ГЕНЕРАЦИЯ DICOM SR ===")
    success_sr = generator.generate_sr_from_json(json_report_path, original_dicom_path, output_sr_path)

    print("\n=== ГЕНЕРАЦИЯ JSON ДЛЯ API ===")
    success_json = generator.generate_json_report(json_report_path, output_json_api_path)

    if success_sr and success_json:
        print("\n✅ Все отчеты созданы успешно!")
        print(f"📄 DICOM SR: {output_sr_path}")
        print(f"📄 JSON API: {output_json_api_path}")
    else:
        print("\n❌ Некоторые отчеты не были созданы")
        if success_sr:
            print(f"✅ DICOM SR создан: {output_sr_path}")
        if success_json:
            print(f"✅ JSON API создан: {output_json_api_path}")