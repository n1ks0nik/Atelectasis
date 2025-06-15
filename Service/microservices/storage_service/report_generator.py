import os
import json
from datetime import datetime
from pydicom import dcmread, Dataset
from pydicom.uid import generate_uid
from pydicom.dataset import FileDataset
import pydicom.uid
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        ds.Modality = "SR"
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 999
        ds.SeriesDate = datetime.now().strftime('%Y%m%d')
        ds.SeriesTime = datetime.now().strftime('%H%M%S')
        ds.SeriesDescription = "Анализ Ателектаза ИИ"

        # General Equipment Module
        ds.Manufacturer = self.manufacturer
        ds.ManufacturerModelName = "Анализ Ателектаза ИИ"
        ds.SoftwareVersions = self.software_version
        ds.DeviceSerialNumber = "AI-DETECT-001"

        # SR Document General Module
        ds.ContentDate = datetime.now().strftime('%Y%m%d')
        ds.ContentTime = datetime.now().strftime('%H%M%S.%f')[:-3]
        ds.InstanceNumber = 1
        ds.CompletionFlag = "COMPLETE"
        ds.VerificationFlag = "UNVERIFIED"

        # Document Title
        ds.ConceptNameCodeSequence = [self._create_code("18748-4", "LN", "Диагностический отчет")]

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
        coord_item.ConceptNameCodeSequence = [self._create_code("111030", "DCM", "Область изображения")]

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
        root_container.ConceptNameCodeSequence = [self._create_code("18748-4", "LN", "Диагностический отчет")]
        root_container.ContinuityOfContent = "SEPARATE"
        root_container.ContentSequence = []

        # Modality
        original_modality = getattr(original_ds, 'Modality', 'DX')  # По умолчанию DX для рентгена
        modality_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121139", "DCM", "Модальность"),
            original_modality
        )
        root_container.ContentSequence.append(modality_item)

        # ObserverType
        observer_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121005", "DCM", "Тип наблюдателя"),
            self.observer_type
        )
        root_container.ContentSequence.append(observer_item)

        # Статус анализа
        status_translation = {
            'normal': 'Норма',
            'atelectasis_only': 'Ателектаз',
            'other_pathologies': 'Другие патологии'
        }
        status_text = status_translation.get(report_data['status'], report_data['status'])

        status_item = self._create_text_content(
            "CONTAINS",
            self._create_code("33999-4", "LN", "Статус"),
            status_text
        )
        root_container.ContentSequence.append(status_item)

        # Probability (вероятность ателектаза 0-1)
        probability_item = self._create_num_content(
            "CONTAINS",
            self._create_code("113011", "DCM", "Вероятность ателектаза"),
            report_data['atelectasis_probability'],
            self._create_code("1", "UCUM", "(безразмерная величина)")  # Безразмерная величина
        )
        root_container.ContentSequence.append(probability_item)

        # Findings (текстовое заключение)
        findings_text = report_data['conclusion']

        # Добавляем обязательное предупреждение
        warning_text = "Заключение сгенерировано ИИ. Требуется подтверждение врача."
        if not findings_text.endswith('.'):
            findings_text += '.'
        findings_text += f" {warning_text}"

        findings_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121071", "DCM", "Заключение"),
            findings_text
        )
        root_container.ContentSequence.append(findings_item)

        # Локализация в анатомических терминах (если есть)
        if report_data.get('location'):

            location_translation = {
                'upper lobe, right lung': 'Верхняя доля правого легкого',
                'middle lobe, right lung': 'Средняя доля правого легкого',
                'lower lobe, right lung': 'Нижняя доля правого легкого',
                'upper lobe, left lung': 'Верхняя доля левого легкого',
                'lower lobe, left lung': 'Нижняя доля левого легкого',
                'middle zone, right lung': 'Средняя зона правого легкого',
                'middle zone, left lung': 'Средняя зона левого легкого',
                'lower zone, right lung': 'Нижняя зона правого легкого',
                'lower zone, left lung': 'Нижняя зона левого легкого'
            }

            russian_location = location_translation.get(
                report_data['location'].lower(),
                report_data['location']
            )

            location_item = self._create_text_content(
                "CONTAINS",
                self._create_code("363698007", "SCT", "Локализация"),
                russian_location
            )
            root_container.ContentSequence.append(location_item)

        # Другие патологии (если есть)
        if report_data.get('other_pathologies_prob', 0) >= 0.3:
            pathologies_item = self._create_num_content(
                "CONTAINS",
                self._create_code("363698007", "SCT", "Вероятность других патологий"),
                report_data['other_pathologies_prob'],
                self._create_code("2", "UCUM", "(безразмерная величина)")
            )
            root_container.ContentSequence.append(pathologies_item)

        # BoundingBox (локализация в DICOM-координатах)
        if report_data.get('bbox') and len(report_data['bbox']) == 4:
            # Текстовое описание координат
            bbox_description = self._create_bbox_text_description(report_data['bbox'])
            bbox_text_item = self._create_text_content(
                "CONTAINS",
                self._create_code("111001", "DCM", "Координаты области патологии"),
                bbox_description
            )
            root_container.ContentSequence.append(bbox_text_item)

            # Пространственные координаты
            reference_uid = getattr(original_ds, 'SOPInstanceUID', None)
            bbox_spatial_item = self._create_spatial_coordinates(report_data['bbox'], reference_uid)
            root_container.ContentSequence.append(bbox_spatial_item)

        # Предупреждение и отказ от ответственности
        warning_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121130", "DCM", "Предупреждение"),
            "Заключение сгенерировано ИИ. Требуется подтверждение врача."
        )
        root_container.ContentSequence.append(warning_item)

        disclaimer_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121131", "DCM", "Отказ от ответственности"),
            "Для поддержки принятия врачебных решений. Не заменяет консультацию специалиста. Только для исследовательских целей."
        )
        root_container.ContentSequence.append(disclaimer_item)

        # Техническая информация о системе ИИ
        tech_container = Dataset()
        tech_container.RelationshipType = "CONTAINS"
        tech_container.ValueType = "CONTAINER"
        tech_container.ConceptNameCodeSequence = [self._create_code("113876", "DCM", "Технические данные")]
        tech_container.ContinuityOfContent = "SEPARATE"
        tech_container.ContentSequence = []

        # Информация о системе
        system_item = self._create_text_content(
            "CONTAINS",
            self._create_code("113878", "DCM", "Система"),
            f"{self.manufacturer} {self.software_version}"
        )
        tech_container.ContentSequence.append(system_item)

        # Время анализа
        analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_item = self._create_text_content(
            "CONTAINS",
            self._create_code("111526", "DCM", "Дата исследования"),
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
                    "xmin": source_data['bbox'][0] if source_data['bbox'] else None,
                    "ymin": source_data['bbox'][1] if source_data['bbox'] else None,
                    "xmax": source_data['bbox'][2] if source_data['bbox'] else None,
                    "ymax": source_data['bbox'][3] if source_data['bbox'] else None,
                    "coordinate_system": "DICOM"
                },
                "conclusion": source_data['conclusion'],
                "warning": "Заключение сгенерировано ИИ. Требуется подтверждение врача.",

                # Дополнительные поля
                "status": source_data.get('status', 'completed'),
                "location_description": source_data.get('location', ''),
                "other_pathologies_prob": source_data.get('other_pathologies_prob', []),

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


    def create_annotated_image_dicom(self, original_ds, bbox, atel_probability, other_pathologies_prob, output_path):
        """
        Создает DICOM с аннотированным изображением (Secondary Capture)
        """
        try:
            # Извлекаем пиксельные данные
            pixel_array = original_ds.pixel_array.astype(np.float32)

            # Применяем RescaleSlope и RescaleIntercept если есть
            if hasattr(original_ds, 'RescaleSlope') and hasattr(original_ds, 'RescaleIntercept'):
                pixel_array = pixel_array * original_ds.RescaleSlope + original_ds.RescaleIntercept

            # Нормализация к диапазону 0-255 для визуализации
            pixel_array = ((pixel_array - pixel_array.min()) /
                          (pixel_array.max() - pixel_array.min() + 1e-8) * 255).astype(np.uint8)

            # Конвертируем в RGB для рисования цветных аннотаций
            if len(pixel_array.shape) == 2:
                img_rgb = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = pixel_array

            # Рисуем bbox если есть
            if bbox and len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox
                # Красный прямоугольник
                cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

                # Добавляем текст с вероятностью
                text = f"Ателектаз: {atel_probability:.1%}"
                font = cv2.FONT_HERSHEY_COMPLEX
                font_scale = 1.0
                thickness = 2

                # Получаем размер текста
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Позиция текста (над bbox)
                text_x = x_min
                text_y = y_min - 10 if y_min - 10 > text_height else y_min + text_height + 10

                # Фон для текста
                cv2.rectangle(img_rgb,
                            (text_x - 5, text_y - text_height - 5),
                            (text_x + text_width + 5, text_y + 5),
                            (0, 0, 0), -1)

                # Сам текст (белый)
                cv2.putText(img_rgb, text, (text_x, text_y), font,
                           font_scale, (255, 255, 255), thickness)

            # Добавляем общую информацию
            if atel_probability >= 0.7:
                status = 'Обнаружен ателектаз'
            elif other_pathologies_prob >= 0.3:
                status = 'Обнаружены другие патологии'
            else:
                status = 'Норма'

            info_text = [
                f"Анализ ИИ - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Статус: {status}",
                "Только для исследовательских целей."
            ]

            y_offset = 30
            for i, line in enumerate(info_text):
                cv2.putText(img_rgb, line, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

            # Создаем новый DICOM Dataset для Secondary Capture
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            ds_annotated = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

            ds_annotated.SpecificCharacterSet = 'ISO_IR 192'

            # Копируем метаданные пациента и исследования
            ds_annotated.PatientName = getattr(original_ds, 'PatientName', "Anonymous^Patient")
            ds_annotated.PatientID = getattr(original_ds, 'PatientID', "ANON_ID_001")
            ds_annotated.PatientBirthDate = getattr(original_ds, 'PatientBirthDate', '')
            ds_annotated.PatientSex = getattr(original_ds, 'PatientSex', '')

            # Study информация (та же, что и у оригинала)
            ds_annotated.StudyInstanceUID = original_ds.StudyInstanceUID
            ds_annotated.StudyDate = original_ds.StudyDate
            ds_annotated.StudyTime = original_ds.StudyTime
            ds_annotated.AccessionNumber = getattr(original_ds, 'AccessionNumber', '')
            ds_annotated.StudyDescription = "Анализ Ателектаза ИИ"

            # Series информация (новая серия)
            ds_annotated.SeriesInstanceUID = generate_uid()
            ds_annotated.SeriesNumber = 2  # Серия 2 для аннотированных изображений
            ds_annotated.SeriesDescription = "Изображение обработано ИИ"
            ds_annotated.Modality = "OT"  # Other

            # SOP информация
            ds_annotated.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds_annotated.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds_annotated.InstanceNumber = 1

            # Image информация
            ds_annotated.SamplesPerPixel = 3  # RGB
            ds_annotated.PhotometricInterpretation = "RGB"
            ds_annotated.Rows, ds_annotated.Columns = img_rgb.shape[:2]
            ds_annotated.BitsAllocated = 8
            ds_annotated.BitsStored = 8
            ds_annotated.HighBit = 7
            ds_annotated.PixelRepresentation = 0
            ds_annotated.PlanarConfiguration = 0

            # Конвертируем изображение для DICOM
            ds_annotated.PixelData = img_rgb.tobytes()

            # Добавляем информацию об аннотации
            ds_annotated.ImageComments = f"ИИ обнаружил ателектаз с вероятностью {atel_probability:.1%}"
            ds_annotated.DerivationDescription = "Локализация патологии на основе ИИ"

            # Сохраняем
            ds_annotated.save_as(output_path)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create annotated image: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_complete_report(self, json_path, original_dicom_path, output_dir, study_id):
        """
        Генерирует полный отчет: аннотированное изображение + SR
        """
        try:
            # Загружаем данные
            with open(json_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            original_ds = dcmread(original_dicom_path)
            
            # Создаем директорию для серии
            series_dir = os.path.join(output_dir, study_id)
            os.makedirs(series_dir, exist_ok=True)
            
            results = []
            
            # Создаем аннотированное изображение
            annotated_path = os.path.join(series_dir, f"{study_id}_annotated.dcm")
            if self.create_annotated_image_dicom(
                original_ds,
                report_data.get('bbox'),
                report_data.get('atelectasis_probability', 0),
                report_data.get('other_pathologies_prob', 0),
                annotated_path
            ):
                logger.info(f"✅ Annotated image created: {annotated_path}")
                results.append(annotated_path)
            
            # 2. Создаем SR отчет
            sr_path = os.path.join(series_dir, f"{study_id}_sr.dcm")
            if self.generate_sr_from_json(json_path, original_dicom_path, sr_path):
                logger.info(f"✅ SR report created: {sr_path}")
                results.append(sr_path)
            
            # Создаем файл-манифест серии
            manifest_path = os.path.join(series_dir, "series_manifest.json")
            manifest = {
                "study_id": study_id,
                "study_instance_uid": str(original_ds.StudyInstanceUID),
                "created_at": datetime.now().isoformat(),
                "files": [os.path.basename(f) for f in results],
                "total_instances": len(results),
                "description": "AI Atelectasis Analysis Report Series"
            }
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to generate complete report: {e}")
            return []


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