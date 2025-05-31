import os
import json
import torch
import numpy as np
from datetime import datetime
from B_dicom_handler import DicomHandler
from D_classify_and_detect_json import process_dicom_with_ai
from sam_proekt.E_Dicom_generation import generate_dicom_sr_from_json


class AtelectasisPipeline:
    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = output_dir
        self.dicom_handler = DicomHandler()

        # Создаем директории для результатов
        self.json_dir = os.path.join(output_dir, "json_reports")
        self.sr_dir = os.path.join(output_dir, "dicom_sr")
        self.log_dir = os.path.join(output_dir, "logs")

        for dir_path in [self.json_dir, self.sr_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def process_dicom(self, dicom_path):
        """
        Полный пайплайн обработки DICOM файла
        """
        print(f"[1/5] Обработка файла: {dicom_path}")

        # 1. Предобработка и анонимизация
        print("[2/5] Предобработка и анонимизация...")
        img, metadata = self.dicom_handler.preprocess_dicom(dicom_path)

        # 2. Анализ нейросетью
        print("[3/5] Анализ изображения нейросетью...")
        ai_results = process_dicom_with_ai(
            dicom_path,
            self.model_path,
            save_json=False  # Мы сохраним сами
        )

        # 3. Добавляем метаданные и предупреждение
        ai_results["metadata"] = metadata
        ai_results["warning"] = "Заключение сгенерировано ИИ. Требуется подтверждение врача."
        ai_results["processing_timestamp"] = datetime.now().isoformat()

        # 4. Сохраняем JSON отчет
        base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        json_path = os.path.join(self.json_dir, f"{base_name}_report.json")

        print(f"[4/5] Сохранение JSON отчета: {json_path}")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ai_results, f, ensure_ascii=False, indent=4)

        # 5. Генерация DICOM SR
        sr_path = os.path.join(self.sr_dir, f"{base_name}_sr.dcm")
        print(f"[5/5] Генерация DICOM SR: {sr_path}")

        success = generate_dicom_sr_from_json(json_path, dicom_path, sr_path)

        if success:
            print(f"[✓] Обработка завершена успешно!")
            return {
                "status": "success",
                "json_report": json_path,
                "dicom_sr": sr_path,
                "results": ai_results
            }
        else:
            print(f"[✗] Ошибка при создании DICOM SR")
            return {
                "status": "partial_success",
                "json_report": json_path,
                "results": ai_results
            }


def main():
    # Настройки
    model_path = r"C:\Users\CYBER ARTEL\PycharmProjects\atelectasis_classification_and_detection\best_deit_scm_model — копия.pth"
    output_dir = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\results"
    dicom_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_dicom\pre_dicom\test_atelectasis_5.dcm"

    # Создаем пайплайн
    pipeline = AtelectasisPipeline(model_path, output_dir)

    # Обрабатываем файл
    result = pipeline.process_dicom(dicom_path)

    # Выводим результаты
    if result["status"] == "success":
        print("\n=== РЕЗУЛЬТАТЫ ===")
        print(f"Статус: {result['results']['status']}")
        print(f"Вероятность ателектаза: {result['results']['atelectasis_probability']:.2%}")
        if result['results']['bbox']:
            print(f"Локализация: {result['results']['bbox']}")
        print(f"Заключение: {result['results']['conclusion']}")
        print(f"\nJSON отчет: {result['json_report']}")
        print(f"DICOM SR: {result['dicom_sr']}")


if __name__ == "__main__":
    main()