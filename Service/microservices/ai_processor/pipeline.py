import os
import json
from datetime import datetime
from dicom_handler import DicomHandler
from detector import AtelectasisDetector


class AtelectasisPipeline:
    def __init__(self, model_path=None, output_dir=None):
        self.model_path = model_path or os.getenv('MODEL_PATH', './model/best_deit_scm_model.pth')
        self.output_dir = output_dir or os.getenv('OUTPUT_DIR', './output')
        self.dicom_handler = DicomHandler()
        self.detector = AtelectasisDetector(self.model_path)

        # Создаем директории для результатов
        self.json_dir = os.path.join(self.output_dir, "json_reports")
        self.log_dir = os.path.join(self.output_dir, "logs")

        for dir_path in [self.json_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def process_dicom(self, dicom_path):
        """
        Полный пайплайн обработки DICOM файла
        """
        try:
            print(f"[1/3] Обработка файла: {dicom_path}")

            # 1. Анализ нейросетью
            print("[2/3] Анализ изображения нейросетью...")
            ai_results = self.detector.process_dicom(dicom_path)

            # 2. Добавляем метаданные
            ai_results["processing_timestamp"] = datetime.now().isoformat()

            # 3. Сохраняем JSON отчет
            base_name = os.path.splitext(os.path.basename(dicom_path))[0]
            json_path = os.path.join(self.json_dir, f"{base_name}_report.json")

            print(f"[3/3] Сохранение JSON отчета: {json_path}")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(ai_results, f, ensure_ascii=False, indent=4)

            print(f"[✓] Обработка завершена успешно!")
            return {
                "status": "success",
                "json_report": json_path,
                "results": ai_results
            }

        except Exception as e:
            print(f"[✗] Ошибка при обработке: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }