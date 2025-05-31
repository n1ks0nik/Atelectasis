import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pydicom
import cv2
import json
from torchvision import transforms
from B_dicom_handler import DicomHandler

# Импортируем вашу модель
from Neural_nets_training.B_training_teacher_classifier import DeiTClassifierWSOL


class AtelectasisDetector:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.dicom_handler = DicomHandler()

        # Трансформации изображения
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Загрузка модели
        self.model = self._load_model()

    def _load_model(self):
        """Загружает модель из checkpoint"""
        model = DeiTClassifierWSOL(num_classes=3, pretrained=False, scm_blocks=4)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def heatmap_to_bbox(self, heatmap, threshold=0.5, original_size=(1024, 1024)):
        """
        Преобразует тепловую карту в bbox с учетом DICOM координат
        """
        # Нормализация тепловой карты
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_bin = (heatmap > threshold).astype(np.uint8) * 255

        # Поиск контуров
        contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Находим самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Масштабируем координаты к оригинальному размеру DICOM
        scale_x = original_size[0] / heatmap.shape[1]
        scale_y = original_size[1] / heatmap.shape[0]

        x_min = int(x * scale_x)
        y_min = int(y * scale_y)
        x_max = int((x + w) * scale_x)
        y_max = int((y + h) * scale_y)

        return [x_min, y_min, x_max, y_max]

    def classify_pathology(self, probabilities):
        """
        Классифицирует патологию на основе вероятностей
        """
        atelectasis_prob = probabilities[0][0].item()
        normal_prob = probabilities[0][1].item()
        other_pathologies_prob = probabilities[0][2].item()

        # Определяем статус согласно требованиям
        if normal_prob >= 0.9:
            status = "normal"
            atelectasis_str = f"{atelectasis_prob * 100:.1f}%"
            conclusion = f"Признаков ателектаза не обнаружено. Вероятность: {atelectasis_str}"
            other_pathologies = []
        elif atelectasis_prob >= 0.7:
            status = "atelectasis_only"
            atelectasis_str = f"{atelectasis_prob * 100:.1f}%"
            conclusion = f"Обнаружены признаки ателектаза (вероятность: {atelectasis_str}). Требуется подтверждение врача."
            other_pathologies = []
        elif other_pathologies_prob >= 0.3:
            status = "other_pathologies"
            atelectasis_str = f"{atelectasis_prob * 100:.1f}%"
            # Примеры других патологий (в реальной системе нужна более детальная классификация)
            possible_pathologies = ["pleural_effusion", "pneumothorax", "consolidation"]
            # Генерируем случайные вероятности для демонстрации
            confidence_levels = [f"{np.random.uniform(0.3, 0.8) * 100:.0f}%" for _ in possible_pathologies[:2]]
            other_pathologies = possible_pathologies[:2]
            conclusion = f"Обнаружены другие патологии. Вероятность ателектаза: {atelectasis_str}. Требуется подтверждение врача."
        else:
            status = "normal"
            atelectasis_str = f"{atelectasis_prob * 100:.1f}%"
            conclusion = f"Признаков ателектаза не обнаружено. Вероятность: {atelectasis_str}"
            other_pathologies = []

        return {
            "status": status,
            "atelectasis_probability": atelectasis_prob,
            "atelectasis_probability_str": atelectasis_str,
            "conclusion": conclusion,
            "other_pathologies": other_pathologies,
            "confidence_levels": confidence_levels if status == "other_pathologies" else []
        }

    def process_dicom(self, dicom_path):
        """
        Обрабатывает DICOM файл и возвращает результаты анализа
        """
        # Загрузка DICOM
        ds = pydicom.dcmread(dicom_path)
        original_size = (ds.Columns, ds.Rows) if hasattr(ds, 'Columns') and hasattr(ds, 'Rows') else (1024, 1024)

        # Предобработка изображения
        img_array, metadata = self.dicom_handler.preprocess_dicom(dicom_path)

        # Преобразование в PIL Image для трансформаций
        img_pil = Image.fromarray(img_array)
        image_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # Получение предсказаний
        with torch.no_grad():
            # Классификация
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)

            # Локализация
            heatmap = self.model.localize(image_tensor)[0].cpu().numpy()

        # Интерполяция тепловой карты
        heatmap_up = F.interpolate(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear', align_corners=False
        )[0, 0].numpy()

        # Классификация патологии
        classification_result = self.classify_pathology(probabilities)

        # Получение bbox только если обнаружен ателектаз
        bbox = None
        location = None
        if classification_result["status"] == "atelectasis_only":
            bbox = self.heatmap_to_bbox(heatmap_up, threshold=0.4, original_size=original_size)
            if bbox:
                # Определяем локализацию на основе координат
                x_center = (bbox[0] + bbox[2]) / 2 / original_size[0]
                y_center = (bbox[1] + bbox[3]) / 2 / original_size[1]

                if x_center < 0.5:
                    side = "left"
                else:
                    side = "right"

                if y_center < 0.33:
                    zone = "upper"
                elif y_center < 0.66:
                    zone = "middle"
                else:
                    zone = "lower"

                location = f"{zone} zone, {side} lung"

        # Формируем финальный результат
        result = {
            **classification_result,
            "bbox": bbox if bbox else [],
            "location": location,
            "warning": "Заключение сгенерировано ИИ. Требуется подтверждение врача.",
            "metadata": metadata
        }

        return result


def process_dicom_with_ai(dicom_path, checkpoint_path, save_json=True, output_dir=None):
    """
    Функция для обратной совместимости
    """
    detector = AtelectasisDetector(checkpoint_path)
    result = detector.process_dicom(dicom_path)

    if save_json and output_dir:
        # Сохраняем результат в JSON
        dicom_filename = os.path.splitext(os.path.basename(dicom_path))[0]
        output_json_path = os.path.join(output_dir, dicom_filename + "_report.json")

        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

        print(f"Результат сохранен в файл: {output_json_path}")

    return result


# Для автономного запуска
if __name__ == "__main__":
    # Настройки
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = r"C:\Users\CYBER ARTEL\PycharmProjects\atelectasis_classification_and_detection\best_deit_scm_model — копия.pth"
    dicom_file_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_dicom\pre_dicom\test_atelectasis_5.dcm"
    output_dir = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_json\post_json"

    # Обработка
    detector = AtelectasisDetector(checkpoint_path, device)
    result = detector.process_dicom(dicom_file_path)

    # Сохранение результата
    os.makedirs(output_dir, exist_ok=True)
    dicom_filename = os.path.splitext(os.path.basename(dicom_file_path))[0]
    output_json_path = os.path.join(output_dir, dicom_filename + ".json")

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    # Вывод результата
    print(json.dumps(result, ensure_ascii=False, indent=4))