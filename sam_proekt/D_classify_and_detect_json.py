import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pydicom
import cv2
import json
from torchvision import transforms
from Neural_nets_training.B_training_teacher_classifier import DeiTClassifierWSOL

# --- Настройки ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = r"C:\Users\CYBER ARTEL\PycharmProjects\atelectasis_classification_and_detection\best_deit_scm_model — копия.pth"
dicom_file_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_dicom\pre_dicom\test_atelectasis_2.dcm"

# --- Трансформации изображения ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Функция для преобразования тепловой карты в bbox ---
def heatmap_to_bbox(heatmap, threshold=0.5):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_bin = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return [x, y, x + w, y + h]

# --- Загрузка модели ---
model = DeiTClassifierWSOL(num_classes=3, pretrained=False, scm_blocks=4)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# --- Загрузка и обработка DICOM изображения ---
def load_dicom_image(dicom_file_path):
    # Чтение DICOM файла
    dicom_data = pydicom.dcmread(dicom_file_path)
    # Извлечение изображения из DICOM (полагаем, что оно в пикселях)
    image_array = dicom_data.pixel_array
    # Преобразуем в изображение PIL для дальнейшей обработки
    return Image.fromarray(image_array)

# Загрузка и преобразование DICOM изображения
original_image = load_dicom_image(dicom_file_path).convert("RGB")
image_tensor = transform(original_image).unsqueeze(0).to(device)

# --- Получение тепловой карты ---
with torch.no_grad():
    heatmap = model.localize(image_tensor)[0].cpu().numpy()

# --- Интерполяция тепловой карты ---
heatmap_up = F.interpolate(
    torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
    size=(224, 224),
    mode='bilinear', align_corners=False
)[0, 0].numpy()

# --- Получение bbox ---
predicted_bbox = heatmap_to_bbox(heatmap_up, threshold=0.4)

# --- Классификация: вероятность ателектаза и других патологий ---
output = model(image_tensor)
probabilities = F.softmax(output, dim=1)  # Указываем dim явно
atelectasis_prob = probabilities[0][0].item()  # Вероятность ателектаза (класс 0)
normal_prob = probabilities[0][1].item()  # Вероятность нормы (класс 1)
other_pathologies_prob = probabilities[0][2].item()  # Вероятность иных патологий (класс 2)

# Определяем статус
if atelectasis_prob >= 0.7:
    status = "atelectasis_only"
    conclusion = "Обнаружены признаки ателектаза. Требуется подтверждение врача."
    other_pathologies = []  # Нет других патологий
elif normal_prob >= 0.7:
    status = "normal"
    conclusion = f"Признаков ателектаза не обнаружено. Вероятность: {atelectasis_prob:.2f}"
    other_pathologies = []  # Нет других патологий
elif other_pathologies_prob >= 0.7:
    status = "other_pathologies"
    conclusion = "Обнаружены другие патологии. Требуется подтверждение врача."
    other_pathologies = ["other_pathologies"]  # Могут быть и другие патологии
else:
    status = "normal"
    conclusion = f"Признаков ателектаза не обнаружено. Вероятность: {atelectasis_prob:.2f}"
    other_pathologies = []  # Нет других патологий

# --- Подготовка вывода ---
output_json = {
    "status": status,
    "atelectasis_probability": atelectasis_prob,
    "bbox": predicted_bbox if predicted_bbox else [],
    "conclusion": conclusion,
    "other_pathologies": other_pathologies
}

# Получаем имя файла без расширения
dicom_filename = os.path.splitext(os.path.basename(dicom_file_path))[0]

# Указываем директорию для сохранения
save_dir = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_json\post_json"

# Формируем полный путь для JSON файла, заменяя расширение на .json
output_json_path = os.path.join(save_dir, dicom_filename + ".json")

# Сохраняем результат в файл JSON
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(output_json, json_file, ensure_ascii=False, indent=4)

print(f"Результат сохранен в файл: {output_json_path}")


# Вывод JSON
print(json.dumps(output_json, ensure_ascii=False, indent=4))
