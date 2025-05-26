import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from torchvision import transforms
from B_training_teacher_classifier import DeiTClassifierWSOL

# --- Настройки ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_folder = "C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\for_detection"
annotation_csv = "C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\csv_files\\Atelectasis_Bbox_List_2017.csv"
target_label = "Atelectasis"
checkpoint_path = "best_deit_scm_model — копия.pth"

# --- Преобразования ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Загрузка модели ---
model = DeiTClassifierWSOL(num_classes=3, pretrained=False, scm_blocks=4)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# --- Загрузка аннотаций ---
df = pd.read_csv(annotation_csv)
df = df[df['Finding Label'] == target_label]
bbox_dict = {row['Image Index']: [row['x'], row['y'], row['x'] + row['w'], row['y'] + row['h']]
             for _, row in df.iterrows()}

# --- IoU metric ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea != 0 else 0.0

def compute_giou(boxA, boxB):
    iou = compute_iou(boxA, boxB)
    xC1 = min(boxA[0], boxB[0])
    yC1 = min(boxA[1], boxB[1])
    xC2 = max(boxA[2], boxB[2])
    yC2 = max(boxA[3], boxB[3])
    enclose_area = max(0, xC2 - xC1) * max(0, yC2 - yC1)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    unionArea = boxAArea + boxBArea - (iou * unionArea if (unionArea := boxAArea + boxBArea - iou * boxAArea) != 0 else 0)
    giou = iou - (enclose_area - unionArea) / enclose_area
    return giou

# --- Функция: генерация bbox из heatmap ---
def heatmap_to_bbox(heatmap, threshold=0.5):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_bin = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return [x, y, x + w, y + h]

# --- Расчет расстояния Хаунсдорфа ---
def compute_hausdorff_distance(boxA, boxB):
    # Преобразование bbox в точки границ
    def bbox_to_boundary(box):
        x1, y1, x2, y2 = box
        return np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

    boundaryA = bbox_to_boundary(boxA)
    boundaryB = bbox_to_boundary(boxB)

    # Вычисление направленного расстояния Хаунсдорфа
    hausdorff_AtoB = directed_hausdorff(boundaryA, boundaryB)[0]
    hausdorff_BtoA = directed_hausdorff(boundaryB, boundaryA)[0]

    # Симметричное расстояние Хаунсдорфа
    return max(hausdorff_AtoB, hausdorff_BtoA)

# --- Расчет среднего расстояния между границами ---
# --- Расчет среднего расстояния между границами (в процентах) ---
# --- Расчет среднего расстояния между границами (в процентах) ---
def compute_average_boundary_distance(boxA, boxB, image_width, image_height):
    def bbox_to_boundary(box):
        x1, y1, x2, y2 = box
        return np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

    boundaryA = bbox_to_boundary(boxA)
    boundaryB = bbox_to_boundary(boxB)

    distances = []
    for pointA in boundaryA:
        min_dist = min(np.linalg.norm(pointA - pointB) for pointB in boundaryB)
        # Нормализуем расстояние относительно диагонали изображения
        normalized_dist = min_dist / np.sqrt(image_width**2 + image_height**2)
        distances.append(normalized_dist)

    # Среднее значение в процентах
    return np.mean(distances) * 100

# --- Основной цикл ---
os.makedirs("heatmap_comparisons", exist_ok=True)
ious = []
gious = []
hausdorff_distances = []
avg_boundary_distances = []

for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    full_path = os.path.join(image_folder, filename)
    original_image = Image.open(full_path).convert("RGB")
    w_orig, h_orig = original_image.size  # Исходные размеры изображения

    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap = model.localize(image_tensor)[0].cpu().numpy()

    heatmap_up = F.interpolate(torch.tensor(heatmap).unsqueeze(0).unsqueeze(0), size=(224, 224),
                               mode='bilinear', align_corners=False)[0, 0].numpy()
    predicted_bbox = heatmap_to_bbox(heatmap_up, threshold=0.4)
    gt_bbox_orig = bbox_dict.get(filename, None)

    # Обработка изображения
    img_vis = np.array(original_image.resize((224, 224))).copy()
    overlay = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min() + 1e-8)
    overlay = cv2.applyColorMap(np.uint8(255 * overlay), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img_vis, 0.6, overlay, 0.4, 0)

    if predicted_bbox:
        cv2.rectangle(overlayed, tuple(predicted_bbox[:2]), tuple(predicted_bbox[2:]), (0, 255, 0), 2)

    if gt_bbox_orig:
        # Масштабируем оригинальные координаты в 224x224
        x1, y1, x2, y2 = gt_bbox_orig
        gt_bbox_scaled = [
            int(x1 * (224 / w_orig)),
            int(y1 * (224 / h_orig)),
            int(x2 * (224 / w_orig)),
            int(y2 * (224 / h_orig))
        ]
        cv2.rectangle(overlayed, tuple(gt_bbox_scaled[:2]), tuple(gt_bbox_scaled[2:]), (255, 0, 0), 2)

        if predicted_bbox:
            iou = compute_iou(predicted_bbox, gt_bbox_scaled)
            giou = compute_giou(predicted_bbox, gt_bbox_scaled)
            hausdorff = compute_hausdorff_distance(predicted_bbox, gt_bbox_scaled)
            avg_boundary_dist = compute_average_boundary_distance(
                predicted_bbox, gt_bbox_scaled, image_width=224, image_height=224
            )

            ious.append(iou)
            gious.append(giou)
            hausdorff_distances.append(hausdorff)
            avg_boundary_distances.append(avg_boundary_dist)

            cv2.putText(overlayed, f"IoU: {iou:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlayed, f"Hausdorff: {hausdorff:.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlayed, f"Avg Boundary: {avg_boundary_dist:.2f}%", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out_path = os.path.join("heatmap_comparisons", f"compare_{filename}")
    cv2.imwrite(out_path, overlayed[:, :, ::-1])
    print(f"Saved: {out_path}")

if ious:
    print(f"Mean IoU over {len(ious)} samples: {np.mean(ious):.4f}")
if gious:
    print(f"Mean GIoU over {len(gious)} samples: {np.mean(gious):.4f}")
if hausdorff_distances:
    print(f"Mean Hausdorff Distance over {len(hausdorff_distances)} samples: {np.mean(hausdorff_distances):.4f}")
if avg_boundary_distances:
    print(f"Mean Average Boundary Distance over {len(avg_boundary_distances)} samples: {np.mean(avg_boundary_distances):.4f}%")