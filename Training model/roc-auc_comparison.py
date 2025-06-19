import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
from itertools import cycle
from sklearn.model_selection import train_test_split
from B_training_teacher_classifier import DeiTClassifierWSOL
from A_making_data_loader_for_classification import classification_val_loader


# DicomStyleImagePreprocessor class (from your file)
class DicomStyleImagePreprocessor:
    """Custom transform that applies DICOM-style preprocessing to regular images"""

    def __init__(self, target_size=(224, 224), apply_clahe=True):
        self.target_size = target_size
        self.apply_clahe = apply_clahe

    def apply_clahe_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement"""
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_uint8)

    def __call__(self, pil_image):
        # Convert PIL to numpy array
        if pil_image.mode == 'RGB':
            # Convert RGB to grayscale for medical image style processing
            img = np.array(pil_image.convert('L')).astype(np.float32)
        else:
            img = np.array(pil_image).astype(np.float32)

        # Intensity normalization to 0-255 range (similar to DICOM processing)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min + 1e-8) * 255.0
        else:
            img = np.zeros_like(img)

        # Apply CLAHE for contrast enhancement
        if self.apply_clahe:
            img = self.apply_clahe_enhancement(img)

        # Resize image
        img = cv2.resize(img, self.target_size)

        # Convert grayscale to RGB by stacking channels (like in DICOM handler)
        img_rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)

        # Convert back to PIL Image for torchvision transforms
        return Image.fromarray(img_rgb)


class EvaluationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {
            'Atelectasis': 0,
            'No_pathologies': 1,
            'Other_pathologies': 2
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Collect all image paths and labels (избегаем дублирования)
        added_files = set()  # Отслеживаем уже добавленные файлы

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                # Check for various image formats
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                    for img_path in class_dir.glob(ext):
                        file_key = str(img_path.resolve())  # Используем абсолютный путь как ключ
                        if file_key not in added_files:
                            self.samples.append((str(img_path), class_idx))
                            added_files.add(file_key)

        print(f"Найдено изображений:")
        for class_name, class_idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == class_idx)
            print(f"  {class_name}: {count}")

            # Дополнительная диагностика - покажем какие расширения найдены
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                extensions_found = {}
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                    files = list(class_dir.glob(ext))
                    if files:
                        extensions_found[ext] = len(files)
                if extensions_found:
                    print(f"    Расширения: {extensions_found}")

        print(f"Всего: {len(self.samples)}")

        # Проверим на дубликаты по имени файла (без пути)
        file_names = [Path(path).name for path, _ in self.samples]
        unique_names = set(file_names)
        if len(file_names) != len(unique_names):
            print(f"ВНИМАНИЕ: Обнаружены дубликаты! Уникальных имен файлов: {len(unique_names)}")
            # Найдем дубликаты
            from collections import Counter
            name_counts = Counter(file_names)
            duplicates = {name: count for name, count in name_counts.items() if count > 1}
            if duplicates:
                print(f"Дублированные файлы: {duplicates}")
        else:
            print("Дубликатов не найдено.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            print(f"Ошибка при загрузке {img_path}: {e}")
            # Return a dummy image in case of error
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label, img_path


def create_stratified_split(dataset, test_size=0.2, random_state=42):
    """Создает стратифицированное разделение датасета"""
    # Получаем все индексы и соответствующие метки
    indices = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in indices]

    # Стратифицированное разделение
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Создаем subset'ы
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Показываем статистику разделения
    print(f"\nСтратифицированное разделение (test_size={test_size}):")
    print(f"Общее количество образцов: {len(dataset)}")
    print(f"Обучающая выборка: {len(train_dataset)}")
    print(f"Тестовая выборка: {len(test_dataset)}")

    # Проверяем баланс классов
    class_names = ['Atelectasis', 'No_pathologies', 'Other_pathologies']

    # Исходное распределение
    original_dist = {}
    for class_idx in range(len(class_names)):
        count = sum(1 for label in labels if label == class_idx)
        original_dist[class_idx] = count

    # Распределение в тестовой выборке
    test_labels = [labels[i] for i in test_indices]
    test_dist = {}
    for class_idx in range(len(class_names)):
        count = sum(1 for label in test_labels if label == class_idx)
        test_dist[class_idx] = count

    print(f"\nРаспределение по классам:")
    print(f"{'Класс':<20} {'Исходно':<10} {'Тест':<10} {'% в тесте':<12}")
    print("-" * 55)
    for class_idx, class_name in enumerate(class_names):
        orig_count = original_dist[class_idx]
        test_count = test_dist[class_idx]
        test_percentage = (test_count / orig_count * 100) if orig_count > 0 else 0
        print(f"{class_name:<20} {orig_count:<10} {test_count:<10} {test_percentage:<12.1f}%")

    return train_dataset, test_dataset


def evaluate_model(model, dataloader, device, num_classes=3):
    """Evaluate model and return predictions and true labels"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels), all_paths


def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """Plot ROC curves for multiclass classification"""
    n_classes = len(class_names)

    # Binarize the output
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_prob[:, i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = roc_auc_score(y_true_bin, y_prob, average="micro")

    # Plot all curves
    plt.figure(figsize=(12, 8))

    # Plot ROC curve for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve {class_names[i]} (AUC = {roc_auc[i]:.3f})')

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-class Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return roc_auc


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return cm


def calculate_class_metrics(y_true, y_prob, class_names):
    """Calculate detailed metrics for each class"""
    n_classes = len(class_names)
    from sklearn.preprocessing import label_binarize

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Calculate AUC for each class
    class_aucs = {}
    for i in range(n_classes):
        try:
            auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            class_aucs[class_names[i]] = auc
        except ValueError as e:
            print(f"Не удалось вычислить AUC для класса {class_names[i]}: {e}")
            class_aucs[class_names[i]] = 0.0

    return class_aucs


def main():
    # Настройки
    checkpoint_path = r"C:\Users\CYBER ARTEL\PycharmProjects\atelectasis_classification_and_detection\best_deit_scm_model_2.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Используется устройство: {device}")

    # Использование тестовой выборки из импортированного файла
    test_dataloader = classification_val_loader

    print(f"\nИспользуется тестовая выборка из classification_val_loader")
    print(f"Количество батчей: {len(test_dataloader)}")

    # Оценим примерное количество образцов
    batch_size = test_dataloader.batch_size if hasattr(test_dataloader, 'batch_size') else 'неизвестно'
    print(f"Размер батча: {batch_size}")

    # Загрузка модели
    model = DeiTClassifierWSOL(num_classes=3, pretrained=False, scm_blocks=4)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Оценка модели на импортированной тестовой выборке
    print("\nЗапуск оценки модели на тестовой выборке...")
    predictions, probabilities, true_labels, paths = evaluate_model(model, test_dataloader, device)

    # Названия классов
    class_names = ['Atelectasis', 'No_pathologies', 'Other_pathologies']

    # Построение ROC кривых
    print("Построение ROC кривых...")
    roc_aucs = plot_roc_curves(true_labels, probabilities, class_names, 'test_roc_curves.png')

    # Построение матрицы ошибок
    print("Построение матрицы ошибок...")
    cm = plot_confusion_matrix(true_labels, predictions, class_names, 'test_confusion_matrix.png')

    # Вычисление метрик для каждого класса
    class_aucs = calculate_class_metrics(true_labels, probabilities, class_names)

    # Печать результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ROC AUC АНАЛИЗА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)

    print(f"\nРазмер тестовой выборки: {len(true_labels)} образцов")
    print(f"Используется готовый classification_val_loader")

    print("\nAUC для каждого класса:")
    for class_name, auc in class_aucs.items():
        print(f"  {class_name}: {auc:.4f}")

    print(f"\nMicro-average AUC: {roc_aucs['micro']:.4f}")

    # Детальный отчет классификации
    print("\nДетальный отчет классификации (импортированная тестовая выборка):")
    print(classification_report(true_labels, predictions, target_names=class_names))

    # Сохранение результатов в CSV
    results_df = pd.DataFrame({
        'Image_Path': paths,
        'True_Label': [class_names[label] for label in true_labels],
        'Predicted_Label': [class_names[pred] for pred in predictions],
        'Atelectasis_Prob': probabilities[:, 0],
        'No_pathologies_Prob': probabilities[:, 1],
        'Other_pathologies_Prob': probabilities[:, 2],
        'Correct_Prediction': true_labels == predictions
    })

    results_df.to_csv('test_results.csv', index=False)
    print("\nДетальные результаты тестирования сохранены в 'test_results.csv'")

    # Анализ ошибок на импортированной тестовой выборке
    print("\nАнализ ошибок на импортированной тестовой выборке:")
    incorrect_predictions = results_df[results_df['Correct_Prediction'] == False]
    correct_predictions = results_df[results_df['Correct_Prediction'] == True]

    print(f"Правильных предсказаний: {len(correct_predictions)}")
    print(f"Неправильных предсказаний: {len(incorrect_predictions)}")
    print(f"Точность на импортированной тестовой выборке: {len(correct_predictions) / len(results_df) * 100:.2f}%")

    if len(incorrect_predictions) > 0:
        print("\nРаспределение ошибок по классам:")
        error_analysis = incorrect_predictions.groupby(['True_Label', 'Predicted_Label']).size()
        print(error_analysis)

        # Анализ ошибок по каждому классу
        print("\nДетальный анализ ошибок:")
        for true_class in class_names:
            class_errors = incorrect_predictions[incorrect_predictions['True_Label'] == true_class]
            if len(class_errors) > 0:
                total_class_samples = len(results_df[results_df['True_Label'] == true_class])
                error_rate = len(class_errors) / total_class_samples * 100
                print(f"\n{true_class}:")
                print(f"  Всего образцов: {total_class_samples}")
                print(f"  Ошибок: {len(class_errors)} ({error_rate:.1f}%)")

                # Куда чаще всего ошибочно классифицируются
                misclassified_as = class_errors['Predicted_Label'].value_counts()
                print(f"  Чаще всего ошибочно классифицируется как:")
                for pred_class, count in misclassified_as.items():
                    print(f"    {pred_class}: {count} раз")

    # Дополнительная статистика по доверительности предсказаний
    print("\nСтатистика доверительности предсказаний:")
    max_probs = np.max(probabilities, axis=1)
    print(f"Средняя максимальная вероятность: {np.mean(max_probs):.3f}")
    print(f"Медианная максимальная вероятность: {np.median(max_probs):.3f}")
    print(f"Минимальная максимальная вероятность: {np.min(max_probs):.3f}")

    # Анализ неуверенных предсказаний (низкая максимальная вероятность)
    uncertain_threshold = 0.5
    uncertain_predictions = results_df[np.max(probabilities, axis=1) < uncertain_threshold]
    if len(uncertain_predictions) > 0:
        print(f"\nНеуверенные предсказания (макс. вероятность < {uncertain_threshold}):")
        print(f"Количество: {len(uncertain_predictions)}")
        print(
            f"Точность среди неуверенных: {len(uncertain_predictions[uncertain_predictions['Correct_Prediction'] == True]) / len(uncertain_predictions) * 100:.1f}%")


if __name__ == "__main__":
    main()