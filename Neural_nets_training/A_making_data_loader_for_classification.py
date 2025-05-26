from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Трансформации для картинок
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Используем ImageFolder: автоматом распознает подпапки как классы
dataset = datasets.ImageFolder(root='C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\new_classes', transform=transform)

# Разделение на train и val
from sklearn.model_selection import train_test_split

indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=[dataset.targets[i] for i in indices])

from torch.utils.data import Subset

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

classification_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
classification_val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Выводим информацию о классах
print("Классы:", dataset.classes)  # ['class1', 'class2', ...]
print("Соответствие класс-индекс:", dataset.class_to_idx)  # {'class1': 0, 'class2': 1, ...}

if __name__ == '__main__':
    # Проверка размерности одного батча
    batch = next(iter(classification_train_loader))
    images, labels = batch
    print(f"Размерность изображений: {images.shape}")  # Должно быть [batch_size, channels, height, width]
    print(f"Размерность меток: {labels.shape}")  # Должно быть [batch_size]

