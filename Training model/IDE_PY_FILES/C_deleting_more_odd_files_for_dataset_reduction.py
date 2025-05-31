import os
import pandas as pd

# --- Пути ---
DATASET_DIR = 'C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\new_classes'  # ← замените на свой путь к папке new_classes
CSV_PATH = 'C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\csv_files\\dataset_paths_representative.csv'

# --- 1. Загружаем список репрезентативных изображений ---
df = pd.read_csv(CSV_PATH)
keep_files = set(df['Image Index'].unique())
print(f"Будут сохранены следующие файлы: {len(keep_files)} шт.")

# --- 2. Проходим по всем подпапкам и удаляем лишние файлы ---
deleted_count = 0

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):  # фильтруем по расширению
            if file not in keep_files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Удалён: {file}")
                except Exception as e:
                    print(f"Ошибка при удалении {file}: {e}")

print(f"\n✅ Удалено файлов: {deleted_count}")