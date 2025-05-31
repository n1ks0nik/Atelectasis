import os
import shutil
import pandas as pd

# --- Пути ---
# Список исходных папок с изображениями
source_folders = [f'C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_files\\3\\{folder}\\images' for folder in
                 ['images_001', 'images_002', 'images_003', 'images_004',
                  'images_005', 'images_006', 'images_007', 'images_008',
                  'images_009', 'images_010', 'images_011', 'images_012']]

# Целевая папка для детекции
destination_folder = r'C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\for_detection'

# Путь к CSV с bounding box метками
bbox_csv_path = r'C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_files\3\Atelectasis_BBox_List_2017.csv'

# --- Подготовка ---
# Создаем целевую папку, если её нет
os.makedirs(destination_folder, exist_ok=True)

# Загружаем список изображений с ателектазом из CSV
atelectasis_df = pd.read_csv(bbox_csv_path)
atelectasis_files = atelectasis_df['Image Index'].unique().tolist()

print(f"Найдено {len(atelectasis_files)} файлов для копирования.")

# --- Копирование файлов ---
copied_count = 0
not_found_files = []

for file_name in atelectasis_files:
    found = False
    for src_folder in source_folders:
        source_path = os.path.join(src_folder, file_name)
        if os.path.exists(source_path):
            dest_path = os.path.join(destination_folder, file_name)
            try:
                shutil.copy2(source_path, dest_path)
                print(f"✅ Скопировано: {file_name}")
                copied_count += 1
                found = True
                break
            except Exception as e:
                print(f"❌ Ошибка при копировании {file_name}: {e}")
                not_found_files.append(file_name)
                found = True
                break
    if not found:
        print(f"⚠️ Файл не найден: {file_name}")
        not_found_files.append(file_name)

# --- Отчёт ---
print("\n--- Отчёт ---")
print(f"Всего файлов в списке: {len(atelectasis_files)}")
print(f"Успешно скопировано: {copied_count}")
print(f"Не найдено/ошибки: {len(not_found_files)}")

# Сохраняем список ненайденных файлов (для отладки)
missing_log_path = os.path.join(destination_folder, "missing_files.txt")
with open(missing_log_path, 'w') as f:
    for fname in not_found_files:
        f.write(fname + '\n')

print(f"\nСписок ненайденных файлов сохранён в: {missing_log_path}")