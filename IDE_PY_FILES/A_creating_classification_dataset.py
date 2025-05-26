import os
import shutil
import pandas as pd

# Путь к CSV
csv_path = 'C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_files\\3\\Data_Entry_2017.csv'
# Папки с картинками
image_folders = [f'C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_files\\3\\{folder}\\images' for folder in
                 ['images_001', 'images_002', 'images_003', 'images_004',
                  'images_005', 'images_006', 'images_007', 'images_008',
                  'images_009', 'images_010', 'images_011', 'images_012']]

# Путь куда будем сохранять новый датасет
output_root = 'C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\new_classes'
os.makedirs(os.path.join(output_root, 'Atelectasis'), exist_ok=True)
os.makedirs(os.path.join(output_root, 'Other_pathologies'), exist_ok=True)
os.makedirs(os.path.join(output_root, 'No_pathologies'), exist_ok=True)

# Загрузка CSV
df = pd.read_csv(csv_path)


# Функция поиска полного пути картинки
def find_image_path(image_name):
    for folder in image_folders:
        path = os.path.join(folder, image_name)
        if os.path.exists(path):
            return path
    return None


# Списки для сохранения путей
records = []

# Проход по всем строкам
for idx, row in df.iterrows():
    image_name = row['Image Index']
    labels = row['Finding Labels'].split('|')  # Разбиваем по "|"

    # Определяем класс
    if labels == ['No Finding']:
        class_label = 'No_pathologies'
    elif labels == ['Atelectasis']:
        class_label = 'Atelectasis'
    else:
        class_label = 'Other_pathologies'

    # Находим реальный путь к картинке
    src_path = find_image_path(image_name)
    if src_path is not None:
        dst_path = os.path.join(output_root, class_label, image_name)
        shutil.copy(src_path, dst_path)
        records.append((dst_path, class_label))

# Сохраняем пути в CSV
df_paths = pd.DataFrame(records, columns=['path', 'class'])
csv_save_path = 'dataset_paths.csv'
df_paths.to_csv(csv_save_path, index=False)

print(f'Готово! Отобрано {len(records)} картинок.')
print(f'Пути сохранены в {csv_save_path}')
