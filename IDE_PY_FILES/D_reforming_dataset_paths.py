import os
import pandas as pd

# Пути
base_dir = 'C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\new_classes'
data_entry_path = 'C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_files\\3\\Data_Entry_2017.csv'
output_path = 'dataset_paths.csv'

# Словарь: {папка -> класс}
class_mapping = {
    'Atelectasis': 'Atelectasis',
    'No_pathologies': 'No pathologies',
    'Other_pathologies': 'Other pathologies'
}

# Шаг 1: Создаем DataFrame из папок
data = []

for folder_name, class_label in class_mapping.items():
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Папка не найдена: {folder_path}")
        continue
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # фильтр по расширению
            data.append({'Image Index': file_name, 'Class': class_label})

df_dataset = pd.DataFrame(data, columns=['Image Index', 'Class'])

# Шаг 2: Загружаем Data_Entry_2017.csv
df_data_entry = pd.read_csv(data_entry_path)

# Оставляем только нужные столбцы
df_metadata = df_data_entry[['Image Index', 'Patient Age', 'Patient Gender']]

# Шаг 3: Объединяем по 'Image Index'
df_final = pd.merge(df_dataset, df_metadata, on='Image Index', how='left')

# Шаг 4: Сохраняем результат
df_final.to_csv(output_path, index=False)

# Вывод информации о созданном датасете
print(f"\nСоздан новый dataset_paths.csv с {len(df_final)} записями:")
print(df_final['Class'].value_counts())
print("\nПример данных:")
print(df_final.head())