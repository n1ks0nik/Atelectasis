import pandas as pd
import numpy as np

# --- 1. Загрузка данных ---
df = pd.read_csv("C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\csv_files\\dataset_paths.csv")

# --- 2. Разделение на Atelectasis и не Atelectasis ---
df_atel = df[df['Class'] == 'Atelectasis']
df_others = df[df['Class'] != 'Atelectasis']

# --- 3. Точная стратификация по возрасту через квантили ---
df_others['AgeQuantile'] = pd.qcut(df_others['Patient Age'], q=20, duplicates='drop')

# --- 4. Целевой размер новой выборки ---
target_total_size = int(len(df) * 0.4)
print(f"\nЦелевой размер новой выборки (кроме Atelectasis): {target_total_size}")

# --- 5. Пропорции классов (без Atelectasis) ---
class_proportions = df_others['Class'].value_counts(normalize=True)
print("\nДоли классов в оригинальном датасете (кроме Atelectasis):")
print(class_proportions)

# --- 6. Функция стратифицированного сэмплирования ---
def stratified_sampler(group, class_proportions, target_total_size):
    class_name, gender, _ = group.name
    desired_class_size = int(class_proportions[class_name] * target_total_size)
    desired_group_size = int(desired_class_size * (len(group) / len(df_others[df_others['Class'] == class_name])))

    max_upsample_factor = 2
    actual_size = min(desired_group_size, len(group) * max_upsample_factor)

    if actual_size < 1:
        return pd.DataFrame()

    return group.sample(n=actual_size, replace=(len(group) < actual_size), random_state=42)


# --- 7. Группировка и выборка ---
try:
    grouped = df_others.groupby(['Class', 'Patient Gender', 'AgeQuantile'], group_keys=False)
    stratified_sample_others = grouped.apply(lambda g: stratified_sampler(g, class_proportions, target_total_size))
    stratified_sample_others = stratified_sample_others.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- 8. Объединяем с Atelectasis ---
    final_stratified_df = pd.concat([df_atel, stratified_sample_others], ignore_index=True)
    final_stratified_df = final_stratified_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nРазмер новой выборки: {len(final_stratified_df)}")

    # --- 9. Сохранение нового датасета ---
    final_stratified_df.to_csv("dataset_paths_representative.csv", index=False)
    print("Новая выборка сохранена в 'dataset_paths_representative.csv'")

except Exception as e:
    print("Ошибка при создании выборки:", e)