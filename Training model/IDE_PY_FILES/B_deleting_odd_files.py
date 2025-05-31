import pandas as pd
import os

labels_df = pd.read_csv('C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_files\\3\\Data_Entry_2017.csv')
folder = 'C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\new_classes\\Other_pathologies'
deleted_count = 0

for idx, row in labels_df.iterrows():
    labels = row['Finding Labels'].lower().split('|')
    fname = row['Image Index']
    filepath = os.path.join(folder, fname)

    if 'atelectasis' in labels and len(labels) > 1 and os.path.isfile(filepath):
        os.remove(filepath)
        deleted_count += 1
        print(f"Удалено: {fname}")

print(f"\nИтого удалено файлов: {deleted_count}")