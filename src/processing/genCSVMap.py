import os
import pandas as pd
from tqdm import tqdm
from glob import glob

BASE_PATH = 'data'  

CSV_PATH = os.path.join(BASE_PATH, 'Data_Entry_2017.csv')
TRAIN_VAL_PATH = os.path.join(BASE_PATH, 'train_val_list.txt')
TEST_PATH = os.path.join(BASE_PATH, 'test_list.txt')


labels_train_val = pd.read_csv(TRAIN_VAL_PATH, names=['Image_Index'])
labels_test = pd.read_csv(TEST_PATH, names=['Image_Index'])

# Etiquetas conocidas del PDF de FAQ
disease_labels = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

cols = [
    'Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
    'Patient_Age', 'Patient_Gender', 'View_Position',
    'Original_Image_Width', 'Original_Image_Height',
    'Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Extra'
]
labels_df = pd.read_csv(CSV_PATH, names=cols, header=0)

labels_df['Finding_Labels'] = labels_df['Finding_Labels'].fillna('No Finding')
labels_df['Patient_Gender'] = labels_df['Patient_Gender'].fillna('U')


for disease in tqdm(disease_labels, desc="One-hot encoding diseases"):
    labels_df[disease] = labels_df['Finding_Labels'].map(
        lambda result: 1 if disease in result else 0
    )


gender_onehot = pd.get_dummies(labels_df['Patient_Gender'], prefix='Gender')
labels_df = pd.concat([labels_df, gender_onehot], axis=1)

print("Buscando imágenes en subcarpetas...")
num_glob = glob(os.path.join(BASE_PATH, "**", "images", "*.png"), recursive=True)
img_path = {os.path.basename(x): x for x in num_glob}
print(f"{len(img_path)} imágenes encontradas en estructura de carpetas.")

# Asociar ruta a cada imagen
labels_df['Path'] = labels_df['Image_Index'].map(img_path.get)

#Guardamos
labels_df.to_csv('data/IdxDataset.csv')

train_val_df = labels_df.merge(labels_train_val, on='Image_Index', how='inner')
test_df = labels_df.merge(labels_test, on='Image_Index', how='inner')

print(f"Train/Val: {len(train_val_df)} imágenes")
print(f"Test: {len(test_df)} imágenes")

print("\nEjemplo:")
print(train_val_df[['Image_Index', 'Finding_Labels', 'Patient_Gender', 'Gender_F', 'Gender_M', 'Path']].head())
