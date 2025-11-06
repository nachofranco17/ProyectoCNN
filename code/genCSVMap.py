import os
import pandas as pd
from tqdm import tqdm
from glob import glob

def prepare_labels_dataset(
    base_path: str,
    csv_filename: str = 'Data_Entry_2017.csv',
    train_val_list: str = 'train_val_list.txt',
    test_list: str = 'test_list.txt',
    output_filename: str = 'IdxDataset.csv',
    disease_labels: list = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if disease_labels is None:
        disease_labels = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
        ]

    csv_path = os.path.join(base_path, csv_filename)
    train_val_path = os.path.join(base_path, train_val_list)
    test_path = os.path.join(base_path, test_list)

    labels_train_val = pd.read_csv(train_val_path, names=['Image_Index'])
    labels_test = pd.read_csv(test_path, names=['Image_Index'])

    cols = [
        'Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
        'Patient_Age', 'Patient_Gender', 'View_Position',
        'Original_Image_Width', 'Original_Image_Height',
        'Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Extra'
    ]
    labels_df = pd.read_csv(csv_path, names=cols, header=0)

    labels_df['Finding_Labels'] = labels_df['Finding_Labels'].fillna('No Finding')
    labels_df['Patient_Gender'] = labels_df['Patient_Gender'].fillna('U')
    labels_df['View_Position'] = labels_df['View_Position'].fillna('UNK')

    def clean_labels(x):
        parts = [p.strip() for p in x.split('|') if p.strip() != 'No Finding']
        return '|'.join(parts)

    labels_df['Finding_Labels'] = labels_df['Finding_Labels'].apply(clean_labels)
    labels_df = labels_df[labels_df['Finding_Labels'] != ''].reset_index(drop=True)

    for disease in tqdm(disease_labels, desc="One-hot encoding diseases"):
        labels_df[disease] = labels_df['Finding_Labels'].map(lambda result: 1 if disease in result else 0)

    gender_map = {'M': 1, 'F': 0, 'U': -1}
    labels_df['Gender_Code'] = labels_df['Patient_Gender'].map(gender_map).fillna(-1)

    view_map = {'PA': 1, 'AP': 0, 'UNK': -1}
    labels_df['View_Code'] = labels_df['View_Position'].map(view_map).fillna(-1)

    print("Buscando imágenes en subcarpetas...")
    num_glob = glob(os.path.join(base_path, "**", "images", "*.png"), recursive=True)
    img_path = {os.path.basename(x): x for x in num_glob}
    print(f"{len(img_path)} imágenes encontradas en estructura de carpetas.")

    labels_df['Path'] = labels_df['Image_Index'].map(img_path.get)

    columns_to_keep = ['Image_Index', 'Path', 'Gender_Code', 'View_Code'] + disease_labels
    processed_df = labels_df[columns_to_keep].dropna(subset=['Path'])

    output_path = os.path.join(base_path, output_filename)
    processed_df.to_csv(output_path, index=False)
    print(f"Dataset procesado guardado en: {output_path}")

    train_val_df = processed_df.merge(labels_train_val, on='Image_Index', how='inner')
    test_df = processed_df.merge(labels_test, on='Image_Index', how='inner')

    print(f"Train/Val: {len(train_val_df)} imágenes")
    print(f"Test: {len(test_df)} imágenes")

    print("\nEjemplo (primeras 5 filas):")
    print(processed_df.head())

    return processed_df, train_val_df, test_df
