import os
from pathlib import Path
from collections import defaultdict
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
    """
    Procesa el CSV de labels y mapea a las imágenes en disco sin eliminar imágenes.
    Devuelve:
      - processed_df: DataFrame completo (todas las filas del CSV, con Path si se encontró)
      - train_val_df: subconjunto marcado como train/val (no se eliminan filas del processed_df)
      - test_df: subconjunto marcado como test
    """
    if disease_labels is None:
        disease_labels = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
        ]

    base_path = str(base_path)
    csv_path = os.path.join(base_path, csv_filename)
    train_val_path = os.path.join(base_path, train_val_list)
    test_path = os.path.join(base_path, test_list)

    # Cargar listas (asumiendo un nombre por linea)
    labels_train_val = pd.read_csv(train_val_path, names=['Image_Index'], header=None)
    labels_test = pd.read_csv(test_path, names=['Image_Index'], header=None)
    print(f"labels train size: {labels_train_val.shape[0]}  |  labels test size: {labels_test.shape[0]}")

    # Columnas esperadas en Data_Entry_2017.csv
    cols = [
        'Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
        'Patient_Age', 'Patient_Gender', 'View_Position',
        'Original_Image_Width', 'Original_Image_Height',
        'Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Extra'
    ]
    labels_df = pd.read_csv(csv_path, names=cols, header=0)

    # Limpieza básica de valores faltantes
    labels_df['Finding_Labels'] = labels_df['Finding_Labels'].fillna('No Finding')
    labels_df['Patient_Gender'] = labels_df['Patient_Gender'].fillna('U')
    labels_df['View_Position'] = labels_df['View_Position'].fillna('UNK')
    labels_df['Patient_Age'] = labels_df['Patient_Age'].fillna(labels_df['Patient_Age'].mean())

    def clean_labels(x):
        parts = [p.strip() for p in str(x).split('|') if p.strip() != 'No Finding']
        return '|'.join(parts)

    labels_df['Finding_Labels'] = labels_df['Finding_Labels'].apply(clean_labels)
    # NOTA: no filtramos filas con Finding_Labels vacías para no eliminar imágenes

    # One-hot para enfermedades
    for disease in tqdm(disease_labels, desc="One-hot encoding diseases"):
        labels_df[disease] = labels_df['Finding_Labels'].map(lambda result: 1 if disease in result else 0)

    # Mapas rápidos
    gender_map = {'M': 1, 'F': 0, 'U': -1}
    labels_df['Gender_Code'] = labels_df['Patient_Gender'].map(gender_map).fillna(-1)
    view_map = {'PA': 1, 'AP': 0, 'UNK': -1}
    labels_df['View_Code'] = labels_df['View_Position'].map(view_map).fillna(-1)

    # Buscar imágenes recursivamente: varias extensiones (case-insensitive)
    print("Buscando imágenes en subcarpetas (extensiones png/jpg/jpeg) ...")
    allowed_suffixes = {'.png', '.jpg', '.jpeg'}
    img_paths_all = []
    for p in Path(base_path).rglob('*'):
        if p.is_file() and p.suffix.lower() in allowed_suffixes:
            img_paths_all.append(str(p))

    print(f"{len(img_paths_all)} archivos de imágenes encontrados en la estructura de carpetas.")

    # Construir mapas para búsqueda robusta:
    # - map_full: clave -> ruta completa (clave: ruta relativa desde base, en lower)
    # - map_basename: basename lower -> [rutas]
    map_full = dict()
    map_basename = defaultdict(list)

    for fullp in img_paths_all:
        p = Path(fullp)
        # ruta relativa relativa al base_path
        try:
            rel = str(p.relative_to(base_path))
        except Exception:
            rel = str(p)
        key_full = rel.replace('\\', '/').lower()
        map_full[key_full] = fullp

        bname = p.name.lower()
        map_basename[bname].append(fullp)

    # Función para resolver Image_Index a Path
    def resolve_path(image_index: str):
        if pd.isna(image_index):
            return None, []
        img_index = str(image_index).strip()
        # 1) probar coincidencia exacta por ruta relativa (case-insensitive)
        k = img_index.replace('\\', '/').lower()
        if k in map_full:
            return map_full[k], [map_full[k]]

        # 2) probar basename directo (case-insensitive)
        b = Path(img_index).name.lower()
        if b in map_basename:
            lst = map_basename[b]
            # devolver la primera como principal, pero devolver la lista completa
            return lst[0], lst

        # 3) probar añadiendo extensiones habituales si la entrada no tiene extensión
        if '.' not in b:
            for ext in allowed_suffixes:
                trial = b + ext
                if trial in map_basename:
                    lst = map_basename[trial]
                    return lst[0], lst

        # 4) no encontrado
        return None, []

    # Aplicar resolución a todo el dataframe
    resolved_primary = []
    resolved_all = []
    unresolved = 0
    multiple_candidates = 0

    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Resolviendo paths"):
        img_index = row['Image_Index']
        primary, all_candidates = resolve_path(img_index)
        resolved_primary.append(primary)
        resolved_all.append(all_candidates)
        if primary is None:
            unresolved += 1
        if len(all_candidates) > 1:
            multiple_candidates += 1

    labels_df['Path'] = resolved_primary
    labels_df['All_Paths'] = resolved_all
    labels_df['Missing_Path'] = labels_df['Path'].isna()
    labels_df['Num_Candidates'] = labels_df['All_Paths'].map(len)

    print(f"Resolución: {len(labels_df) - unresolved} encontrados, {unresolved} no encontrados.")
    print(f"{multiple_candidates} entradas con múltiples candidatos (usa 'All_Paths' para ver opciones).")

    # Columnas finales a mantener (mantenemos filas aunque Path sea NaN)
    columns_to_keep = ['Image_Index', 'Path', 'All_Paths', 'Num_Candidates', 'Missing_Path',
                       'Patient_Age', 'Gender_Code', 'View_Code'] + disease_labels
    processed_df = labels_df[columns_to_keep].copy()

    # Guardar CSV con todo (no se eliminan filas)
    output_path = os.path.join(base_path, output_filename)
    processed_df.to_csv(output_path, index=False)
    print(f"Dataset procesado guardado en: {output_path}")

    # Marcar si están en train/val y test (sin eliminar)
    train_set = set(labels_train_val['Image_Index'].astype(str).str.strip().tolist())
    test_set = set(labels_test['Image_Index'].astype(str).str.strip().tolist())

    # Para comparación, comparamos usando Image_Index tal cual y también su basename
    def in_set(img_index, s):
        if pd.isna(img_index):
            return False
        img_index = str(img_index).strip()
        if img_index in s:
            return True
        if Path(img_index).name in s:
            return True
        return False

    processed_df['in_train_val'] = processed_df['Image_Index'].map(lambda x: in_set(x, train_set))
    processed_df['in_test'] = processed_df['Image_Index'].map(lambda x: in_set(x, test_set))

    # Construir dataframes de train/val y test como subconjuntos (no eliminan el processed_df)
    train_val_df = processed_df[processed_df['in_train_val']].copy()
    test_df = processed_df[processed_df['in_test']].copy()

    print(f"Train/Val (marcados): {len(train_val_df)} imágenes")
    print(f"Test (marcados): {len(test_df)} imágenes")
    print(f"Total filas en processed_df (sin eliminar): {len(processed_df)}")

    # Ejemplo de salida
    print("\nEjemplo (primeras 5 filas de processed_df):")
    print(processed_df.head())

    return processed_df, train_val_df, test_df
