import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

class datasetMgr(Dataset):
    
    def __init__(self, dataframe, disease_columns, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.disease_columns = disease_columns
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'Path']

        # Transformamos en RGB porque los modelos esperan imagenes con los canales RGB y tensores
        # hechos a partir de RGB images.        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        labels = self.df.loc[idx, self.disease_columns].values.astype(np.float32)
        
        gender = self.df.loc[idx, 'Gender_Code']
        view = self.df.loc[idx, 'View_Code']
        age = self.df.loc[idx, 'Patient_Age'] / 100.0 

        meta = torch.tensor([gender, view, age], dtype=torch.float32)


        if self.transform:
            image = self.transform(image)

        
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, meta, labels