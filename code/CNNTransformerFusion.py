import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

class CNNTransformerFusion(nn.Module):
    def __init__(self, num_classes=14, num_meta=3, d_model=256, nhead=4, num_layers=2):
        super().__init__()

        base = models.resnet121(weights='IMAGENET1K_V1')
        self.cnn = nn.Sequential(*list(base.children())[:-1])  # sin FC
        cnn_feat_dim = base.fc.in_features  # 1024 en ResNet-121
        self.img_proj = nn.Linear(cnn_feat_dim, d_model)
        self.meta_proj = nn.Linear(num_meta, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, images, meta):
        x = self.cnn(images)
        x = torch.flatten(x, 1)  # (batch, 1024)
        img_embed = self.img_proj(x).unsqueeze(1)  # (batch, 1, d_model)

        meta_embed = self.meta_proj(meta).unsqueeze(1)  # (batch, 1, d_model)

        seq = torch.cat([meta_embed, img_embed], dim=1)  # (batch, 2, d_model)

        out = self.transformer(seq)

        fused = out.mean(dim=1)
        out = self.fc(fused)

        return out
