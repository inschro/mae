import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from timm import create_model
import numpy as np
import os
from sklearn.covariance import LedoitWolf
import cv2
import matplotlib.pyplot as plt
import models_mae
from PIL import Image
# A script that chat gpt created after a little bit of guidance and further prompting

# Parameters
BATCH_SIZE = 16
IMAGE_SIZE = 224
MODEL_NAME = 'mae_vit_base_patch16'
DATASET_PATH = r'/home/darius/Dokumente/Research/mae/data/mvtec-ad'
WEIGHTS_PATH = r'/home/darius/Dokumente/Research/mae/jobs/20240724105007/outputs/checkpoint-70.pth'
CATEGORY = 'bottle'

MASKING_ARGS = {
    'masking_ratio': 0.0,
}

def prepare_model(chkpt_dir, arch=MODEL_NAME):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

# Feature Extractor using ViT
class FeatureExtractor(nn.Module):
    def __init__(self,chkpt_dir, model_name,):
        super(FeatureExtractor, self).__init__()
        self.model = prepare_model(chkpt_dir, model_name)

    def forward(self, x):
        x = self.model.forward_encoder(x,masking_type="random_masking", **MASKING_ARGS)
        return x


# Load Dataset
class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Load Dataset
def load_mvtec_ad(dataset_path, category, transform):
    train_path = os.path.join(dataset_path, category, 'train', 'good')
    test_path = os.path.join(dataset_path, category, 'test')
    train_dataset = MVTecDataset(train_path, transform=transform)
    test_dataset = MVTecDataset(test_path, transform=transform)
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# PaDiM Implementation
class PaDiM:
    def __init__(self, feature_extractor, device):
        self.feature_extractor = feature_extractor.to(device)
        self.device = device

    def fit(self, dataloader):
        self.feature_extractor.eval()
        features = []
        with torch.no_grad():
            for batch, _ in dataloader:
                batch = batch.to(self.device)
                feature,__,___ = self.feature_extractor(batch)
                features.append(feature.cpu().numpy())
        features = np.concatenate(features, axis=0)
        features = features.reshape([batch.shape[0],-1])
        N, C = features.shape
        features = features.reshape(N, C, -1).transpose(0, 2, 1).reshape(-1, C)
        self.mean = np.mean(features, axis=0)
        self.cov = LedoitWolf().fit(features).covariance_

    def anomaly_score(self, feature):
        feature = feature.reshape(-1, feature.shape[1])
        inv_cov = np.linalg.inv(self.cov)
        diff = feature - self.mean
        scores = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        return scores.reshape(IMAGE_SIZE // 16, IMAGE_SIZE // 16)

    def predict(self, dataloader):
        self.feature_extractor.eval()
        anomaly_maps = []
        with torch.no_grad():
            for batch, _ in dataloader:
                batch = batch.to(self.device)
                feature = self.feature_extractor(batch)
                for i in range(feature.size(0)):
                    score = self.anomaly_score(feature[i].cpu().numpy())
                    anomaly_maps.append(score)
        return anomaly_maps

# Transformations
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Data
train_loader, test_loader = load_mvtec_ad(DATASET_PATH, CATEGORY, transform)

# Initialize Feature Extractor and PaDiM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
feature_extractor = FeatureExtractor(WEIGHTS_PATH,MODEL_NAME)
padim = PaDiM(feature_extractor, device)

# Train PaDiM
padim.fit(train_loader)

# Predict Anomalies
anomaly_maps = padim.predict(test_loader)

# Visualize Anomaly Maps
for i, anomaly_map in enumerate(anomaly_maps):
    plt.subplot(1, len(anomaly_maps), i+1)
    plt.imshow(anomaly_map, cmap='hot')
    plt.title(f'Anomaly {i+1}')
plt.show()
