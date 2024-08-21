import sys
sys.path.append(r'C:\Users\Ingo\Desktop\Code Stuff\mae\mae')

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from timm import create_model
import numpy as np
import os
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import models_mae
from PIL import Image
# A script that chat gpt created after a little bit of guidance and further prompting

# Parameters
BATCH_SIZE = 16
IMAGE_SIZE = 224
MODEL_NAME = 'mae_vit_large_patch16'
DATASET_PATH = r'C:\Users\Ingo\Desktop\mvtec_anomaly_detection'
WEIGHTS_PATH = r'C:\Users\Ingo\Desktop\Code Stuff\mae\mae\mae_visualize_vit_large.pth'
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
            for i, (batch, _) in enumerate(dataloader):
                print(f'Extracting features: {i * BATCH_SIZE}/{len(dataloader.dataset)}', end='\r')
                batch = batch.to(self.device)
                feature,__,___ = self.feature_extractor(batch)
                feature = feature[:,:-1,:] #ignore the cls token
                features.append(feature.cpu().numpy())
        features = np.concatenate(features, axis=0)
        N, L, C = features.shape
        features = features.reshape(-1, C)  # (N_batches * batch_size * L, C)
        self.mean = np.mean(features, axis=0)
        self.cov = LedoitWolf().fit(features).covariance_

    def anomaly_score(self, feature):
        L, C = feature.shape
        inv_cov = np.linalg.inv(self.cov)
        diff = feature - self.mean
        scores = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        return scores.reshape(int(np.sqrt(L)), int(np.sqrt(L)))
    
    def predict(self, dataloader):
        self.feature_extractor.eval()
        anomaly_maps = []
        original_images = []
        with torch.no_grad():
            for batch, _ in dataloader:
                batch = batch.to(self.device)
                feature,__,___ = self.feature_extractor(batch)
                feature = feature[:,:-1,:] #ignore the cls token
                for i in range(feature.size(0)):
                    score = self.anomaly_score(feature[i].cpu().numpy())
                    anomaly_maps.append(score)
                    original_images.append(batch[i].cpu().numpy().transpose(1, 2, 0))  # Save original image
        return anomaly_maps, original_images

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


anomaly_maps, original_images = padim.predict(test_loader)
print("anomaly maps calculation done")

# Visualize Anomaly Maps with Original Images using matplotlib
num_maps = min(10, len(anomaly_maps))  # Display at most 10 anomaly maps
fig, axes = plt.subplots(2, num_maps, figsize=(20, 10))

import cv2

for i in range(num_maps):
    original_image = original_images[i]
    anomaly_map = anomaly_maps[i]

    # Plot original image
    axes[0, i].imshow((original_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
    axes[0, i].set_title(f'Original {i+1}')
    axes[0, i].axis('off')

    # Plot anomaly map overlayed on the original image
    axes[1, i].imshow((original_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
    axes[1, i].imshow(cv2.resize(anomaly_map,(IMAGE_SIZE,IMAGE_SIZE)), cmap='hot', alpha=0.5)  # Overlay anomaly map
    axes[1, i].set_title(f'Anomaly {i+1}')
    axes[1, i].axis('off')

plt.show()

