from masking import MaskingModule
import matplotlib.pyplot as plt
from PIL import Image
import requests
import numpy as np
from models_mae import mae_vit_base_patch16
import torch
print(torch.__version__)


masking = MaskingModule()

img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((224, 224))
img = np.array(img) / 255.

# plot original image left
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# create entropy map
img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # (1, 3, 224, 224)
model = mae_vit_base_patch16()
patchified_img = model.patchify(img) # [1, L, 3*16*16]
print(patchified_img.shape)
entropies = masking.entropy(patchified_img, dim=-1) # [1, L]
print(entropies.shape)
entropies = entropies.squeeze().reshape(14, 14)



# convert to numpy
entropies = entropies.detach().cpu().numpy()

# plot entropy map right
plt.subplot(1, 2, 2)
plt.imshow(entropies, cmap='hot')
plt.title('Entropy Map')
plt.axis('off')
plt.colorbar()

plt.show()