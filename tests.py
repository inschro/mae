from models_mae import mae_vit_base_patch16
from torchinfo import summary
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

image_path = './imagenet1k/00005/013319552935532.jpg'
image = Image.open(image_path)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = transform(image).unsqueeze(0)

model = mae_vit_base_patch16()

# load pretrained weights
checkpoint = torch.load('mae_finetuned_vit_base.pth', map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=False)
print(msg)
exit()

with torch.no_grad():
    loss, patches, mask = model(image.float(), mask_ratio=0.5)


output = model.unpatchify(patches)
mask_reshaped = mask.view(1, 14, 14)
upsampled_mask = torch.nn.functional.interpolate(mask_reshaped.unsqueeze(1), size=(224, 224), mode='nearest').squeeze(1)
masked_image = image * upsampled_mask.unsqueeze(1)

# print all the shapes
print('Image Shape:', image.shape)
print('Output Shape:', output.shape)
print('Mask Shape:', mask.shape)
print('Patches Shape:', patches.shape)
print('Mask Reshaped Shape:', mask_reshaped.shape)
print('Upsampled Mask Shape:', upsampled_mask.shape)
print('Masked Image Shape:', masked_image.shape)

# unnormalize the image
image = image * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
output = output * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
masked_image = masked_image * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

# plot original image, masked image and output image
plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.imshow(image.squeeze().permute(1, 2, 0))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(masked_image.squeeze().permute(1, 2, 0))
plt.title('Masked Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(output.squeeze().permute(1, 2, 0))
plt.title('Output Image')
plt.axis('off')

plt.show()



