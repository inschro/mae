
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import models_mae

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, masking_type, **masking_args):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), masking_type=masking_type, **masking_args)
    print(f"Loss: {loss.item()}")
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()




# load an image
#img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
#img_url = 'https://www.travelandleisure.com/thmb/h97kSvljd2QYH2nUy3Y9ZNgO_pw=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/plane-data-BUSYROUTES1217-f4f84b08d47f4951b11c148cee2c3dea.jpg'
#img_url = 'https://www.dailypaws.com/thmb/ZHs0nxwPjwixC4YkqyRcO9DB2bg=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/striped-cat-playing-flower-552781357-2000-f8b1f07162594830bdba8a24e2268ad6.jpg'
#img_url = 'https://cataas.com/cat'

img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

plt.rcParams['figure.figsize'] = [5, 5]
show_image(torch.tensor(img))

masking_args = {
    'masking_ratio': 0.0,
    'threshold': 0.4,
    'ratios' : [0.95, 0.8, 0.7, 0.6, 0.5], #effective ratio = 0.2 + 0.25 + 0.3 = 0.75
    'random' : False
}


# chkpt_dir = r'/home/darius/Dokumente/Research/mae/jobs/20240712135557/outputs/checkpoint-10.pth'
# model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
# run_one_image(img, model_mae, masking_type='random_masking', **masking_args)
# run_one_image(img, model_mae, masking_type='entropy_masking', **masking_args)

chkpt_dir = r'/home/darius/Dokumente/Research/mae/jobs/20240722121452/outputs/checkpoint-70.pth'
#chkpt_dir = r'/home/darius/Dokumente/Research/mae/jobs/20240713152426/outputs/checkpoint-399.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
#run_one_image(img, model_mae, masking_type='random_masking', **masking_args)
#run_one_image(img, model_mae, masking_type='entropy_masking', **masking_args)
#run_one_image(img, model_mae, masking_type='entropy_masking_bins', **masking_args)
run_one_image(img, model_mae, masking_type='random_masking', **masking_args)
