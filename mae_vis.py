
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import models_mae
import util.masking as masking


# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3

    # Convert image to tensor if necessary
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    # Apply normalization and clip values
    image = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int()

    # Convert back to NumPy for `plt.imshow`
    image = image.numpy()

    # Show image
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
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


    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")
    plt.savefig(f"./demo/tmp/{masking_type}.png")
    plt.close()
    return im_masked[0]

def run_one_image_with_low_entropy_overlay(img, model, masking_type, low_entropy_ratio,im_masked, **masking_args):
    """
    Visualize low-entropy patches by overlaying a red mask.
    """
    x = torch.tensor(img)

    # Prepare input for the model
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # Get patches
    img_patches = model.patchify(torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2))

    # Calculate entropy for each patch
    entropies = masking.MaskingModule().entropy(img_patches,num_bins=256)
    
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

    # Determine low-entropy patches
    N, L, _ = img_patches.shape  # Patches shape
    num_low_entropy = int(L * low_entropy_ratio)
    entropies_flat = entropies.view(N, L)
    sorted_indices = torch.argsort(entropies_flat, dim=1)  # Ascending order
    low_entropy_indices = sorted_indices[:, :num_low_entropy]

    # Visualize original image
    plt.subplot(1, 3, 1)
    show_image(img, "Original Image")

    # Visualize with red overlay for low-entropy patches
    red_overlay = img.copy()
    patch_size = int(224 / model.patch_embed.grid_size[0])  # Assume square grid
    red_mask = np.zeros_like(img)

    for idx in low_entropy_indices[0]:  # Assuming single image
        row = idx // model.patch_embed.grid_size[1]
        col = idx % model.patch_embed.grid_size[1]
        red_mask[
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
        ] = [1.0, 0.0, 0.0]  # Red overlay

    plt.subplot(1, 3, 2)
    plt.imshow(
        torch.clip((torch.tensor(img) * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    )
    plt.imshow(red_mask, alpha=0.5)
    plt.title("Low-Entropy Patches")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    show_image(im_masked, "masked")
    plt.savefig("./demo/tmp/low_entropy_overlay.png")
    plt.close()



def run_heatmap_on_image(img,model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    x = model.patchify(torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2))

    # run MAE
    masker =  masking.MaskingModule()
    y = masker.entropy(x,num_bins=10)
    #y = masker.entropy_kde(x)
    y = y.view((1,14,14))

    plt.subplot(1,3,1)
    plt.imshow(y[0],cmap='hot' )
    plt.subplot(1,3,2)
    plt.imshow(torch.clip((torch.tensor(img) * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.subplot(1,3,3)
    entropy_map = np.array(Image.fromarray(y[0].detach().to("cpu").numpy()).resize((224, 224),Image.NEAREST))
    plt.imshow(torch.clip((torch.tensor(img)* imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.imshow(entropy_map,cmap='hot',alpha=0.5)
    
    plt.savefig("./demo/tmp/heatmap.png")
    plt.close()




# load an image
# img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
# img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
#
# img_url = 'https://www.travelandleisure.com/thmb/h97kSvljd2QYH2nUy3Y9ZNgO_pw=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/plane-data-BUSYROUTES1217-f4f84b08d47f4951b11c148cee2c3dea.jpg'
# img_url = 'https://www.dailypaws.com/thmb/ZHs0nxwPjwixC4YkqyRcO9DB2bg=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/striped-cat-playing-flower-552781357-2000-f8b1f07162594830bdba8a24e2268ad6.jpg'
img_url = 'https://cataas.com/cat'



img = Image.open(requests.get(img_url, stream=True).raw)
# img = Image.open("/home/mae_entr/work/mae/demo/increasing_entropy_image.png")
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

#masking_args = {
#    'masking_ratio': 0.0,
#    'threshold': 0.4,
#    'ratios' : [0.9,0.6,0.6,0.9], #effective ratio = 0.2 + 0.25 + 0.3 = 0.75
#    'random' : True
#}

masking_args = {
    'masking_ratio': 0.75,
    'alpha': 0.1,
    'threshold': 0.4,
    'codec_type' : 'jpeg',
    'reverse' : False,
    'ratios' : [0.9,1,0.8,0.5], #effective ratio = 0.2 + 0.25 + 0.3 = 0.75
    'random' : False,
    'low_entropy_ratio' : 0.0,
}
#torch.random.manual_seed(4)
# chkpt_dir = r'/home/darius/Dokumente/Research/mae/jobs/20240712135557/outputs/checkpoint-10.pth'
# model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
# run_one_image(img, model_mae, masking_type='random_masking', **masking_args)
# run_one_image(img, model_mae, masking_type='entropy_masking', **masking_args)
torch.random.manual_seed(69)
#chkpt_dir = r'/home/mae_entr/work/mae/jobs/pretrain/selective_crop/maelarge_imnet1k_rand60_lowent20/outputs/checkpoint-latest.pth'
chkpt_dir = r'/home/mae_entr/work/mae/jobs/experiments/selective_crop/maelarge_imnet1k_rand75_lowent10_entropyweighting_pixelwise/outputs/checkpoint-latest.pth'
# chkpt_dir = r'/home/mae_entr/work/mae/jobs/pretrain/random75/_pretrain/random75_maelarge_random75_imnet1k_epoch400_warmup20_a100dali/outputs/checkpoint-399.pth'
# chkpt_dir = "/home/mae_entr/work/mae/jobs/pretrain/selective_crop/maelarge_imnet1k_rand60_lowent20/outputs/checkpoint-latest.pth"
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
m = run_one_image(img, model_mae, masking_type='random_masking', **masking_args)
#run_one_image_with_low_entropy_overlay(
#    img, model_mae, masking_type='selective_crop',im_masked=m, **masking_args
#)

#run_one_image(img, model_mae, masking_type='entropy_masking', **masking_args)
#run_one_image(img, model_mae, masking_type='entropy_masking_bins', **masking_args)
#run_one_image(img, model_mae, masking_type='codec_based_masking', **masking_args)
#run_heatmap_on_image(img, model_mae)
