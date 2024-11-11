import torch
from torch import nn
import torch.nn.functional as F
from torch.fft import fft as torchfft #weird version bug
torch.pi = torch.acos(torch.zeros(1)).item() * 2

class MaskingModule(nn.Module):
    def __init__(self, epsilon = 1e-19):
        super(MaskingModule, self).__init__()
        self.epsilon = epsilon
        

    def forward(self, x, img_pat, masking_type, **masking_args):
        # Use getattr to dynamically select the method based on mask_type
        if hasattr(self, masking_type):
            method = getattr(self, masking_type)
            return method(x,img_pat, **masking_args)
        else:
            raise ValueError(f"Unsupported mask type: {masking_type}")
        
    def random_masking(self, x, img_pat, masking_ratio=0.75, **kwargs): #TODO THIS is a temporary workaround as img_pat is only accessed in entropy based masking
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - masking_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def random_masking_variable(self, x, img_pat, masking_ratio_min=0.5, masking_ratio_max=0.75, **kwargs):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        masking_ratio = torch.rand(1).item() * (masking_ratio_max - masking_ratio_min) + masking_ratio_min
        return self.random_masking(x, masking_ratio=masking_ratio, **kwargs)
    
    def entropy_masking(self, x, img_pat, masking_ratio=0.75, reverse=False, **kwargs): #TODO THIS is a temporary workaround as img_pat is only accessed in entropy based masking
        """
        Perform per-sample entropy-based masking by sorting by entropy.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - masking_ratio))

        # compute entropy
        entropies = self.entropy(img_pat, num_bins=10)
        
        # sort by entropy
        descending = not reverse
        ids_shuffle = torch.argsort(entropies, dim=1, descending=descending) # descend: large is keep, small is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def frequency_masking(self, x,img_pat, masking_ratio=0.75, **kwargs):
        """
        Perform per-sample frequency-based masking by sorting by frequency.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - masking_ratio))

        # compute mean frequency
        mean_frequencies = self.mean_frequency(img_pat)
        
        # sort by entropy
        ids_shuffle = torch.argsort(mean_frequencies, dim=1, descending=True) # descend: large is keep, small is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def entropy_kde_masking(self, x, img_pat, masking_ratio=0.75, **kwargs): #TODO THIS is a temporary workaround as img_pat is only accessed in entropy based masking
        """
        Perform per-sample entropy-based masking by sorting by entropy.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - masking_ratio))

        # compute entropy
        entropies = self.entropy_kde(img_pat)
        
        # sort by entropy
        ids_shuffle = torch.argsort(entropies, dim=1, descending=False) # descend: large is keep, small is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    
    def entropy_masking_bins(self, x, img_pat, ratios=[0.99, 0.0, 0.005, 0.99], random = True, **kwargs):
        """
        Perform per-sample entropy-based masking by sorting by entropy.
        ratios: list of float, ratios of patches to remove, from lowest to highest entropy
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        num_intervals = len(ratios)
        
        # Compute entropy
        entropies = self.entropy_kde(img_pat) # [N, L]
        
        # Sort by increasing entropy
        ids_sorted = torch.argsort(entropies, dim=1, descending=True)
        ids_restore = torch.argsort(ids_sorted, dim=1)

        bin_size = L // num_intervals

        # create bins of shape [N, bin_size, num_intervals]
        bins = torch.ones([N, num_intervals, bin_size], device=x.device)
        for bin_idx, ratio in enumerate(ratios):
            len_keep = int(bin_size * (1 - ratio))
            bins[:, bin_idx, :len_keep] = 0

            if random:
                bins[:, bin_idx] = bins[:, bin_idx, torch.randperm(bin_size)]

        # flatten bins to mask
        mask = bins.reshape([N, L])
        ids_keep = ids_sorted[mask==0].reshape([N,-1])
        
        # Unshuffle the mask to get the original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Create the masked x based on the mask
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        
        return x_masked, mask, ids_restore
    
    def entropy_masking_threshold(self, x, img_pat, threshold=0.5, **kwargs):
        """
        Perform per-sample entropy-based masking by thresholding entropy.
        x: [N, L, D], sequence
        threshold: float, threshold value for entropy to keep
        """
        N, L, D = x.shape

        # compute entropy
        entropies = self.entropy(img_pat, num_bins=10)

        # sort by entropy
        ids_shuffle = torch.argsort(entropies, dim=1, descending=True) # descend: large is keep, small is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # count the number of tokens to keep, make sure at least one token is kept and at least one token is removed
        minimum_keep = kwargs.get('minimum_keep', 1)
        minimum_mask = kwargs.get('minimum_mask', 1)
        len_keep = (entropies > threshold).sum(dim=1)
        len_keep = torch.clamp(len_keep, min=minimum_keep, max=L - minimum_mask).float().mean().round().long().item()
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    @staticmethod
    def entropy(tensor, dim=-1, num_bins=64):
        """
        Calculate the entropy of a tensor along a specified dimension.

        Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to calculate the entropy. Default is the last dimension.
        num_bins (int): Number of bins to quantize the tensor values.

        Returns:
        torch.Tensor: Tensor containing entropy values along the specified dimension.
        """
        # Normalize the tensor to range [0, 1]
        min_val, _ = torch.min(tensor, dim=dim, keepdim=True)
        max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-19)

        # Quantize the tensor
        quantized_tensor = torch.floor(normalized_tensor * (num_bins - 1)).long()

        # Calculate the counts of each unique value along the specified dimension
        unique_vals, inverse_indices = torch.unique(quantized_tensor, sorted=True, return_inverse=True)
        counts = torch.zeros_like(quantized_tensor, dtype=torch.float).scatter_add_(dim, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))

        # Calculate probabilities
        probs = counts / counts.sum(dim=dim, keepdim=True)
        
        # Calculate the log of the probabilities
        log_probs = torch.log(probs + 1e-9)  # Add a small value to avoid log(0)
        
        # Calculate the entropy
        entropy = -torch.sum(probs * log_probs, dim=dim)
        
        return entropy
    
    def entropy_kde(self, values: torch.Tensor, n_bins = 64 , sigma = 0.01): 
        """
        Calculate the entropy of a tensor along a specified dimension.

        Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to calculate the entropy. Default is the last dimension.
        num_bins (int): Number of bins to quantize the tensor values.

        Returns:
        torch.Tensor: Tensor containing entropy values along the specified dimension.
        """
        bins = torch.linspace(0,1,n_bins).to(values.device)
        sigma = torch.tensor(sigma).to(values.device)
        values_min = values.min()
        values_max = values.max()
        
        # Apply min-max normalization
        normalized_tensor = (values - values_min) / (values_max - values_min)
        pdf = self.__marginal_pdf_kde(normalized_tensor,bins,sigma)
        entropy = - torch.sum(pdf * torch.log(pdf), dim = -1)
        return entropy
    
    
    def __marginal_pdf_kde(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor):
        """
        Calculates the marginal pdf of a batch of Tensors using kernel density estimation with gaussian kernel.

        Args:
        values (torch.Tensor): Input tensor of shape (N, P, L).
        bins (torch.Tensor): Tensor containing bin positions.
        sigma (torch.Tensor): Standard deviation for the Gaussian kernel.

        Returns:
        torch.Tensor: Tensor containing estimated marginal pdf of shape (N, P, B).
        """
        N, P, L = values.shape
        B = bins.shape[0]
        
        # Expand dimensions of bins and sigma for broadcasting
        bins = bins.unsqueeze(0).unsqueeze(0).unsqueeze(3)  # Shape: (1, 1, B, 1)
        sigma = sigma.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # Shape: (1, 1, 1, L)
        
        # Calculate residuals
        residuals = values.unsqueeze(2) - bins  # Shape: (N, P, B, L)
        
        # Apply Gaussian kernel
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))  # Shape: (N, P, B, L)
        
        # Calculate pdf
        pdf = torch.mean(kernel_values, dim=3)  # Shape: (N, P, B)
        
        # Normalize pdf
        normalization = torch.sum(pdf, dim=2, keepdim=True) + self.epsilon  # Shape: (N, P, 1)
        pdf = pdf / normalization + self.epsilon  # Shape: (N, P, B)
        
        return pdf
        
    @staticmethod
    def mean_frequency(in_tensor):
        """
        Calculate the mean frequency of each unique value along the specified dimension.

        Args:
        in_tensor (torch.Tensor): Input tensor. (N, P, L)
        dim (int): Dimension along which to calculate the mean frequency. Default is the last dimension.

        Returns:
        torch.Tensor: Tensor containing mean frequency values along the specified dimension. (N, P)
        """
        # unshuffle color channels
        red = in_tensor[:, :, 0::3]
        green = in_tensor[:, :, 1::3]
        blue = in_tensor[:, :, 2::3]

        print(in_tensor[0, 0, :20])
        print(red.shape, green.shape, blue.shape, in_tensor.shape)

        gray = red + green + blue

        # Use fft along the specified dimension
        frequencies = torch.fft(gray, signal_ndim=2)
        frequencies = torch.abs(frequencies)

        # Calculate the mean frequency
        mean_frequency = torch.mean(frequencies, dim=2)

        return mean_frequency
    
    @staticmethod
    def dct2d(x):
        """
        Implement 2D DCT using FFT
        """
        X1 = torchfft(x, dim=2)
        X2 = torchfft(X1, dim=3)

        def dct_kernel(n):
            k = torch.arange(n, dtype=x.dtype, device=x.device)
            return torch.cos((torch.pi / (2 * n)) * k)

        k1 = dct_kernel(x.shape[2]).view(1, 1, -1, 1)
        k2 = dct_kernel(x.shape[3]).view(1, 1, 1, -1)

        return 4 * torch.real(X2 * k1 * k2)
    
    def calculate_patch_information(self, x, codec_type='jpeg', num_bins=64):
        """
        Calculate the information within a patch using different image codecs.
        
        Args:
        x (torch.Tensor): Input tensor of shape [N, L, D], where N is batch size,
                          L is number of patches, and D is patch dimension.
        codec_type (str): Type of codec to simulate ('jpeg', 'png', or 'entropy').
        num_bins (int): Number of bins for entropy calculation.
        
        Returns:
        torch.Tensor: Information content of each patch.
        """
        
        N, L, D = x.shape
        
        if codec_type == 'jpeg':
            # Reshape to 2D representation without assuming square patch
            x_reshaped = x.view(N * L, 1, -1, 8)  # Assume minimum width of 8
            
            # Pad to next power of 2 if necessary
            h, w = x_reshaped.shape[2], x_reshaped.shape[3]
            target_h = 2**((h - 1).bit_length())
            target_w = 2**((w - 1).bit_length())
            pad_h = target_h - h
            pad_w = target_w - w
            
            if pad_h > 0 or pad_w > 0:
                x_reshaped = F.pad(x_reshaped, (0, pad_w, 0, pad_h))
            
            dct_coeffs = self.dct2d(x_reshaped)
            information = torch.sum(torch.abs(dct_coeffs), dim=(1, 2, 3)).view(N, L)
        
        elif codec_type == 'png':
            # Reshape to 2D representation without assuming square patch
            x_reshaped = x.view(N * L, 1, -1, 8)  # Assume minimum width of 8
            
            grad_x = F.conv2d(x_reshaped, torch.tensor([[[[1, -1]]]], dtype=x.dtype, device=x.device), padding=(0, 1))
            grad_y = F.conv2d(x_reshaped, torch.tensor([[[[1], [-1]]]], dtype=x.dtype, device=x.device), padding=(1, 0))
            information = (torch.sum(torch.abs(grad_x), dim=(1, 2, 3)) + 
                           torch.sum(torch.abs(grad_y), dim=(1, 2, 3))).view(N, L)
        
        elif codec_type == 'entropy':
            information = self.entropy(x, dim=-1, num_bins=num_bins)
        
        else:
            raise ValueError(f"Unsupported codec type: {codec_type}")
        
        return information
    
    def codec_based_masking(self, x, img_pat, masking_ratio=0.75, codec_type='png', num_bins=64, reverse = False, **kwargs):
        """
        Perform per-sample information-based masking by sorting patches based on their information content.
        
        Args:
        x (torch.Tensor): Input tensor of shape [N, L, D], where N is batch size,
                          L is number of patches, and D is patch dimension.
        img_pat (torch.Tensor): Image patches (not used in this method, kept for consistency with other methods).
        masking_ratio (float): Ratio of patches to mask.
        codec_type (str): Type of codec to use for information calculation ('jpeg', 'png', or 'entropy').
        num_bins (int): Number of bins for entropy calculation (used only if codec_type is 'entropy').
        **kwargs: Additional keyword arguments.
        
        Returns:
        tuple: (x_masked, mask, ids_restore)
            x_masked (torch.Tensor): Masked input tensor.
            mask (torch.Tensor): Binary mask tensor.
            ids_restore (torch.Tensor): Tensor containing indices to restore original order.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - masking_ratio))

        # Calculate patch information
        patch_info = self.calculate_patch_information(x, codec_type=codec_type, num_bins=num_bins)
        
        # Sort patches by information content
        ids_shuffle = torch.argsort(patch_info, dim=1, descending=(not reverse))
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the patches with highest information content
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def grid_masking_patch16_7x7(self, x : torch.Tensor, img_pat, masking_ratio=0.25, reverse=True, random = False, **kwargs):
        """
        Divide the patches into 7x7 regions and mask a specified percentage of patches 
        in each region based on a criterion (e.g., highest or lowest values).
        
        Args:
        - x (torch.Tensor): Input tensor of shape [N, L, D].
        - img_pat (torch.Tensor): Image patches (not directly used in this method).
        - masking_ratio (float): Ratio of patches to mask in each 7x7 region.
        - mask_highest (bool): If True, mask the patches with the highest values, otherwise mask the lowest.

        Returns:
        - x_masked (torch.Tensor): Masked input tensor.
        - mask (torch.Tensor): Binary mask tensor.
        - ids_restore (torch.Tensor): Indices to restore the original order.
        """
        
        N, L, D = x.shape
        grid_size = 7
        len_keep = int(grid_size**2 * (1 - masking_ratio)) #

        # compute entropy
        entropies = self.entropy(img_pat, num_bins=64)
        
        # terrible hack
        offset = torch.zeros_like(entropies)
        offset = offset.view(N, 14, 14)
        offset = offset.unfold(1, grid_size, grid_size).unfold(2, grid_size, grid_size)
        offset = offset.reshape(N, -1, grid_size**2)
        offset[:, 1, :] = 100
        offset[:, 2, :] = 200
        offset[:, 3, :] = 300
        offset = offset.reshape(N, 2, 2, grid_size, grid_size)
        offset = offset.permute(0, 1, 3, 2, 4).reshape(N, 14, 14)
        offset = offset.view(N, 196)
        entropies += offset
        
        num_intervals = 4
        
        
        # Sort by increasing entropy
        ids_sorted = torch.argsort(entropies, dim=1, descending= not reverse)
        ids_restore = torch.argsort(ids_sorted, dim=1)

        bin_size = L // num_intervals
        
        # i shall be cursed
        bins = torch.ones([N, num_intervals, bin_size], device=x.device)
        bins[:, :, :len_keep] = 0

        # flatten bins to mask
        mask = bins.reshape([N, L])
        ids_keep = ids_sorted[mask==0].reshape([N,-1])
        
        # Unshuffle the mask to get the original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Create the masked x based on the mask
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        
        return x_masked, mask, ids_restore
    
        

    def grid_masking_patch16_2x2(self, x, img_pat, masking_ratio=0.5, reverse=True, **kwargs):
        """
        Divide the patches into 2x2 regions and mask a specified percentage of patches 
        in each region based on a criterion (e.g., highest or lowest values).
        
        Args:
        - x (torch.Tensor): Input tensor of shape [N, L, D].
        - img_pat (torch.Tensor): Image patches (not directly used in this method).
        - masking_ratio (float): Ratio of patches to mask in each 2x2 region.
        - mask_highest (bool): If True, mask the patches with the highest values, otherwise mask the lowest.

        Returns:
        - x_masked (torch.Tensor): Masked input tensor.
        - mask (torch.Tensor): Binary mask tensor.
        - ids_restore (torch.Tensor): Indices to restore the original order.
        """
        N, L, D = x.shape
        grid_size = 2
        len_keep = int(grid_size**2 * (1 - masking_ratio)) #

        # compute entropy
        entropies = self.entropy(img_pat, num_bins=64)
        
        # terrible hack ... hardcode this perhaps?
        offset = torch.zeros_like(entropies)
        offset = offset.view(N, 14, 14)
        offset = offset.unfold(1, grid_size, grid_size).unfold(2, grid_size, grid_size)
        offset = offset.reshape(N, -1, grid_size**2)
        for i in range(48):
            offset[:, (i+1), :] = (i+1)*100
        offset = offset.reshape(N, 7, 7, grid_size, grid_size)
        offset = offset.permute(0, 1, 3, 2, 4).reshape(N, 14, 14)
        offset = offset.view(N, 196)
        entropies += offset
        
        num_intervals = 49
        
        
        # Sort by increasing entropy
        ids_sorted = torch.argsort(entropies, dim=1, descending= not reverse)
        ids_restore = torch.argsort(ids_sorted, dim=1)

        bin_size = L // num_intervals

        # i shall be cursed
        bins = torch.ones([N, num_intervals, bin_size], device=x.device)
        bins[:, :, :len_keep] = 0

        # flatten bins to mask
        mask = bins.reshape([N, L])
        ids_keep = ids_sorted[mask==0].reshape([N,-1])
        
        # Unshuffle the mask to get the original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Create the masked x based on the mask
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        
        return x_masked, mask, ids_restore
    
if __name__ == '__main__':
    import requests
    from PIL import Image
    import numpy as np
    import models_mae
    import matplotlib
    #matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    img_url = 'https://www.travelandleisure.com/thmb/h97kSvljd2QYH2nUy3Y9ZNgO_pw=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/plane-data-BUSYROUTES1217-f4f84b08d47f4951b11c148cee2c3dea.jpg'

    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    masking = MaskingModule()

    print(img.shape)

    autoencoder = models_mae.mae_vit_base_patch16()

    x = autoencoder.patchify(torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2))

    print(x.shape)

    masked_x, mask, ids_restore = masking.grid_masking_patch16_2x2(x, x)

    print(masked_x.shape, mask.shape, ids_restore.shape)
    print(mask[0, :20])
    print(ids_restore[0, :20])

    masked_x = autoencoder.unpatchify(masked_x)

    print(masked_x.shape)

    plt.imshow(masked_x[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.title('Masked Image')


    plt.show()