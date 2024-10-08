import torch
from torch import nn


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
    
    def entropy_masking(self, x, img_pat, masking_ratio=0.75, **kwargs): #TODO THIS is a temporary workaround as img_pat is only accessed in entropy based masking
        """
        Perform per-sample entropy-based masking by sorting by entropy.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - masking_ratio))

        # compute entropy
        entropies = self.entropy(img_pat, num_bins=10)
        
        # sort by entropy
        ids_shuffle = torch.argsort(entropies, dim=1, descending=True) # descend: large is keep, small is remove
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
        ids_shuffle = torch.argsort(entropies, dim=1, descending=True) # descend: large is keep, small is remove
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
    
if __name__ == '__main__':
    import requests
    from PIL import Image
    import numpy as np
    import models_mae
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

    masked_x, mask, ids_restore = masking.entropy_masking_bins(x, x, ratios=[1, 1, 0, 1], random=False)

    print(masked_x.shape, mask.shape, ids_restore.shape)
    print(mask[0, :20])
    print(ids_restore[0, :20])

    masked_x = autoencoder.unpatchify(masked_x)

    print(masked_x.shape)

    plt.imshow(masked_x[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.title('Masked Image')


    plt.show()