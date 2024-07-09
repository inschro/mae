import torch
from torch import nn


class MaskingModule(nn.Module):
    def __init__(self):
        super(MaskingModule, self).__init__()
        

    def forward(self, x, masking_type, **masking_args):
        # Use getattr to dynamically select the method based on mask_type
        if hasattr(self, masking_type):
            method = getattr(self, masking_type)
            return method(x, **masking_args)
        else:
            raise ValueError(f"Unsupported mask type: {masking_type}")
        
    def random_masking(self, x, masking_ratio=0.75):
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
    
    def entropy_masking(self, x, masking_ratio=0.75):
        """
        Perform per-sample entropy-based masking by sorting by entropy.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - masking_ratio))

        # compute entropy
        entropies = self.entropy(x, num_bins=10)
        
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
    
    def entropy_masking_threshold(self, x, threshold=0.5):
        """
        Perform per-sample threshold-based masking. keep the patches with entropy > threshold.
        x: [N, L, D], sequence
        """

        # compute entropy
        entropies = self.entropy(x, num_bins=10)
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = entropies < threshold

        num_keep = (~mask).sum(dim=1)

        ids_keep = torch.argsort(entropies, dim=1, descending=True)[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        return x_masked, mask, ids_keep
    
    @staticmethod
    def entropy(tensor, dim=-1, num_bins=10):
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