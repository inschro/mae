import torch
from models_mae import mae_vit_base_patch16
import time

model = mae_vit_base_patch16()

masking_type = "random_masking"
masking_args = {
    "masking_ration": 0.75,
}

# test different batch sizes and test memory usage
sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]

for size in sizes:
    input_tensor = torch.randn(size, 3, 244, 244)
    torch.cuda.empty_cache()  # Clear cache before each run
    start_memory = torch.cuda.memory_allocated()
    output = model(input_tensor, masking_type = masking_type, **masking_args)
    end_memory = torch.cuda.memory_allocated()
    print(f"Batch size: {size}, Memory allocated: {end_memory - start_memory} bytes")