from models_mae import mae_vit_base_patch16

from torch.profiler import profile, record_function, ProfilerActivity

from torch.optim import AdamW

import torch

device = "cuda"

model = mae_vit_base_patch16().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

input_tensor = torch.randn(32, 3, 224, 224).to(device)

num_iter = 1

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
    with record_function("random_75"):
        print("random_75")
        for _ in range(num_iter):
            loss, _, _ = model(input_tensor, masking_type="random_masking", masking_ratio=0.75)
            loss.backward()
            optimizer.step()
print(prof.key_averages().table(top_level_events_only=True, row_limit=10))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
    with record_function("entropy_75"):
        print("entropy_75")
        for _ in range(num_iter):
            loss, _, _ = model(input_tensor, masking_type="entropy_masking", masking_ratio=0.75)
            loss.backward()
            optimizer.step()
print(prof.key_averages().table(top_level_events_only=True, row_limit=10))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
    with record_function("entropy_80"):
        print("entropy_80")
        for _ in range(num_iter):
            loss, _, _ = model(input_tensor, masking_type="entropy_masking", masking_ratio=0.80)
            loss.backward()
            optimizer.step()

print(prof.key_averages().table(sort_by="flops", row_limit=10))
