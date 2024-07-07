from torchinfo import summary
from models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14

model = mae_vit_base_patch16()

summary(model, input_size=(1, 3, 224, 224))