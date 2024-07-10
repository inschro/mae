import torch
import json

# load checkpoint
path = r'C:\Users\Ingo\Desktop\Code Stuff\mae\mae\jobs\20240710003215\outputs\checkpoint-5.pth'
checkpoint = torch.load(path)

print(json.dumps(checkpoint, indent=2))