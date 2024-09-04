import json
import torch

def print_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint is not None:
        if checkpoint.get('epoch') is not None:
            print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        if checkpoint.get('args') is not None:
            args_as_dict = vars(checkpoint['args'])
            args_json = json.dumps(args_as_dict, indent=2)
            print(f"Training configuration:\n{args_json}\n\n")

if __name__ == "__main__":
    path = r"C:\Users\Ingo\Desktop\Code Stuff\mae\mae\jobs\20240822215634\_linprobe\outputs\checkpoint-89.pth"
    print_checkpoint(path)