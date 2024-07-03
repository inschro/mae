import torch

# Load the state dictionary from the .pth file
checkpoint_path = 'mae_finetuned_vit_base.pth'
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Function to recursively print all keys in a nested dictionary
def print_keys(d, parent_key=''):
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f'{parent_key}.{k}' if parent_key else k
            print_keys(v, new_key)
    else:
        print(parent_key)

# Check if 'model' key exists and is a dictionary
if 'model' in state_dict:
    if isinstance(state_dict['model'], dict):
        print_keys(state_dict['model'])
    else:
        print('The value under "model" key is not a dictionary.')
else:
    print('No "model" key found in the state dictionary.')
