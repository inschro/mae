import torch

# Load the state dictionary from the .pth file
# checkpoint_path = 'mae_pretrain_vit_base.pth'
# checkpoint_path = 'mae_finetuned_vit_base.pth'
# checkpoint_path = 'mae_timm_vit_base.pth'
# checkpoint_path = 'mae_visualize_vit_large.pth'
checkpoint_path = r'C:\Users\Ingo\Desktop\Code Stuff\mae\mae\jobs\20240710003215\outputs\checkpoint-5.pth'
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Function to recursively print all keys in a nested dictionary
def print_keys(d, parent_key=''):
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f'{parent_key}.{k}' if parent_key else k
            print_keys(v, new_key)
    else:
        print(parent_key)
        if isinstance(d, torch.Tensor): 
            print(d.shape, end='\t\t')
        else:
            print(d, end='\t\t')


print(state_dict['args'])
