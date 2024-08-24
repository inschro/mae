import argparse
import os
import torch
from torchvision import datasets, transforms
from models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
import yaml
from datetime import datetime
import json

def log_evaluation_header(eval_config, log_file, checkpoint=None):
    log_file.write(f"EVALUATION OF MAE AUTOENCODER\n\n")
    log_file.write(f"Evaluation configuration:\n{json.dumps(eval_config, indent=2)}\n\n")
    
    if checkpoint is not None:
        if checkpoint.get('epoch') is not None:
            log_file.write(f"Checkpoint loaded from epoch {checkpoint['epoch']}\n")
        if checkpoint.get('args') is not None:
            args_as_dict = vars(checkpoint['args'])
            args_json = json.dumps(args_as_dict, indent=2)
            log_file.write(f"Training configuration:\n{args_json}\n\n")

    log_file.write(f"------------------------------------------------------------------------------------\n Evaluation results:\n\n")
    log_file.flush()
                
def evaluate_on_setting(model, dataloader, masking_type, masking_ratio, log_file, num_samples, batch_size):
    log_file.write(f"Masking type: {masking_type}\t Masking ratio: {masking_ratio}: \t")
    print(f"\n Masking type: {masking_type}\t Masking ratio: {masking_ratio}: \t")
    with torch.no_grad():
        total_loss = 0
        for idx, (samples, _) in enumerate(dataloader):
            print(f"Batch {idx+1}/{num_samples}", end='\r')
            samples = samples.to(args.device)
            loss, _, _  = model(samples, masking_type=masking_type, masking_ratio=masking_ratio)
            total_loss += loss.item()
            if idx + 1 >= num_samples:
                log_file.write(f"Loss: {total_loss/(num_samples)}\n")
                break
        else:
            log_file.write(f"Loss: {total_loss/len(dataloader)}\n")
        log_file.flush()


            
def main(args):
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)

    match config['model']['architecture']:
        case 'mae_vit_base_patch16':
            model = mae_vit_base_patch16()
        case 'mae_vit_large_patch16':
            model = mae_vit_large_patch16()
        case 'mae_vit_huge_patch14':
            model = mae_vit_huge_patch14()
        case _:
            raise ValueError(f"Model {config['model']['architecture']} not recognized")
        
    checkpoint = torch.load(config['model']['checkpoint'], map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)
    model.eval()

    transform_eval = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset_eval = datasets.ImageFolder(config['data']['path'], transform=transform_eval)
    dataset_eval = torch.utils.data.Subset(dataset_eval, range(config['evaluation']['num_samples'] * config['data']['batch_size']))
    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle=config['data']['shuffle'],
        pin_memory=config['data']['pin_memory'],
        drop_last=config['data']['drop_last'],
        persistent_workers=config['data']['persistent_workers']
    )

    # create output directory
    output_path = config['output']['path']
    os.makedirs(output_path, exist_ok=True)
    evaluation_filename = f"evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(os.path.join(output_path, evaluation_filename), 'w') as log_file:
        log_evaluation_header(config, log_file, checkpoint)
        for masking_type in config['evaluation']['masking_types']:
            for masking_ratio in config['evaluation']['masking_ratios']:
                evaluate_on_setting(
                    model,
                    dataloader_eval,
                    masking_type,
                    masking_ratio,
                    log_file,
                    config['evaluation']['num_samples'],
                    config['data']['batch_size']
                )

        
        log_file.write(f"\n\nEvaluation finished\n")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MAE model on ImageNet')
    parser.add_argument('--config_path', default=r'C:\Users\Ingo\Desktop\Code Stuff\mae\mae\configs\config_eval.yaml', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    
    main(args)