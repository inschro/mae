import argparse
import os
import torch
from torchvision import datasets, transforms
from models_mae import MaskedAutoencoderViT
from util.misc import load_checkpoint

def evaluate(model, data_loader, device, masking_type, **masking_args):
	model.eval()
	total_loss = 0.0
	with torch.no_grad():
		for imgs in data_loader:
			imgs = imgs[0].to(device)  # Assuming imgs is a tuple of (images, targets)
			loss, _, _ = model(imgs, masking_type, **masking_args)
			total_loss += loss.item()
	avg_loss = total_loss / len(data_loader)
	return avg_loss

def main(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load the model
	model = MaskedAutoencoderViT().to(device)
	load_checkpoint(model, args.checkpoint_path, device)

	# Data loading
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
	])
	dataset = datasets.ImageFolder(args.data_path, transform=transform)
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

	# Evaluate
	avg_loss = evaluate(model, data_loader, device, args.masking_type, masking_ratio=args.masking_ratio)
	print(f"Average Loss: {avg_loss}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate a Masked Autoencoder model.")
	parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
	parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory.")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
	parser.add_argument("--masking_type", type=str, default="random", help="Type of masking to use.")
	parser.add_argument("--masking_ratio", type=float, default=0.75, help="Ratio of tokens to mask.")

	args = parser.parse_args()
	main(args)