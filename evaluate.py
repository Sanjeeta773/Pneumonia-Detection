import os
import torch
import matplotlib.pyplot as plt
from datasets import PneumoniaDataset, get_loaders
from torch.utils.data import DataLoader
from model import UNetLext, SimpleClassifier
from utils import dice_score_from_logits
import pandas as pd
from args import get_args

def evaluate(model, val_loader, device):
    args = get_args()
    model.eval()
    all_scores = []

    with torch.no_grad():
        for data in val_loader:
            images = data['image'].to(device)
            if getattr(args, 'model', 'classifier') == 'classifier':
                labels = data['label'].to(device).unsqueeze(1)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                acc = (preds == labels).float().mean().item()
                all_scores.append(acc)
            else:
                masks = data['masks'].to(device)
                outputs = model(images)
                dice = dice_score_from_logits(outputs, masks)
                all_scores.append(dice)

    avg_score = sum(all_scores) / len(all_scores)
    metric_name = 'Accuracy' if getattr(args, 'model', 'classifier') == 'classifier' else 'Dice'
    print(f"Average {metric_name} on Validation Set: {avg_score:.4f}")
    return all_scores

def visualize_predictions(model, dataset, device, num_samples=5, out_dir='predictions'):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    # For classifier mode we save sample images with predicted label in filename
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image']
        with torch.no_grad():
            inp = image.unsqueeze(0).to(device)
            out = model(inp)
            prob = torch.sigmoid(out).item()
            pred = 'PNEUMONIA' if prob > 0.5 else 'NORMAL'

        plt.figure(figsize=(4,4))
        plt.imshow(image[0].cpu(), cmap='gray')
        plt.title(f'Pred: {pred} ({prob:.2f})')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, f'prediction_{i}_{pred}_{prob:.2f}.png'))
        plt.close()

    print(f"Saved {min(num_samples, len(dataset))} predictions to {out_dir}")

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == 'classifier':
        # use folder dataset
        _, val_loader, _ = get_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
        # pick a small dataset instance for visualization
        val_ds = PneumoniaDataset(args.data_dir, split='val', img_size=args.img_size)
        model = SimpleClassifier(input_channels=1)
    else:
        raise NotImplementedError('Segmentation evaluation is not supported after dataset refactor. Use -model classifier')

    best_model_path = "output/best_model.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    # Evaluate
    evaluate(model, val_loader, device)
    visualize_predictions(model, val_ds, device, num_samples=5, out_dir=args.out_dir)
