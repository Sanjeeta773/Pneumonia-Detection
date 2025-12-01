from args import get_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from utils import dice_loss_from_logits, dice_score_from_logits, save_loss_curve


def train_model (model, train_loader, val_loader, device):
    args = get_args()

    
    # Create directory to save weights
    os.makedirs(args.out_dir, exist_ok=True)
    best_model_path = os.path.join(args.out_dir, "best_model.pth")

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay =args.wd)

    model.to(device)

    # We'll support both segmentation (unet) and classification (classifier)
    best_metric = 0.0

    train_losses = []
    val_losses = []
    val_metrics = []
    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0

        for data_batch in train_loader:
            images = data_batch['image'].to(device=device)

            optimizer.zero_grad()

            if getattr(args, 'model', 'classifier') == 'classifier':
                labels = data_batch['label'].to(device=device).unsqueeze(1)
                outputs = model(images)
                loss = bce(outputs, labels)
            else:
                masks = data_batch['masks'].to(device=device)
                outputs = model(images)
                loss_bce = bce(outputs, masks)
                loss_dice = dice_loss_from_logits(outputs, masks)
                loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # validation
        val_loss, val_metric = validate_model(model, val_loader, bce, device)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        if getattr(args, 'model', 'classifier') == 'classifier':
            print(f"Epoch {epoch+1}/{int(args.epochs)} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val Acc {val_metric:.4f}")
        else:
            print(f"Epoch {epoch+1}/{int(args.epochs)} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val Dice {val_metric:.4f}")

        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            print(f" Best model updated (metric={best_metric:.4f}) saved to {best_model_path}")

    save_loss_curve(train_losses, val_losses, val_metrics, args.out_dir)
    mode_name = 'Accuracy' if getattr(args, 'model', 'classifier') == 'classifier' else 'Dice'
    print(f"\nTraining complete. Best {mode_name}: {best_metric:.4f}")




def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    val_score = 0.0

    with torch.no_grad():
        for data_batch in val_loader:
            images = data_batch['image'].to(device)

            if getattr(get_args(), 'model', 'classifier') == 'classifier':
                labels = data_batch['label'].to(device).unsqueeze(1)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                # compute accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                acc = (preds == labels).float().mean().item()
                val_score += acc
            else:
                masks = data_batch['masks'].to(device)
                outputs = model(images)
                loss_bce = loss_fn(outputs, masks)
                loss_dice = dice_loss_from_logits(outputs, masks)
                loss = loss_bce + loss_dice
                val_score += dice_score_from_logits(outputs, masks)

            val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_metric = val_score / len(val_loader)

    return val_epoch_loss, val_epoch_metric











