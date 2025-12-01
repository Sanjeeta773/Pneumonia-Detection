import torch
import matplotlib.pyplot as plt
import os


def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)  # (B,1,H,W)
    targets = targets.float()
    dims = (1,2,3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_score_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    dims = (1,2,3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def save_loss_curve(train_losses, val_losses, val_metrics, out_dir, filename="loss_curve.png"):
    """
    Saves train/validation loss curves to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(val_metrics, label='Validation Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved loss curve at: {save_path}")