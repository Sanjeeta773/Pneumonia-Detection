import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


@torch.no_grad()
def classification_metrics(logits, labels):
    """
    Binary classification metrics for Pneumonia Detection
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_prob = probs.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }

    return metrics


def save_training_curves(train_losses, val_losses, val_f1, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(val_f1, label="Validation F1-score")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()
