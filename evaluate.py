import torch
from utils import classification_metrics


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        outputs = model(images)
        all_logits.append(outputs)
        all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = classification_metrics(logits, labels)

    print("Validation results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    from args import get_args
    from datasets import get_loaders
    from model import SimpleClassifier

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = get_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    model = SimpleClassifier(input_channels=1)
    model.load_state_dict(torch.load("output/best_model.pth", map_location=device))
    model.to(device)

    evaluate(model, val_loader, device)
