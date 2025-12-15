from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils import classification_metrics, save_training_curves


def train_model(model, train_loader, val_loader, device):
    args = get_args()

    os.makedirs(args.out_dir, exist_ok=True)
    best_model_path = os.path.join(args.out_dir, "best_model.pth")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    model.to(device)

    best_f1 = 0.0
    train_losses, val_losses, val_f1_scores = [], [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, val_f1 = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print("✅ Best model saved")

    save_training_curves(train_losses, val_losses, val_f1_scores, args.out_dir)
    print(f"Training complete. Best F1-score: {best_f1:.4f}")


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_logits, all_labels = [], []

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        all_logits.append(outputs)
        all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = classification_metrics(logits, labels)
    return val_loss / len(val_loader), metrics["f1"]
