from torch.utils.data import Dataset, DataLoader
import os
import torch
from torchvision import datasets, transforms
from PIL import Image


class PneumoniaDataset(Dataset):
    def __init__(self, root, split='train', img_size=224, transform=None):
        self.root = root
        self.split = split
        split_root = os.path.join(root, split)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform

        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Split folder not found: {split_root}")

        # Use ImageFolder to index images and labels
        self.dataset = datasets.ImageFolder(split_root, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # label: 0 -> NORMAL, 1 -> PNEUMONIA (depends on folder ordering)
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.float32)
        }


def get_loaders(data_root, batch_size=16, img_size=224, num_workers=4):
    """Create train/val/test DataLoaders from a data root folder.
    Returns train_loader, val_loader, test_loader
    """
    train_ds = PneumoniaDataset(data_root, split='train', img_size=img_size)
    val_ds = PneumoniaDataset(data_root, split='val', img_size=img_size)
    test_ds = PneumoniaDataset(data_root, split='test', img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

