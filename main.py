import os
from args import get_args
import pandas as pd
import torch
from datasets import get_loaders
from torch.utils.data import DataLoader
from model import UNetLext
from trainer import train_model


def main(xray=None):
    args = get_args()

    if args.model == 'classifier':
        # use folder dataset structure
        train_loader, val_loader, test_loader = get_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
        from model import SimpleClassifier
        model = SimpleClassifier(input_channels=1)
    else:
        raise NotImplementedError('Segmentation mode is not supported after dataset refactor. Use -model classifier')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_model(model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()

