import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Model training options')

    parser.add_argument('-data_dir', type=str, default='data/chest_xray',
                        help='Root folder for pneumonia dataset with train/val/test subfolders')
    parser.add_argument('-img_size', type=int, default=224, help='Image resize for training')
    parser.add_argument('-model', type=str, default='classifier', choices=['classifier','unet'],
                        help='Model to use: classifier for pneumonia detection, unet for segmentation')

    parser.add_argument('-batch_size', type=int, default=16,
                        choices=[16, 32, 64])

    parser.add_argument('-lr', type=float, default=1e-3,) # learning rate

    parser.add_argument('-wd', type=float, default=1e-5, ) #weight decay

    parser.add_argument('-epochs', type=float, default=10)

    parser.add_argument('-out_dir', type=str, default='projects/session')

    args = parser.parse_args()

    return args