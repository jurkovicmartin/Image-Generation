import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from dataset import ImageDataset
from models.evaluator import Evaluator

def load_datasets(directory_path: str, image_size: tuple[int, int] = (160, 160)) -> tuple[ImageDataset, ImageDataset]:
    TRAIN_RATIO = (2, 1)
    
    paths = [os.path.join(directory_path, img) for img in os.listdir(directory_path) if img.endswith(("png", "jpg", "jpeg"))]

    transform = transforms.Compose([
        # In case some image has different size
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Normalization
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # Split paths to train and test
    part = len(paths) // sum(TRAIN_RATIO)
    train_paths = paths[:part * TRAIN_RATIO[0]]
    test_paths = paths[part * TRAIN_RATIO[0]:]

    # TRAIN
    train_dataset = ImageDataset(train_paths, transform=transform, rgb=False)
    # TEST
    test_dataset = ImageDataset(test_paths, transform=transform, rgb=False)

    return train_dataset, test_dataset


def main():
    batch_size = 64
    directory = "dataset/profiles_short"
    image_size = (160, 160)

    train_set,  test_set = load_datasets(directory)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = Evaluator(image_size=image_size, input_channels=1, device=device)
    evaluator.train_model(train_loader, epochs=10)
    evaluator.test_model(test_loader)







if __name__ == "__main__":
    main()