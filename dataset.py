from PIL import Image
from torch.utils.data import Dataset
import torch, torchvision

from typing import Any

class ImageDataset(Dataset):
    def __init__(self,
                 paths: list[str],
                 transform: torchvision.transforms =None,
                 labels: torch.Tensor =None,
                 rgb: bool =True):
        """
        Class that represents a dataset of images suitable for Pytorch DataLoader.

        Args:
            paths (list[str]): List of paths to the images.
            transform (torchvision.transforms, optional): Transformation to be applied to the images. Defaults to None.
            labels (torch.Tensor, optional): Labels of the images. Defaults to None.
            rgb (bool, optional): Whether to load images in RGB or grayscale. Defaults to True.
        """
        self.paths = paths
        self.transform = transform
        self.labels = labels
        self.rgb = rgb

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any | tuple[Any, Any]:
        if self.rgb:
            image = Image.open(self.paths[idx]).convert("RGB")
        else:
            # Grayscale image
            image = Image.open(self.paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            # Return image, label pair
            return image, self.labels[idx]
        else:
            return image
