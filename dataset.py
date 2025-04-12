from PIL import Image
from torch.utils.data import Dataset
import torch, torchvision

class ImageDataset(Dataset):
    def __init__(self,
                 paths: list[str],
                 transform: torchvision.transforms =None,
                 labels: torch.Tensor =None,
                 rgb: bool =True):
        
        self.paths = paths
        self.transform = transform
        self.labels = labels
        self.rgb = rgb

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.rgb:
            image = Image.open(self.paths[idx]).convert("RGB")
        else:
            # Grayscale image
            image = Image.open(self.paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image