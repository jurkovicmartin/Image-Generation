import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from dataset import ImageDataset
from models.evaluator import Evaluator
from models.img_generator import ImageGenerator


def load_dataset(directory_path: str, image_size: tuple[int, int]) -> ImageDataset:
    paths = [os.path.join(directory_path, img) for img in os.listdir(directory_path) if img.endswith(("png", "jpg", "jpeg"))]

    transform = transforms.Compose([
        # In case some image has different size
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Normalization
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return ImageDataset(paths, transform, rgb=False)


def create_evaluator(directory_path: str,
                     image_size: tuple[int, int],
                     name: str ="evaluator",
                     img_channels: int =1,
                     learning_rate: float =0.001,
                     end_loss: float =0.1,
                     device: torch.device =torch.device("cpu")):
    
    BATCH_SIZE = 256
    dataset = load_dataset(directory_path, image_size)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    evaluator = Evaluator(image_size, img_channels, device)
    evaluator.train_model(loader, learning_rate, end_loss, 250)
    evaluator.save_model(name)

def main():
    directory = "dataset/numbers/0"
    image_size = (80, 80)
    img_channels = 1
    learning_rate = 0.01
    end_loss = 0.005
    evaluator_name = "evaluator"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Not existing evaluator model
    if not os.path.isfile(f"models/saves/{evaluator_name}.pth"):
        print("Evaluator model was not found. Creating new evaluator model.")
        create_evaluator(directory, image_size, evaluator_name, img_channels, learning_rate, end_loss, device)


    generator = ImageGenerator(image_size, evaluator_name, img_channels, device)
    generator.train(epochs=100, learning_rate=0.001)
    image = generator()

    plt.imshow(image[0][0].cpu().detach().numpy(), cmap="gray")
    plt.show()

    # epochs = [1, 10, 20, 50, 100, 200, 250, 500]
    # images = []
    # for epoch in epochs:
    #     generator = ImageGenerator(image_size, evaluator_name, img_channels, device)
    #     generator.train(epochs=epoch, learning_rate=0.001)
    #     image = generator()
    #     images.append(image)

    # for i in range(len(images)):
    #     plt.imshow(images[i][0][0].cpu().detach().numpy(), cmap="gray")
    #     plt.title(f"Epoch: {epochs[i]}")
    #     plt.show()







if __name__ == "__main__":
    main()