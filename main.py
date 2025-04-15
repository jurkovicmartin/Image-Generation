import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from dataset import ImageDataset
from models.evaluator import Evaluator
from models.img_generator import ImageGenerator


def load_dataloader(directory_path: str, image_size: tuple[int, int], rgb: bool =False) -> torch.utils.data.DataLoader:
    """
    Loads a DataLoader containing images from a given directory path.

    Args:
        directory_path (str): Path to the directory containing the images.
        image_size (tuple[int, int]): Size of the images to be loaded.
        rgb (bool, optional): Whether to load images in RGB or grayscale. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: DataLoader containing the images.

    """
    BATCH_SIZE = 256

    # Loads all paths to files in the directory (png, jpg, jpeg)
    paths = [os.path.join(directory_path, img) for img in os.listdir(directory_path) if img.endswith(("png", "jpg", "jpeg"))]

    transform = transforms.Compose([
        # Uniform size
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Normalization
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = ImageDataset(paths, transform, rgb=rgb)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def create_evaluator(directory_path: str,
                     image_size: tuple[int, int],
                     name: str ="evaluator",
                     img_channels: int =1,
                     learning_rate: float =0.001,
                     end_loss: float =0.1,
                     device: torch.device =torch.device("cpu")):
    
    """
    Creates an Evaluator and trains it on the dataset from the given directory. Trained model is then saved (default models/saves/name.pth).

    Args:
        directory_path (str): Path to the directory containing the training images.
        image_size (tuple[int, int]): Size of the images to be loaded.
        name (str, optional): Name of the evaluator's model save. Defaults to "evaluator".
        img_channels (int, optional): Number of color channels in the images. Defaults to 1 (=grayscale).
        learning_rate (float, optional): Learning rate for the training. Defaults to 0.001.
        end_loss (float, optional): Target loss for the training. Defaults to 0.1.
        device (torch.device, optional): Device to be used for training. Defaults to torch.device("cpu").
    """
    MAX_EPOCHS = 250

    loader = load_dataloader(directory_path, image_size)

    evaluator = Evaluator(image_size, img_channels, device)
    evaluator.train_model(loader, learning_rate, end_loss, MAX_EPOCHS)
    evaluator.save_model(name)


def main():
    ### SETUP
    directory = "dataset/profiles_short"
    image_size = (80, 80)
    img_channels = 1
    learning_rate = 0.01
    end_loss = 0.005
    evaluator_name = "evaluator_prof"
    generator_name = "generator_prof"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### LOADING AND TRAINING THE GENERATIVE SYSTEM
    # Not existing evaluator model = creating a new one
    if not os.path.isfile(f"models/saves/{evaluator_name}.pth"):
        print("Evaluator model was not found. Creating new evaluator model.")
        create_evaluator(directory, image_size, evaluator_name, img_channels, learning_rate, end_loss, device)

    loader = load_dataloader(directory, image_size, rgb=False)


    generator = ImageGenerator(loader, image_size, img_channels, evaluator_name, generator_name,device)
    generator.train(epochs=50, learning_rate=0.001)
    # generator.save_generator()
    image = generator()

    plt.imshow(image[0][0].cpu().detach().numpy(), cmap="gray")
    plt.show()

    ### SEEING GENERATED IMAGES EVOLUTION WITH DIFFERENT EPOCHS

    # epochs = [1, 10, 20, 50, 100, 200, 250, 500]
    # images = []
    # for epoch in epochs:
    #     generator = ImageGenerator(loader, image_size, evaluator_name, img_channels, device)
    #     generator.train(epochs=epoch, learning_rate=0.001)
    #     image = generator()
    #     images.append(image)

    # for i in range(len(images)):
    #     plt.imshow(images[i][0][0].cpu().detach().numpy(), cmap="gray")
    #     plt.title(f"Epoch: {epochs[i]}")
    #     plt.show()



if __name__ == "__main__":
    main()