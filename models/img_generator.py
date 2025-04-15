import torch
import os
import matplotlib.pyplot as plt

from models.evaluator import Evaluator
from models.generator import Generator


class ImageGenerator:
    def __init__(self,
                 images_loader: torch.utils.data.DataLoader,
                 image_size: tuple[int, int],
                 img_channels: int =1,
                 evaluator_name: str ="evaluator",
                 generator_name: str ="generator",
                 device: torch.device =torch.device("cpu")):
        """
        ImageGenerator class that combines Evaluating and Generating models.

        Args:
            images_loader (torch.utils.data.DataLoader): DataLoader containing the reference images.
            image_size (tuple[int, int]): Size of the images to be generated.
            img_channels (int, optional): Number of color channels in the images. Defaults to 1 (=grayscale).
            evaluator_name (str, optional): Name of the Evaluator model to be loaded. Defaults to "evaluator".
            generator_name (str, optional): Name of the Generator model to be loaded. If the generator doesn't exist, it will be created. Defaults to "generator".
            device (torch.device, optional): Device to be used for generation. Defaults to torch.device("cpu").
        """
        # Latent dimension for generator
        self.LATENT_DIM = 128
        self.images_loader = images_loader
        self.image_size = image_size
        self.img_channels = img_channels
        self.device = device

        self.evaluator = Evaluator(self.image_size, self.img_channels, self.device, load_path=f"models/saves/{evaluator_name}.pth")
        
        self.generator = Generator(self.image_size, self.img_channels, self.LATENT_DIM)
        # Load generator if exists
        self.generator_path = f"models/saves/{generator_name}.pth"
        self._load_generator()
        self.generator.to(self.device)


    def __call__(self) -> torch.Tensor:
        print("Generating Image")

        with torch.no_grad():
            self.generator.eval()
            noise = torch.randn(1, self.LATENT_DIM).to(self.device)
            image = self.generator(noise)
            return image
    

    def _load_generator(self):
        """
        Loads the Generator model from the given path if it exists.

        """        
        if self.generator_path and os.path.isfile(self.generator_path):
            self.generator.load_state_dict(torch.load(self.generator_path))
            print(f"Generator loaded to {self.device} device")
        else:
            print(f"Generator not found. Creating new generator.")

    def train(self, epochs: int =50, learning_rate: float =0.001):
        """
        Trains the Generator model.

        Args:
            epochs (int, optional): The number of epochs to train the model. Defaults to 100.
            learning_rate (float, optional): The learning rate for the training. Defaults to 0.001.
        """
        REFERENCE_CHANGE = 10
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)

        print("Training Generator")
        self.generator.train()

        reference = self._load_random_image().to(self.device)
        # Add dimensions to match the generated image dimensions
        reference = reference.unsqueeze(0).unsqueeze(0)

        for epoch in range(epochs):

            if epoch % REFERENCE_CHANGE == 0:
                print("New reference image")
                reference = self._load_random_image().unsqueeze(0).unsqueeze(0).to(self.device)
                # Show reference image
                # reference = self._load_random_image()
                # plt.imshow(reference.numpy(), cmap="gray")
                # plt.show()
                # reference = reference.unsqueeze(0).unsqueeze(0).to(self.device)

            noise = torch.randn(1, self.LATENT_DIM).to(self.device)
            image = self.generator(noise)
            loss = self.evaluator.evaluate(image, reference)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


    def _load_random_image(self) -> torch.Tensor:
        """
        Loads a random image from the images DataLoader.

        Returns:
            torch.Tensor: The loaded image.
        """
        num_batches = len(self.images_loader)
        batch_idx = torch.randint(0, num_batches, (1,)).item()

        for i, batch in enumerate(self.images_loader):
            if i == batch_idx:
                break
        
        images = batch[0]
        num_images = len(images)
        image_idx = torch.randint(0, num_images, (1,)).item()
        return images[image_idx]

    
    def save_generator(self):
        """
        Saves the Generator model to a file in the models/saves directory.

        Provided generator name (in constructor) will be used as save's name.
        """
        torch.save(self.generator.state_dict(), self.generator_path)
        print(f"Model saved to {self.generator_path}")