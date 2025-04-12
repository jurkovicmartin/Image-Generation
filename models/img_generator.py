import torch

from models.evaluator import Evaluator
from models.generator import Generator


class ImageGenerator:
    def __init__(self,
                 image_size: tuple[int, int],
                 evaluator_name: str ="evaluator",
                 img_channels: int =1,
                 device: torch.device =torch.device("cpu")):
        
        self.LATENT_DIM = 128
        self.image_size = image_size
        self.img_channels = img_channels
        self.device = device

        self.generator = Generator(self.image_size, self.img_channels, self.LATENT_DIM)
        self.generator.to(self.device)
        self.evaluator = Evaluator(self.image_size, self.img_channels, self.device, load_path=f"models/saves/{evaluator_name}.pth")


    def __call__(self) -> torch.Tensor:
        print("Generating Image")

        with torch.no_grad():
            self.generator.eval()
            noise = torch.randn(1, self.LATENT_DIM).to(self.device)
            image = self.generator(noise)
            return image
    

    def train(self, epochs: int =100, learning_rate: float =0.001):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.generator.train()

        print("Training Generator")

        for epoch in range(epochs):
            noise = torch.randn(1, self.LATENT_DIM).to(self.device)
            image = self.generator(noise)
            loss = self.evaluator.evaluate(image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
