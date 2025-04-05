import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from autoencoder import ConvAutoencoder
from transforms import *

class Evaluator:
    def __init__(self, image_size: tuple[int, int], input_channels=1, device: torch.device =torch.device("cpu"), load_path: str =None):
        if image_size[0] != image_size[1]:
            raise ValueError("Image size must be a square")
        if image_size[0] % 8 != 0:
            raise ValueError("Image size must be a multiple of 8")
        
        self.image_size = image_size
        self.input_channels = input_channels
        self.device = device

        self.model = ConvAutoencoder(input_channels=self.input_channels)

        if load_path:
            self.model.load_state_dict(torch.load(load_path), map_location=self.device)
            print("Evaluator loaded")
        self.model.to(self.device)

        print(f"Evaluator is ready on {device} device")


    def __call__(self, image: torch.Tensor) -> float:
        self.model.eval()
        output = self.model(image)
        mse = F.mse_loss(output, image)
        return mse.item()


    def train_model(self, dataloader: torch.utils.data.DataLoader, epochs: int =100, learning_rate: float =0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        print("Training Evaluator")
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.model(batch)

                loss = criterion(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")


    def test_model(self, dataloader: torch.utils.data.DataLoader, scuff_probability: float =0.5):
        self.model.eval()
        changed_count = 0

        pass_count = 0
        deny_count = 0

        print("Testing Evaluator")
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)

                for image in batch:
                    if torch.rand(1) < scuff_probability:
                        image = add_noise(image, intensity=0.5)
                        changed_count += 1

                    output = self.model(image)
                    mse = F.mse_loss(output, image)
                    if mse.item() < 0.25:
                        pass_count += 1
                    else:
                        deny_count += 1


        print(f"Scuffed images: {changed_count}\nPassed images: {pass_count}\nDenied images: {deny_count}")


    def save_model(self, name: str ="evaluator"):
        torch.save(self.model.state_dict(), f"saves/{name}.pth")
        print("Model saved")
