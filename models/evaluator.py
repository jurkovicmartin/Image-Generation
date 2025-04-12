import torch
import torch.nn.functional as F
import os

from models.cae import CAE

class Evaluator:
    def __init__(self,
                 image_size: tuple[int, int],
                 img_channels :int =1,
                 device: torch.device =torch.device("cpu"),
                 load_path: str =None):
        
        if image_size[0] != image_size[1]:
            raise ValueError("Image size must be a square")
        if image_size[0] % 8 != 0:
            raise ValueError("Image size must be a multiple of 8")
        
        self.image_size = image_size
        self.img_channels = img_channels
        self.device = device

        self.model = CAE(self.img_channels)

        # Loading existing model
        if load_path and os.path.isfile(load_path):
            self.model.load_state_dict(torch.load(load_path))
            print(f"Evaluator loaded on {device} device")
        # Provided path which does not exist
        elif load_path:
            raise FileNotFoundError(f"File {load_path} does not exist")
        
        self.model.to(self.device)


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


    def evaluate(self, image: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        output = self.model(image)
        mse = F.mse_loss(output, image, reduction="mean")
        return mse


    def train_model(self,
                    dataloader: torch.utils.data.DataLoader,
                    learning_rate: float =0.001,
                    end_loss: float =0.1,
                    max_epochs: int =100):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss(reduction="mean")
        LR_STEPS = 100
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEPS, gamma=0.1)

        print("Training Evaluator")
        self.model.train()

        running_loss = float("inf")
        epoch = 1

        while running_loss / len(dataloader) > end_loss and epoch < max_epochs:
            running_loss = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)

                output = self.model(batch)
                loss = criterion(output, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            lr_scheduler.step()

            if epoch % LR_STEPS == 0:
                print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]["lr"]}")

            print(f"Epoch {epoch}, Loss: {running_loss / len(dataloader)}")
            epoch += 1
        
        print(f"Evaluator training ends on epoch {epoch} with loss {running_loss / len(dataloader)}")
        

    def save_model(self, name: str ="evaluator"):
        torch.save(self.model.state_dict(), f"models/saves/{name}.pth")
        print("Model saved")
