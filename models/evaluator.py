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
        """
        Evaluator class that determines if a given image is similar to a training image or not.

        Args:
            images_directory (str): Path to the directory containing the training images.
            image_size (tuple[int, int]): Size of the images to be evaluated.
            img_channels (int, optional): Number of color channels in the images. Defaults to 1 (=grayscale).
            device (torch.device, optional): Device to be used for evaluation. Defaults to torch.device("cpu").
            load_path (str, optional): Path to a saved model. If given, the model will be loaded from the path. Defaults to None.

        Raises:
            ValueError: If image size is not a square or is not a multiple of 8.
            FileNotFoundError: If a path is given but the file does not exist.
        """
        # Checking image size
        if image_size[0] != image_size[1]:
            raise ValueError("Image size must be a square")
        # Multiple of 8 to suit CAE model topology
        if image_size[0] % 8 != 0:
            raise ValueError("Image size must be a multiple of 8")
        
        self.image_size = image_size
        self.img_channels = img_channels
        self.device = device

        self.model = CAE(self.img_channels)

        # Loading existing model
        if load_path and os.path.isfile(load_path):
            self.model.load_state_dict(torch.load(load_path))
            print(f"Evaluator loaded to {device} device")
        # Provided path to a model which does not exist
        elif load_path:
            raise FileNotFoundError(f"File {load_path} does not exist")
        
        self.model.to(self.device)


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


    def evaluate(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the given image by calculating the mean squared error (MSE) between the output of the evaluating model and the provided image.

        Args:
            image (torch.Tensor): The image to be evaluated.

        Returns:
            torch.Tensor: The mean squared error between the output and the provided image.
        """
        self.model.eval()

        ref_output = self.model(reference)
        output = self.model(image)
        return F.mse_loss(output, ref_output, reduction="mean")


    def train_model(self,
                    dataloader: torch.utils.data.DataLoader,
                    learning_rate: float =0.001,
                    end_loss: float =0.1,
                    max_epochs: int =100):
        """
        Trains the Evaluator model on the given DataLoader.

        Args:
            learning_rate (float, optional): Learning rate for the training. Defaults to 0.001.
            end_loss (float, optional): Target loss for the training. Defaults to 0.1.
            max_epochs (int, optional): Maximum number of epochs for the training. Defaults to 100.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss(reduction="mean")
        # Lowering learning rate over time
        LR_STEPS = 100
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEPS, gamma=0.1)

        print("Training Evaluator")
        self.model.train()

        running_loss = float("inf")
        epoch = 1

        # Run until the loss isn't below target loss OR until the maximum number of epochs is reached
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
        """
        Saves the model to a file in the models/saves directory.

        Args:
            name (str, optional): Name of the model to be saved. Defaults to "evaluator".
        """
        path = f"models/saves/{name}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
