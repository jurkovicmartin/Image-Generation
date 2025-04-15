import torch
import torch.nn as nn
import torch.nn.init as init

class CAE(nn.Module):
    # CONVOLUTIONAL AUTOENCODER
    def __init__(self, img_channels: int =3):
        """
        Initializes the Convolutional Autoencoder (CAE) model.

        Args:
            img_channels (int, optional): The number of color channels in the input image. Defaults to 3 (=RGB).
        """
        super(CAE, self).__init__()

        # Encoder (feature extractor)
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),             
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),               
            nn.LeakyReLU()
        )

        # Decoder (reconstructor)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, img_channels, 3, stride=2, padding=1, output_padding=1),
            # 0 - 1 output
            nn.Sigmoid()  
        )

        # Kaiming weights initialization
        self.encoder.apply(self._initialize_weights)
        self.decoder.apply(self._initialize_weights)


    def _initialize_weights(self, layer: nn.Module):
        """
        Initializes the weights of the Convolutional Autoencoder (CAE) model.

        Applies Kaiming weight initialization to convolutional and transposed convolutional layers.

        Args:
            layer (nn.Module): The module to be initialized.
        """
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Convolutional Autoencoder (CAE) model.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Reconstructed image.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
