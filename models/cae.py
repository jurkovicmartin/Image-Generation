import torch
import torch.nn as nn
import torch.nn.init as init

class CAE(nn.Module):
    # CONVOLUTIONAL AUTOENCODER
    def __init__(self, img_channels: int =3):
        super(CAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),             
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),               
            nn.LeakyReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, img_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  
        )

        # Initialize weights
        self.encoder.apply(self._initialize_weights)
        self.decoder.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
