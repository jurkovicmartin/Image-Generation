import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),  # -> (16, H/2, W/2)
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),               # -> (32, H/4, W/4)
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),               # -> (64, H/8, W/8)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # -> (32, H/4, W/4)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> (16, H/2, W/2)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, input_channels, 3, stride=2, padding=1, output_padding=1),  # -> (C, H, W)
            nn.Sigmoid()  # for normalized input in [0,1]
        )

    def forward(self, x) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
