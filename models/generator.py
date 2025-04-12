import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,  image_size: tuple[int, int], img_channels: int =1, latent_dim: int =128):
        super(Generator, self).__init__()
        # Reduce resolution
        self.reduced_size = image_size[0] // 4

        # Fully connected layers (generates features)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128 * self.reduced_size ** 2),
            nn.LeakyReLU(),
        )

        # Convolutional layers (reconstructs image)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z) -> torch.Tensor:
        out = self.fc(z)
        # Reshape
        out = out.view(-1, 128, self.reduced_size, self.reduced_size)
        return self.conv(out)
