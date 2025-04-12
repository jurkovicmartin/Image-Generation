import torch
import torch.nn as nn

class LatentGenerator(nn.Module):
    def __init__(self, image_size=(160, 160), output_channels=3, latent_dim=100):
        super(LatentGenerator, self).__init__()
        self.image_size = image_size
        self.output_channels = output_channels
        self.latent_dim = latent_dim

        h, w = image_size
        assert h % 8 == 0 and w % 8 == 0, "Image dimensions should be divisible by 8"

        # Project and reshape latent vector into a small feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * (h // 8) * (w // 8)),
            nn.ReLU(True)
        )

        # Deconvolution layers to scale up to final image size
        # self.deconv = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (128, h/4, w/4)
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),

        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> (64, h/2, w/2)
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),

        #     nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),  # -> (C, h, w)
        #     nn.Tanh()  # Outputs in [-1, 1]
        # )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, output_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)
        h, w = self.image_size
        x = self.fc(z)
        x = x.view(batch_size, 256, h // 8, w // 8)
        # x = self.deconv(x)
        x = self.upsample(x)
        return x
