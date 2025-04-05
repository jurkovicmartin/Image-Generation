import torch



def add_noise(image: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
    noise = torch.randn_like(image) * intensity
    return image + noise