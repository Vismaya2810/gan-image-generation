import torch
from models.generator import Generator

G = Generator()

noise = torch.randn(1, 100, 1, 1)

fake_image = G(noise)

print(fake_image.shape)