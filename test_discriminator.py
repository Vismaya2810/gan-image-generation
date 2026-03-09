import torch
from models.discriminator import Discriminator

D = Discriminator()

# create fake image batch
images = torch.randn(1, 3, 64, 64)

output = D(images)

print(output.shape)
print(output)