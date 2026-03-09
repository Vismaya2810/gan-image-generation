import torchvision.utils as vutils
import os


def save_generated_images(images, epoch):

    os.makedirs("results", exist_ok=True)

    file_path = f"results/epoch_{epoch}.png"

    vutils.save_image(
        images,
        file_path,
        normalize=True
    )