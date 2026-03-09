from data.dataloader import get_dataloader

loader = get_dataloader()

for images, _ in loader:
    print(images.shape)
    break