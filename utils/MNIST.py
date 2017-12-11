from torch.utils.data import DataLoader
from torchvision import transforms
import torch

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_dl(dataset, train=True, bs=128):
    ds = dataset(
            './datasets/%s/' % dataset.__name__,
            train=train,
            download=True,
            transform=MNIST_TRANSFORM)
    return DataLoader(
        ds,
        batch_size=bs if train else len(ds),
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )
