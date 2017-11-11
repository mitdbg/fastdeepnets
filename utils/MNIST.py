from torch.utils.data import DataLoader
from torchvision import transforms
import torch

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_dl(dataset, train=True):
    return DataLoader(
        dataset(
            './datasets/%s/' % dataset.__name__,
            train=train,
            download=True,
            transform=MNIST_TRANSFORM),
        batch_size=128,
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )
