import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split

from models.MNIST_1h import MNIST_1h

transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
])

training_dataset = MNIST(
    './datasets/MNIST/',
    train=True,
    download=True,
    transform=transform
)
training_dataloader = DataLoader(
    training_dataset,
    batch_size=64,
    shuffle=True)

def init_models():
    return [MNIST_1h(int(2**(i / 2))) for i in range(24)]


def train(models):
    criterion = nn.CrossEntropyLoss()
    optimizers = [Adam(model.parameters()) for model in models]

    for e in range(0, 5):
        print("Epoch %s" % e)
        for i, (images, labels) in enumerate(training_dataloader):
            print(i)
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)
            for model, optimizer in zip(models, optimizers):
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


def get_activations(model):
    outputs = []
    for images, labels in training_dataloader:
        images = Variable(images, volatile=True)
        outputs.append(model.partial_forward(images))
    return torch.cat(outputs, 0).std(0).data.numpy()

