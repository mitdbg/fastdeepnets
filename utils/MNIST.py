from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch

##
# Elasti transformation code
# Taken from https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
##
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)
########

def data_augment(image):
    image = np.array(image)
    image = elastic_transform(image, 8, 2.2)
    return Image.fromarray(image)

MNIST_TRANSFORM = [
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]

def get_dl(dataset, train=True, augment=False, bs=128):
    t = MNIST_TRANSFORM[:]
    if augment:
        t.insert(0, data_augment)
    print(t)

    ds = dataset(
            './datasets/%s/' % dataset.__name__,
            train=train,
            download=True,
            transform=transforms.Compose(t))
    return DataLoader(
        ds,
        batch_size=bs if train else len(ds),
        num_workers=40,
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )
