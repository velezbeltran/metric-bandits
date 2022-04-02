"""
A simple class to keep thing uniformly formatted.
"""

from torchvision import datasets, transforms


from metric_bandits.constants.paths import MNIST_PATH

MNIST = datasets.MNIST(
    MNIST_PATH,
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
