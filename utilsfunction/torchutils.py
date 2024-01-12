import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def save_tensor_images(image_tensor, size=(1, 28, 28), save_loc=""):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image1 = image_unflat[0]
    save_image(image1, save_loc)


def Mnist_dataLoader(root, save, batch_size, transform):
    download=False
    if not os.path.exists(save):
        os.mkdir(save)

    if not os.path.exists(root):
        os.mkdir(root)
        download = True

    dataloader = DataLoader(
        MNIST(root, download=download, transform=transform),
        batch_size=batch_size,
        shuffle=True)

    return dataloader


def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)


def combine_vectors(x, y, idx=1):
    return torch.cat([x, y], idx)


def get_one_hot_labels(labels, n_classes):
    return  torch.nn.functional.one_hot(labels,n_classes)