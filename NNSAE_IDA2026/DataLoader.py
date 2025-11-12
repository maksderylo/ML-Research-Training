from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
import torchvision
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import os
from torchvision.datasets import ImageFolder

def load_mnist_data(batch_size=256, normalize=True):
    """
    Loads MNIST data.

    Args:
        batch_size (int): The batch size for the data loaders.
        normalize (bool): If True, applies per-sample L2 normalization.
                          If False, returns raw [0,1] tensor data.
                          NMF requires non-normalized, non-negative data.
    """
    # Base transform to get tensor
    base_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Conditional preprocessing
    if normalize:
        def preprocess(x):
            x_flat = x.view(-1)  # Flatten from (1, 28, 28) to (784,)
            x_norm = x_flat / (torch.norm(x_flat) + 1e-8)  # Normalize to unit norm
            return x_norm

        transform = transforms.Compose([
            base_transform,
            transforms.Lambda(preprocess)
        ])
    else:
        # For NMF, just flatten the tensor
        transform = transforms.Compose([
            base_transform,
            transforms.Lambda(lambda x: x.view(-1))
        ])

    # Load datasets with the chosen transform
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_data(dataset_name, batch_size=128, img_size=64, **kwargs):
    """
    Unified data loading function for multiple datasets.
    """

    if dataset_name.lower() == 'mnist':
        return load_mnist_data(batch_size)

    # elif dataset_name.lower() == 'olivetti':
    #     train_split = kwargs.get('train_split', 0.8)
    #     return load_olivetti_data(batch_size, train_split)
    #
    # elif dataset_name.lower() == 'lfw':
    #     min_faces_per_person = kwargs.get('min_faces_per_person', 20)
    #     return load_lfw_data(batch_size, img_size, min_faces_per_person)
    #
    # elif dataset_name.lower() == 'imagenet':
    #     subset_size = kwargs.get('subset_size', 10000)
    #     data_root = kwargs.get('data_root', './data/imagenet')
    #     return load_imagenet_subset(batch_size, subset_size, img_size, data_root)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from ['mnist', 'olivetti', 'lfw', 'imagenet']")
