from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
import torchvision
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import os
from torchvision.datasets import ImageFolder

def load_mnist_data(batch_size=256):
    # First load raw data to compute mean
    raw_transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1] and creates tensor
    ])

    # Load training set to compute mean
    trainset_raw = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=raw_transform)

    # Compute mean over entire training set
    train_loader_temp = DataLoader(trainset_raw, batch_size=len(trainset_raw), shuffle=False)
    all_data = next(iter(train_loader_temp))[0]
    all_data = all_data.view(all_data.size(0), -1)  # Flatten to (N, 784)
    dataset_mean = all_data.mean(dim=0)  # Mean across samples, shape (784,)

    # Define preprocessing transform with mean subtraction and normalization
    def preprocess(x):
        x_flat = x.view(-1)  # Flatten from (1, 28, 28) to (784,)
        x_centered = x_flat - dataset_mean  # Subtract mean
        x_norm = x_centered / (torch.norm(x_centered) + 1e-8)  # Normalize to unit norm
        return x_norm

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(preprocess)
    ])

    # Load datasets with proper preprocessing
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset_mean

def load_olivetti_data(batch_size=32, train_split=0.8):
    """
    Load Olivetti Faces dataset (400 images, 64x64 grayscale)
    Returns data with shape (N, 4096) after flattening
    """
    # Download Olivetti Faces using sklearn
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    data = faces.data  # Already normalized to [0, 1], shape (400, 4096)

    # Convert to torch tensors
    data_tensor = torch.FloatTensor(data)  # Shape: (400, 4096)

    # Compute mean over entire dataset
    dataset_mean = data_tensor.mean(dim=0)  # Shape: (4096,)

    # Define preprocessing function
    def preprocess(x):
        x_centered = x - dataset_mean  # Subtract mean
        x_norm = x_centered / (torch.norm(x_centered) + 1e-8)  # Unit norm
        return x_norm

    # Apply preprocessing to all data
    preprocessed_data = torch.stack([preprocess(x) for x in data_tensor])

    # Create dataset (no labels needed for autoencoder)
    dataset = TensorDataset(preprocessed_data)

    # Split into train/test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset_mean

def load_lfw_data(batch_size=128, img_size=64, min_faces_per_person=20):
    """
    Load Labeled Faces in the Wild dataset with resizing
    """

    # Download LFW with original size
    lfw_people = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=1.0,
        color=False
    )

    print(f"Original LFW shape: {lfw_people.images.shape}")

    # Manually resize to exact dimensions
    resized_images = []
    for img in lfw_people.images:
        # Convert to PIL Image for proper resizing
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        # Resize to exact target size
        pil_img = pil_img.resize((img_size, img_size), Image.LANCZOS)
        # Back to normalized array
        resized = np.array(pil_img).astype(np.float32) / 255.0
        resized_images.append(resized.flatten())

    data_flat = np.array(resized_images)
    print(f"Resized LFW shape: {data_flat.shape}")  # Should be (n_samples, img_size²)

    # Convert to torch
    data_tensor = torch.FloatTensor(data_flat)

    # Compute mean
    dataset_mean = data_tensor.mean(dim=0)

    # Preprocess
    def preprocess(x):
        x_centered = x - dataset_mean
        x_norm = x_centered / (torch.norm(x_centered) + 1e-8)
        return x_norm

    preprocessed_data = torch.stack([preprocess(x) for x in data_tensor])

    # Create dataset
    class LFWDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return (self.data[idx],)

    dataset = LFWDataset(preprocessed_data)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"LFW Dataset loaded: {len(trainset)} train, {len(testset)} test")
    print(f"Image size: {img_size}×{img_size}, Input dimension: {img_size**2}")

    return train_loader, test_loader, dataset_mean

def load_imagenet_subset(batch_size=128, subset_size=50000, img_size=64,
                         data_root='./data/imagenet', use_color=True):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val_organized')

    # Raw transform for mean computation
    if use_color:
        raw_transform = transforms.Compose([
            transforms.Resize(img_size + 8),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        num_channels = 3
    else:
        raw_transform = transforms.Compose([
            transforms.Resize(img_size + 8),
            transforms.CenterCrop(img_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        num_channels = 1

    # Compute mean on subset
    print("Computing dataset mean...")
    trainset_raw = ImageFolder(train_dir, transform=raw_transform)
    mean_indices = torch.randperm(len(trainset_raw))[:10000].tolist()
    trainset_mean = Subset(trainset_raw, mean_indices)

    temp_loader = DataLoader(trainset_mean, batch_size=256,
                            num_workers=4, pin_memory=True)

    mean_accumulator = None
    count = 0
    for batch_data, _ in temp_loader:
        batch_flat = batch_data.view(batch_data.size(0), -1)
        if mean_accumulator is None:
            mean_accumulator = batch_flat.sum(dim=0)
        else:
            mean_accumulator += batch_flat.sum(dim=0)
        count += batch_data.size(0)

    dataset_mean = mean_accumulator / count
    print(f"✓ Mean computed over {count} images")

    # Preprocessing with mean subtraction
    def preprocess(x):
        x_flat = x.view(-1)
        x_centered = x_flat - dataset_mean
        norm = torch.norm(x_centered)
        return x_centered / (norm + 1e-8) if norm > 1e-8 else x_centered

    if use_color:
        final_transform = transforms.Compose([
            transforms.Resize(img_size + 8),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Lambda(preprocess)
        ])
    else:
        final_transform = transforms.Compose([
            transforms.Resize(img_size + 8),
            transforms.CenterCrop(img_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(preprocess)
        ])

    trainset = ImageFolder(train_dir, transform=final_transform)
    testset = ImageFolder(val_dir, transform=final_transform)

    if subset_size and subset_size < len(trainset):
        subset_indices = torch.randperm(len(trainset))[:subset_size].tolist()
        trainset = Subset(trainset, subset_indices)

    train_loader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"✓ Loaders ready: {len(train_loader)} train batches")
    return train_loader, test_loader, dataset_mean

def load_data(dataset_name, batch_size=128, img_size=64, **kwargs):
    """
    Unified data loading function for multiple datasets.
    """

    if dataset_name.lower() == 'mnist':
        return load_mnist_data(batch_size)

    elif dataset_name.lower() == 'olivetti':
        train_split = kwargs.get('train_split', 0.8)
        return load_olivetti_data(batch_size, train_split)

    elif dataset_name.lower() == 'lfw':
        min_faces_per_person = kwargs.get('min_faces_per_person', 20)
        return load_lfw_data(batch_size, img_size, min_faces_per_person)

    elif dataset_name.lower() == 'imagenet':
        subset_size = kwargs.get('subset_size', 10000)
        data_root = kwargs.get('data_root', './data/imagenet')
        return load_imagenet_subset(batch_size, subset_size, img_size, data_root)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from ['mnist', 'olivetti', 'lfw', 'imagenet']")
