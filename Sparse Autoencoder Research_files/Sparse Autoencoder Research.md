# SAE implementations


```python
import torch.nn as nn
import torch
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, k_top=20):
        super(SparseAutoencoder, self).__init__()
        self.training = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k_top = k_top
        self.name = "Default Sparse Autoencoder"

        # Encoder maps input to hidden representation
        self.encoder = nn.Linear(input_size, hidden_size)

        # Decoder maps hidden representation back to input space
        self.decoder = nn.Linear(hidden_size, input_size)

    def _topk_mask(self, activations: torch.Tensor) -> torch.Tensor:
        # activations: (batch, hidden)
        k = max(0, min(self.k_top, activations.size(1)))
        _, idx = torch.topk(activations, k, dim=1)
        mask = torch.zeros_like(activations)
        mask.scatter_(1, idx, 1.0)
        return mask

    def forward(self, x):
        pre_activations = self.encoder(x)
        pre_activations = F.relu(pre_activations)
        mask = self._topk_mask(pre_activations)
        h = pre_activations * mask
        x_hat = self.decoder(h)
        return h, x_hat


    def compute_loss(self, x, h, x_hat):
        # We compute sum of squares and normalize by batch size
        recon_loss = torch.sum((x - x_hat) ** 2) / (x.size(0))

        return recon_loss
```


```python

class SparseAutoencoderInit(SparseAutoencoder):
    def __init__(self, input_size=784, hidden_size=64, k_top=20):
        super(SparseAutoencoderInit, self).__init__(input_size, hidden_size, k_top)

        self.name = "Sparse Autoencoder with just weight initialization"
        # Initialize encoder weights first with random directions
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        # Initialize the decoder to be the transpose of the encoder weights
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

```


```python

class SparseAutoencoderJumpReLU(SparseAutoencoder):
    def __init__(self, input_size=784, hidden_size=64, k_top=20, jump_value=0.1):
        super(SparseAutoencoderJumpReLU, self).__init__(input_size, hidden_size, k_top)
        self.name = "Sparse Autoencoder with Jump ReLU"
        self.jump_value = jump_value

    def forward(self, x: torch.Tensor):
        h_raw = self.encoder(x)
        mask = self._topk_mask(h_raw)
        h = h_raw * mask
        # Apply JumpReLU
        h = torch.where(h > self.jump_value, h, torch.zeros_like(h))
        x_hat = self.decoder(h)
        return h, x_hat
```


```python

class SparseAutoencoderInitJumpReLU(SparseAutoencoder):
    def __init__(self, input_size=784, hidden_size=64, k_top=20, jump_value=0.1):
        super(SparseAutoencoderInitJumpReLU, self).__init__(input_size, hidden_size, k_top)
        self.name = "Sparse Autoencoder with Initialization and Jump ReLU"
        self.jump_value = jump_value

        # Initialize encoder weights first with random directions
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        # Initialize the decoder to be the transpose of the encoder weights
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())


    def forward(self, x: torch.Tensor):
        h_raw = self.encoder(x)
        mask = self._topk_mask(h_raw)
        h = h_raw * mask
        # Apply JumpReLU
        h = torch.where(h > self.jump_value, h, torch.zeros_like(h))
        x_hat = self.decoder(h)
        return h, x_hat
```

Implementing auxiliary loss SAE


```python

class SparseAutoencoderAuxLoss(SparseAutoencoder):
    def __init__(self, input_size, hidden_size, k_top, k_aux, k_aux_param, dead_feature_threshold):
        super(SparseAutoencoderAuxLoss, self).__init__(input_size, hidden_size, k_top)
        self.name = "Sparse Autoencoder with Auxiliary Loss"
        # k_aux is typically 2*k or more to revive dead features
        self.k_aux = k_aux if k_aux is not None else 2 * k_top
        self.k_aux_param = k_aux_param
        # Track dead features: count steps since each feature was last active
        self.register_buffer('steps_since_active', torch.zeros(hidden_size))
        self.dead_feature_threshold = dead_feature_threshold

    # Function to track which features are dead
    def _update_dead_features(self, h: torch.Tensor):
        # Feature is active if ANY sample in batch activates it
        active_mask = (h.abs() > 1e-8).any(dim=0)

        # Increment counter for inactive features, reset for active ones
        self.steps_since_active += 1
        self.steps_since_active[active_mask] = 0

    def _get_dead_feature_mask(self) -> torch.Tensor:
        """Return boolean mask of dead features"""
        return self.steps_since_active > self.dead_feature_threshold

    def forward(self, x: torch.Tensor):
        h_raw = self.encoder(x)
        mask = self._topk_mask(h_raw)
        h = h_raw * mask
        x_hat = self.decoder(h)

        # Track dead features during training
        if self.training:
            self._update_dead_features(h)

        return h, x_hat

    def compute_loss(self, x, h, x_hat):
        # Main reconstruction loss
        recon_error = torch.sum((x - x_hat) ** 2)
        recon_loss = recon_error / self.input_size

        # Auxiliary loss using dead features only
        aux_loss = torch.tensor(0.0, device=x.device)

        if self.training:
            dead_mask = self._get_dead_feature_mask()  # (hidden_size,)
            n_dead = dead_mask.sum().item()

            if n_dead > 0:
                # Compute reconstruction error: e = x - x_hat
                recon_error_vec = x - x_hat  # (batch, input_size)

                # Get raw activations again (before TopK masking)
                with torch.no_grad():
                    h_raw = self.encoder(x)

                # Select only dead features
                h_dead = h_raw * dead_mask.float().unsqueeze(0)  # (batch, hidden_size)

                # Select top-k_aux dead features
                k_aux_features = min(self.k_aux, n_dead)
                _, idx_aux = torch.topk(h_dead, k_aux_features, dim=1)
                mask_aux = torch.zeros_like(h_dead)
                mask_aux.scatter_(1, idx_aux, 1.0)

                # Sparse activations using only dead features
                z_aux = h_raw * mask_aux  # (batch, hidden_size)

                # Reconstruct error using dead features
                e_hat = self.decoder(z_aux)  # (batch, input_size)

                # Auxiliary loss: ||e - e_hat||^2
                aux_loss = torch.sum((recon_error_vec - e_hat) ** 2) / self.input_size

        # Total loss
        total_loss = recon_loss + self.k_aux_param * aux_loss

        return total_loss, recon_loss, aux_loss
```

Complete with relu, init and aux loss implementation.


```python

class SparseAutoencoderComplete(SparseAutoencoder):
    def __init__(self, input_size, hidden_size, k_top, k_aux, k_aux_param, dead_feature_threshold, jump_value):
        super(SparseAutoencoderComplete, self).__init__(input_size, hidden_size, k_top)
        self.name = "Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss"
        self.jump_value = jump_value

        # k_aux is typically 2*k or more to revive dead features
        self.k_aux = k_aux if k_aux is not None else 2 * k_top
        self.k_aux_param = k_aux_param
        # Track dead features: count steps since each feature was last active
        self.register_buffer('steps_since_active', torch.zeros(hidden_size))
        self.dead_feature_threshold = dead_feature_threshold

        # Initialize encoder weights first with random directions
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        # Initialize the decoder to be the transpose of the encoder weights
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

    # Function to track which features are dead
    def _update_dead_features(self, h: torch.Tensor):
        # Feature is active if ANY sample in batch activates it
        active_mask = (h.abs() > 1e-8).any(dim=0)

        # Increment counter for inactive features, reset for active ones
        self.steps_since_active += 1
        self.steps_since_active[active_mask] = 0

    def _get_dead_feature_mask(self) -> torch.Tensor:
        """Return boolean mask of dead features"""
        return self.steps_since_active > self.dead_feature_threshold

    def forward(self, x: torch.Tensor):
        pre_activations = self.encoder(x)
        pre_activations = F.relu(pre_activations)
        mask = self._topk_mask(pre_activations)
        h = pre_activations * mask
        x_hat = self.decoder(h)

        # Track dead features during training
        if self.training:
            self._update_dead_features(h)

        return h, x_hat

    def compute_loss(self, x, h, x_hat):
        # Main reconstruction loss
        recon_error = torch.sum((x - x_hat) ** 2)
        recon_loss = recon_error / self.input_size

        # Auxiliary loss using dead features only
        aux_loss = torch.tensor(0.0, device=x.device)

        if self.training:
            dead_mask = self._get_dead_feature_mask()  # (hidden_size,)
            n_dead = dead_mask.sum().item()

            if n_dead > 0:
                # Compute reconstruction error: e = x - x_hat
                recon_error_vec = x - x_hat  # (batch, input_size)

                # Get raw activations again (before TopK masking)
                with torch.no_grad():
                    h_raw = self.encoder(x)

                # Select only dead features
                h_dead = h_raw * dead_mask.float().unsqueeze(0)  # (batch, hidden_size)

                # Select top-k_aux dead features
                k_aux_features = min(self.k_aux, n_dead)
                _, idx_aux = torch.topk(h_dead, k_aux_features, dim=1)
                mask_aux = torch.zeros_like(h_dead)
                mask_aux.scatter_(1, idx_aux, 1.0)

                # Sparse activations using only dead features
                z_aux = h_raw * mask_aux  # (batch, hidden_size)

                # Reconstruct error using dead features
                e_hat = self.decoder(z_aux)  # (batch, input_size)

                # Auxiliary loss: ||e - e_hat||^2
                aux_loss = torch.sum((recon_error_vec - e_hat) ** 2) / self.input_size

        # Total loss
        total_loss = recon_loss + self.k_aux_param * aux_loss

        return total_loss, recon_loss, aux_loss
```

# Data Loading and Preprocessing


```python
from torch import optim
import torchvision
from torch.utils.data import TensorDataset, Subset
from sklearn.datasets import fetch_olivetti_faces
import torchvision.transforms as transforms

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
```


```python

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset_mean
```


```python

def load_olivetti_data(batch_size=32, train_split=0.8):
    """
    Load Olivetti Faces dataset (400 images, 64x64 grayscale)
    Returns data with shape (N, 4096) after flattening
    """
    # Download Olivetti Faces using sklearn
```


```python
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
```


```python
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

def load_imagenet_subset(batch_size=128, subset_size=50000, img_size=64,
                         data_root='./data/imagenet'):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val_organized')

    # Raw transform for mean computation
    raw_transform = transforms.Compose([
        transforms.Resize(img_size + 8),
        transforms.CenterCrop(img_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

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

    final_transform = transforms.Compose([
        transforms.Resize(img_size + 8),
        transforms.CenterCrop(img_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(preprocess)
    ])

    # Load datasets
    trainset = ImageFolder(train_dir, transform=final_transform)
    testset = ImageFolder(val_dir, transform=final_transform)

    # Subset training data
    if subset_size and subset_size < len(trainset):
        subset_indices = torch.randperm(len(trainset))[:subset_size].tolist()
        trainset = Subset(trainset, subset_indices)

    train_loader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"✓ Loaders ready: {len(train_loader)} train batches")
    return train_loader, test_loader, dataset_mean

```


```python
from torch.utils.data import Dataset

def load_lfw_data(batch_size=128, img_size=64, min_faces_per_person=20):
    """
    Load Labeled Faces in the Wild dataset with proper resizing

    Args:
        batch_size: Batch size
        img_size: Resize to (img_size, img_size) - actual pixels
        min_faces_per_person: Filter people with fewer images
    """
    # Download LFW with original size
    lfw_people = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=1.0,  # Keep original size, we'll resize manually
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
```


```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from PIL import Image

def load_data(dataset_name, batch_size=128, img_size=64, **kwargs):
    """
    Unified data loading function for multiple datasets.

    Args:
        dataset_name: One of ['mnist', 'olivetti', 'lfw', 'imagenet']
        batch_size: Batch size for DataLoader
        img_size: Image size for face datasets (default 64)
        **kwargs: Additional dataset-specific arguments

    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        dataset_mean: Mean vector used for preprocessing
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

```

# Training function


```python
#
def train_sparse_autoencoder(train_loader, num_epochs=50, learning_rate=0.001,
                            input_size=784, hidden_size=64, k_top=20,
                            JumpReLU=0.1, k_aux=None, k_aux_param=1/32,
                            dead_feature_threshold=1000, modelType="SAE",
                            dataset_type="mnist"):
    """
    Train sparse autoencoder with support for different datasets

    Args:
        train_loader: DataLoader for training data
        dataset_type: 'mnist', 'olivetti', or 'imagenet' to handle different unpacking
        ... (other args as before)
    """
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if modelType == "SAE":
        model = SparseAutoencoder(input_size=input_size, hidden_size=hidden_size, k_top=k_top).to(device)
    elif modelType == "SAE_Init_JumpReLU":
        model = SparseAutoencoderInitJumpReLU(input_size=input_size, hidden_size=hidden_size, k_top=k_top, jump_value=JumpReLU).to(device)
    elif modelType == "SAE_JumpReLU":
        model = SparseAutoencoderJumpReLU(input_size=input_size, hidden_size=hidden_size, k_top=k_top, jump_value=JumpReLU).to(device)
    elif modelType == "SAE_Init":
        model = SparseAutoencoderInit(input_size=input_size, hidden_size=hidden_size, k_top=k_top).to(device)
    elif modelType == "SAE_AuxLoss":
        model = SparseAutoencoderAuxLoss(input_size=input_size, hidden_size=hidden_size, k_top=k_top, k_aux=k_aux,
                                         k_aux_param=k_aux_param, dead_feature_threshold=dead_feature_threshold).to(device)
    elif modelType == "Complete":
        model = SparseAutoencoderComplete(input_size=input_size, hidden_size=hidden_size, k_top=k_top, k_aux=k_aux,
                                         k_aux_param=k_aux_param, dead_feature_threshold=dead_feature_threshold, jump_value=JumpReLU).to(device)
    else:
        raise ValueError("Invalid modelType specified.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            # Handle different data loader formats
            if dataset_type in ['olivetti', 'lfw']:
                # Olivetti returns single-element tuple: (inputs,)
                inputs, = data  # Note the comma - unpacks single element
                inputs = inputs.to(device)
            elif dataset_type in ['mnist', 'imagenet']:
                # MNIST and ImageNet return (inputs, labels)
                inputs, _ = data
                # No need to reshape - already preprocessed to correct shape
                inputs = inputs.to(device)
            else:
                raise ValueError(f"Unknown dataset_type: {dataset_type}")

            optimizer.zero_grad()
            h, outputs = model(inputs)

            if modelType == "SAE_AuxLoss" or modelType == "Complete":
                loss, mse_loss, aux_loss = model.compute_loss(inputs, h, outputs)
            else:
                loss = model.compute_loss(inputs, h, outputs)

            loss.backward()
            optimizer.step()

            # Clamp weights to enforce non-negativity
            with torch.no_grad():
                model.decoder.weight.clamp_(0.0)

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    print('Finished Training')
    return model
```

# Visualization functions


```python
import matplotlib.pyplot as plt
import math
import numpy as np

def visualize_weights_decoder(model, num_features=64):
    """
    Visualize decoder weights - AUTO-DETECTS dimensions
    """
    # Auto-detect input size from model
    input_size = model.decoder.weight.shape[0]
    print(f"Auto-detected input_size: {input_size}")

    # Determine image shape
    if input_size == 784:
        img_shape = (28, 28)
        dataset_type = 'mnist'
    elif input_size == 4096:
        img_shape = (64, 64)
    else:
        # Non-square or unusual size - try square root
        side = int(np.sqrt(input_size))
        if side * side == input_size:
            img_shape = (side, side)
        else:
            # Non-square - find factors
            for h in range(int(np.sqrt(input_size)), 0, -1):
                if input_size % h == 0:
                    w = input_size // h
                    img_shape = (h, w)
                    break

    print(f"Using image shape: {img_shape}")

    weights = model.decoder.weight.data.cpu().numpy().T
    num_features = min(num_features, weights.shape[0])

    # Grid dimensions
    x_images = int(math.ceil(math.sqrt(num_features)))
    y_images = int(math.ceil(num_features / x_images))

    plt.figure(figsize=(x_images * 2, y_images * 2))
    model_name = getattr(model, 'name', 'SAE')
    plt.suptitle(f'{model_name} Decoder Weights ({img_shape[0]}×{img_shape[1]})',
                 fontsize=14, y=0.995)

    for i in range(num_features):
        plt.subplot(y_images, x_images, i + 1)
        weight_img = weights[i].reshape(img_shape)

        # Normalize
        weight_img = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min() + 1e-8)

        plt.imshow(weight_img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title(f'F{i}', fontsize=8)

    plt.tight_layout()
    plt.show()


def visualize_weights_encoder(model, num_features=64):
    """
    Visualize encoder weights - AUTO-DETECTS dimensions
    """
    # Get weights
    if hasattr(model.encoder, 'weight'):
        weights = model.encoder.weight.data.cpu().numpy()
    elif isinstance(model.encoder, torch.nn.Sequential):
        weights = model.encoder[0].weight.data.cpu().numpy()
    else:
        raise ValueError("Unknown encoder structure")

    # Auto-detect input size
    input_size = weights.shape[1]
    print(f"Auto-detected input_size: {input_size}")

    # Determine image shape
    if input_size == 784:
        img_shape = (28, 28)
    elif input_size == 4096:
        img_shape = (64, 64)
    else:
        side = int(np.sqrt(input_size))
        if side * side == input_size:
            img_shape = (side, side)
        else:
            for h in range(int(np.sqrt(input_size)), 0, -1):
                if input_size % h == 0:
                    w = input_size // h
                    img_shape = (h, w)
                    break

    print(f"Using image shape: {img_shape}")

    num_features = min(num_features, weights.shape[0])

    x_images = int(math.ceil(math.sqrt(num_features)))
    y_images = int(math.ceil(num_features / x_images))

    plt.figure(figsize=(x_images * 2, y_images * 2))
    model_name = getattr(model, 'name', 'SAE')
    plt.suptitle(f'{model_name} Encoder Weights ({img_shape[0]}×{img_shape[1]})',
                 fontsize=14, y=0.995)

    for i in range(num_features):
        plt.subplot(y_images, x_images, i + 1)
        weight_img = weights[i].reshape(img_shape)
        weight_img = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min() + 1e-8)
        plt.imshow(weight_img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title(f'F{i}', fontsize=8)

    plt.tight_layout()
    plt.show()


def visualize_reconstructions(model, data_loader, num_samples=10, dataset_type='olivetti'):
    """
    Visualize reconstructions - AUTO-DETECTS dimensions
    """
    model.eval()
    device = next(model.parameters()).device

    # Get data
    data_iter = iter(data_loader)
    data = next(data_iter)

    if dataset_type == 'olivetti' or len(data) == 1:
        inputs, = data
    else:
        inputs, _ = data

    inputs = inputs[:num_samples].to(device)

    # Auto-detect dimensions
    input_size = inputs.shape[1]
    if input_size == 784:
        img_shape = (28, 28)
    elif input_size == 4096:
        img_shape = (64, 64)
    else:
        side = int(np.sqrt(input_size))
        if side * side == input_size:
            img_shape = (side, side)
        else:
            for h in range(int(np.sqrt(input_size)), 0, -1):
                if input_size % h == 0:
                    w = input_size // h
                    img_shape = (h, w)
                    break

    # Get reconstructions
    with torch.no_grad():
        _, reconstructions = model(inputs)

    inputs = inputs.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    model_name = getattr(model, 'name', 'SAE')
    plt.suptitle(f'{model_name} Reconstructions ({img_shape[0]}×{img_shape[1]})', fontsize=14)

    for i in range(num_samples):
        axes[0, i].imshow(inputs[i].reshape(img_shape), cmap='gray', interpolation='nearest')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        axes[1, i].imshow(reconstructions[i].reshape(img_shape), cmap='gray', interpolation='nearest')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    plt.show()
```

Functions to count dead neurons and test loss on the dataset given


```python

def count_dead_neurons(model, data_loader, dataset_type='mnist'):
    """
    Count dead neurons (features that never activate)

    Args:
        model: Trained SAE model
        data_loader: DataLoader with data
        dataset_type: 'mnist', 'olivetti', or 'imagenet' for proper unpacking

    Returns:
        num_dead: Number of dead neurons
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    dead_neurons = torch.ones(model.hidden_size, dtype=torch.bool).to(device)

    with torch.no_grad():
        for data in data_loader:
            # Handle different data loader formats
            if dataset_type in ['olivetti', 'lfw']:
                inputs, = data  # Single-element tuple
            else:  # mnist or imagenet
                inputs, _ = data  # (inputs, labels) tuple

            inputs = inputs.to(device)  # Already preprocessed, no reshape needed
            h, _ = model(inputs)

            # A neuron is alive if it activates (h > 0) for any sample
            dead_neurons &= (h.sum(dim=0) == 0)

    num_dead = dead_neurons.sum().item()
    model_name = getattr(model, 'name', 'SAE')
    print(f'Number of dead neurons in {model_name}: {num_dead} out of {model.hidden_size} '
          f'({100*num_dead/model.hidden_size:.2f}%)')
    return num_dead


def test_loss(model, data_loader, dataset_type='mnist'):
    """
    Compute average test loss

    Args:
        model: Trained SAE model
        data_loader: DataLoader with test data
        dataset_type: 'mnist', 'olivetti', or 'imagenet' for proper unpacking

    Returns:
        avg_loss: Average loss over test set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in data_loader:
            # Handle different data loader formats
            if dataset_type in ['olivetti', 'lfw']:
                inputs, = data  # Single-element tuple
            else:  # mnist or imagenet
                inputs, _ = data  # (inputs, labels) tuple

            inputs = inputs.to(device)  # Already preprocessed, no reshape needed
            h, outputs = model(inputs)

            # Handle different loss outputs
            loss_output = model.compute_loss(inputs, h, outputs)
            if isinstance(loss_output, tuple):
                loss, *_ = loss_output  # Unpack if tuple (e.g., with aux loss)
            else:
                loss = loss_output

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    model_name = getattr(model, 'name', 'SAE')
    print(f'Test Loss for {model_name}: {avg_loss:.6f}')
    return avg_loss


def get_activation_statistics(model, data_loader, dataset_type='mnist'):
    """
    Get comprehensive statistics about feature activations

    Args:
        model: Trained SAE model
        data_loader: DataLoader with data
        dataset_type: 'mnist', 'olivetti', or 'imagenet'

    Returns:
        stats: Dictionary with activation statistics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    activation_counts = torch.zeros(model.hidden_size).to(device)
    activation_sums = torch.zeros(model.hidden_size).to(device)
    total_samples = 0

    with torch.no_grad():
        for data in data_loader:
            # Handle different data loader formats
            if dataset_type in ['olivetti', 'lfw']:
                inputs, = data
            else:
                inputs, _ = data

            inputs = inputs.to(device)
            h, _ = model(inputs)

            # Count how many times each feature activates (h > 0)
            activation_counts += (h > 0).sum(dim=0).float()
            activation_sums += h.sum(dim=0)
            total_samples += inputs.size(0)

    # Move to CPU for analysis
    activation_counts = activation_counts.cpu().numpy()
    activation_sums = activation_sums.cpu().numpy()

    # Compute statistics
    activation_freq = activation_counts / total_samples  # Fraction of samples each feature activates on
    mean_activation = activation_sums / total_samples    # Average activation strength

    stats = {
        'total_features': model.hidden_size,
        'dead_features': np.sum(activation_counts == 0),
        'active_features': np.sum(activation_counts > 0),
        'mean_activation_frequency': np.mean(activation_freq),
        'median_activation_frequency': np.median(activation_freq),
        'mean_activation_strength': np.mean(mean_activation[activation_counts > 0]),  # Among active features
        'activation_frequencies': activation_freq,
        'activation_strengths': mean_activation
    }

    # Print summary
    model_name = getattr(model, 'name', 'SAE')
    print(f"\n=== Activation Statistics for {model_name} ===")
    print(f"Total features: {stats['total_features']}")
    print(f"Dead features: {stats['dead_features']} ({100*stats['dead_features']/stats['total_features']:.2f}%)")
    print(f"Active features: {stats['active_features']} ({100*stats['active_features']/stats['total_features']:.2f}%)")
    print(f"Mean activation frequency: {stats['mean_activation_frequency']:.4f}")
    print(f"Median activation frequency: {stats['median_activation_frequency']:.4f}")
    print(f"Mean activation strength (active features): {stats['mean_activation_strength']:.4f}")

    return stats


def plot_activation_histogram(model, data_loader, dataset_type='mnist'):
    """
    Plot histogram of feature activation frequencies

    Args:
        model: Trained SAE model
        data_loader: DataLoader with data
        dataset_type: 'mnist', 'olivetti', or 'imagenet'
    """
    stats = get_activation_statistics(model, data_loader, dataset_type)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_name = getattr(model, 'name', 'SAE')

    # Histogram of activation frequencies
    axes[0].hist(stats['activation_frequencies'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Activation Frequency (fraction of samples)')
    axes[0].set_ylabel('Number of Features')
    axes[0].set_title(f'{model_name}: Feature Activation Frequencies')
    axes[0].axvline(stats['mean_activation_frequency'], color='r', linestyle='--',
                    label=f'Mean: {stats["mean_activation_frequency"]:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of activation strengths (excluding dead features)
    active_strengths = stats['activation_strengths'][stats['activation_strengths'] > 0]
    axes[1].hist(active_strengths, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Mean Activation Strength')
    axes[1].set_ylabel('Number of Features')
    axes[1].set_title(f'{model_name}: Feature Activation Strengths (Active Features Only)')
    axes[1].axvline(stats['mean_activation_strength'], color='r', linestyle='--',
                    label=f'Mean: {stats["mean_activation_strength"]:.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

```

Initializing


```python
# train_loader, test_loader, mean = load_olivetti_data(batch_size=32)
train_loader, test_loader, mean = load_lfw_data(batch_size=128, img_size=64)
```

# Base usage


```python
train_loader, test_loader, mean = load_data(dataset_name='mnist', batch_size=128)

modelBase = train_sparse_autoencoder(
    train_loader,
    num_epochs=50,
    learning_rate=0.001,
    input_size=784,
    hidden_size=256,
    k_top=40,
    modelType="Complete",
    dataset_type="mnist"
)

```


```python
visualize_weights_decoder(modelBase, num_features=256)
count_dead_neurons(modelBase, train_loader, dataset_type='mnist')
test_loss(modelBase, test_loader, dataset_type='mnist')
plot_activation_histogram(modelBase, train_loader, dataset_type='mnist')
```


```python
train_loader, test_loader, mean = load_data(dataset_name='imagenet',
                                            batch_size=128,
                                            img_size=64)

modelImagenet = train_sparse_autoencoder(
    train_loader,
    num_epochs=50,
    learning_rate=0.001,
    input_size=4096,
    hidden_size=256,
    k_top=40,
    modelType="SAE_Init",
    dataset_type="imagenet"
)

```


```python
visualize_weights_decoder(modelImagenet, num_features=256)
count_dead_neurons(modelImagenet, train_loader, dataset_type='imagenet')
test_loss(modelImagenet, test_loader, dataset_type='imagenet')
plot_activation_histogram(modelImagenet, train_loader, dataset_type='imagenet')
```

    Auto-detected input_size: 4096
    Using image shape: (64, 64)



    
![png](Sparse%20Autoencoder%20Research_files/Sparse%20Autoencoder%20Research_29_1.png)
    


    Number of dead neurons in Sparse Autoencoder with just weight initialization: 40 out of 256 (15.62%)


# TopK Sparsity Analysis


```python
# Test: K sensitivity sweep
dataset_configs = [
    {'name': 'mnist', 'input_size': 784, 'batch_size': 256},
    {'name': 'olivetti', 'input_size': 4096, 'batch_size': 32},
    {'name': 'lfw', 'input_size': 4096, 'batch_size': 128}
]

k_values = [5, 10, 20, 30, 40, 50, 64, 100, 128]
hidden_size = 256  # Fixed overcomplete representation

results = []
for dataset_config in dataset_configs:
    train_loader, test_loader, mean = load_data(dataset_config['name'],
                                                 dataset_config['batch_size'])

    for k in k_values:
        model = train_sparse_autoencoder(
            train_loader,
            num_epochs=50,
            input_size=dataset_config['input_size'],
            hidden_size=hidden_size,
            k_top=k,
            modelType="SAE",
            dataset_type=dataset_config['name']
        )

        # Metrics
        test_mse = test_loss(model, test_loader, dataset_config['name'])
        dead_neurons = count_dead_neurons(model, train_loader, dataset_config['name'])
        stats = get_activation_statistics(model, train_loader, dataset_config['name'])

        results.append({
            'dataset': dataset_config['name'],
            'k': k,
            'sparsity_ratio': k/hidden_size,
            'test_mse': test_mse,
            'dead_neurons': dead_neurons,
            'active_features': stats['active_features'],
            'mean_activation_freq': stats['mean_activation_frequency']
        })

```


```python

```

same test different initializations


```python
# Test: K sensitivity sweep
dataset_configs = [
    {'name': 'mnist', 'input_size': 784, 'batch_size': 256},
    {'name': 'olivetti', 'input_size': 4096, 'batch_size': 32},
    {'name': 'lfw', 'input_size': 4096, 'batch_size': 128}
]

k_values = [5, 10, 20, 30, 40, 50, 64, 100, 128]
hidden_size = 128  # Fixed overcomplete representation

results_init = []
for dataset_config in dataset_configs:
    train_loader, test_loader, mean = load_data(dataset_config['name'],
                                                 dataset_config['batch_size'])

    for k in k_values:
        model = train_sparse_autoencoder(
            train_loader,
            num_epochs=50,
            input_size=dataset_config['input_size'],
            hidden_size=hidden_size,
            k_top=k,
            modelType="SAE_Init",
            dataset_type=dataset_config['name']
        )

        # Metrics
        test_mse = test_loss(model, test_loader, dataset_config['name'])
        dead_neurons = count_dead_neurons(model, train_loader, dataset_config['name'])
        stats = get_activation_statistics(model, train_loader, dataset_config['name'])

        results_init.append({
            'dataset': dataset_config['name'],
            'k': k,
            'sparsity_ratio': k/hidden_size,
            'test_mse': test_mse,
            'dead_neurons': dead_neurons,
            'active_features': stats['active_features'],
            'mean_activation_freq': stats['mean_activation_frequency']
        })

```


```python
# Test: K sensitivity sweep
dataset_configs = [
    {'name': 'mnist', 'input_size': 784, 'batch_size': 256},
    {'name': 'olivetti', 'input_size': 4096, 'batch_size': 32},
    {'name': 'lfw', 'input_size': 4096, 'batch_size': 128}
]

k_values = [5, 10, 20, 30, 40, 50, 64, 100, 128]
hidden_size = 128  # Fixed overcomplete representation

results_complete = []
for dataset_config in dataset_configs:
    train_loader, test_loader, mean = load_data(dataset_config['name'],
                                                 dataset_config['batch_size'])

    for k in k_values:
        model = train_sparse_autoencoder(
            train_loader,
            num_epochs=50,
            input_size=dataset_config['input_size'],
            hidden_size=hidden_size,
            k_top=k,
            modelType="Complete",
            dataset_type=dataset_config['name']
        )

        # Metrics
        test_mse = test_loss(model, test_loader, dataset_config['name'])
        dead_neurons = count_dead_neurons(model, train_loader, dataset_config['name'])
        stats = get_activation_statistics(model, train_loader, dataset_config['name'])

        results_complete.append({
            'dataset': dataset_config['name'],
            'k': k,
            'sparsity_ratio': k/hidden_size,
            'test_mse': test_mse,
            'dead_neurons': dead_neurons,
            'active_features': stats['active_features'],
            'mean_activation_freq': stats['mean_activation_frequency']
        })

```

    Epoch [1/50], Loss: 0.2572
    Epoch [2/50], Loss: 0.1375
    Epoch [3/50], Loss: 0.1305
    Epoch [4/50], Loss: 0.1281
    Epoch [5/50], Loss: 0.1341
    Epoch [6/50], Loss: 0.1334
    Epoch [7/50], Loss: 0.1329
    Epoch [8/50], Loss: 0.1322
    Epoch [9/50], Loss: 0.1325
    Epoch [10/50], Loss: 0.1330
    Epoch [11/50], Loss: 0.1336
    Epoch [12/50], Loss: 0.1341
    Epoch [13/50], Loss: 0.1346
    Epoch [14/50], Loss: 0.1348
    Epoch [15/50], Loss: 0.1349
    Epoch [16/50], Loss: 0.1351
    Epoch [17/50], Loss: 0.1358
    Epoch [18/50], Loss: 0.1364
    Epoch [19/50], Loss: 0.1364
    Epoch [20/50], Loss: 0.1366
    Epoch [21/50], Loss: 0.1369
    Epoch [22/50], Loss: 0.1373
    Epoch [23/50], Loss: 0.1383
    Epoch [24/50], Loss: 0.1385
    Epoch [25/50], Loss: 0.1383
    Epoch [26/50], Loss: 0.1387
    Epoch [27/50], Loss: 0.1396
    Epoch [28/50], Loss: 0.1411
    Epoch [29/50], Loss: 0.1428
    Epoch [30/50], Loss: 0.1441
    Epoch [31/50], Loss: 0.1456
    Epoch [32/50], Loss: 0.1472
    Epoch [33/50], Loss: 0.1494
    Epoch [34/50], Loss: 0.1522
    Epoch [35/50], Loss: 0.1546
    Epoch [36/50], Loss: 0.1563
    Epoch [37/50], Loss: 0.1580
    Epoch [38/50], Loss: 0.1602
    Epoch [39/50], Loss: 0.1627
    Epoch [40/50], Loss: 0.1654
    Epoch [41/50], Loss: 0.1684
    Epoch [42/50], Loss: 0.1716
    Epoch [43/50], Loss: 0.1750
    Epoch [44/50], Loss: 0.1781
    Epoch [45/50], Loss: 0.1809
    Epoch [46/50], Loss: 0.1831
    Epoch [47/50], Loss: 0.1852
    Epoch [48/50], Loss: 0.1867
    Epoch [49/50], Loss: 0.1878
    Epoch [50/50], Loss: 0.1887
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.175633
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 68 out of 128 (53.12%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 68 (53.12%)
    Active features: 60 (46.88%)
    Mean activation frequency: 0.0391
    Median activation frequency: 0.0000
    Mean activation strength (active features): 0.1043
    Epoch [1/50], Loss: 0.2309
    Epoch [2/50], Loss: 0.1107
    Epoch [3/50], Loss: 0.1010
    Epoch [4/50], Loss: 0.0970
    Epoch [5/50], Loss: 0.1002
    Epoch [6/50], Loss: 0.1001
    Epoch [7/50], Loss: 0.0990
    Epoch [8/50], Loss: 0.0989
    Epoch [9/50], Loss: 0.0997
    Epoch [10/50], Loss: 0.1008
    Epoch [11/50], Loss: 0.1022
    Epoch [12/50], Loss: 0.1041
    Epoch [13/50], Loss: 0.1063
    Epoch [14/50], Loss: 0.1087
    Epoch [15/50], Loss: 0.1113
    Epoch [16/50], Loss: 0.1143
    Epoch [17/50], Loss: 0.1178
    Epoch [18/50], Loss: 0.1216
    Epoch [19/50], Loss: 0.1260
    Epoch [20/50], Loss: 0.1306
    Epoch [21/50], Loss: 0.1354
    Epoch [22/50], Loss: 0.1399
    Epoch [23/50], Loss: 0.1434
    Epoch [24/50], Loss: 0.1448
    Epoch [25/50], Loss: 0.1447
    Epoch [26/50], Loss: 0.1446
    Epoch [27/50], Loss: 0.1446
    Epoch [28/50], Loss: 0.1447
    Epoch [29/50], Loss: 0.1448
    Epoch [30/50], Loss: 0.1449
    Epoch [31/50], Loss: 0.1448
    Epoch [32/50], Loss: 0.1449
    Epoch [33/50], Loss: 0.1450
    Epoch [34/50], Loss: 0.1451
    Epoch [35/50], Loss: 0.1452
    Epoch [36/50], Loss: 0.1452
    Epoch [37/50], Loss: 0.1452
    Epoch [38/50], Loss: 0.1453
    Epoch [39/50], Loss: 0.1454
    Epoch [40/50], Loss: 0.1455
    Epoch [41/50], Loss: 0.1457
    Epoch [42/50], Loss: 0.1458
    Epoch [43/50], Loss: 0.1460
    Epoch [44/50], Loss: 0.1462
    Epoch [45/50], Loss: 0.1463
    Epoch [46/50], Loss: 0.1465
    Epoch [47/50], Loss: 0.1468
    Epoch [48/50], Loss: 0.1471
    Epoch [49/50], Loss: 0.1474
    Epoch [50/50], Loss: 0.1477
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.136519
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 82 out of 128 (64.06%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 82 (64.06%)
    Active features: 46 (35.94%)
    Mean activation frequency: 0.0781
    Median activation frequency: 0.0000
    Mean activation strength (active features): 0.5427
    Epoch [1/50], Loss: 0.2272
    Epoch [2/50], Loss: 0.0883
    Epoch [3/50], Loss: 0.0753
    Epoch [4/50], Loss: 0.0695
    Epoch [5/50], Loss: 0.0810
    Epoch [6/50], Loss: 0.0764
    Epoch [7/50], Loss: 0.0629
    Epoch [8/50], Loss: 0.0613
    Epoch [9/50], Loss: 0.0604
    Epoch [10/50], Loss: 0.0755
    Epoch [11/50], Loss: 0.0721
    Epoch [12/50], Loss: 0.0723
    Epoch [13/50], Loss: 0.0730
    Epoch [14/50], Loss: 0.0729
    Epoch [15/50], Loss: 0.0713
    Epoch [16/50], Loss: 0.0706
    Epoch [17/50], Loss: 0.0705
    Epoch [18/50], Loss: 0.0706
    Epoch [19/50], Loss: 0.0708
    Epoch [20/50], Loss: 0.0714
    Epoch [21/50], Loss: 0.0721
    Epoch [22/50], Loss: 0.0728
    Epoch [23/50], Loss: 0.0736
    Epoch [24/50], Loss: 0.0744
    Epoch [25/50], Loss: 0.0746
    Epoch [26/50], Loss: 0.0757
    Epoch [27/50], Loss: 0.0754
    Epoch [28/50], Loss: 0.0758
    Epoch [29/50], Loss: 0.0762
    Epoch [30/50], Loss: 0.0763
    Epoch [31/50], Loss: 0.0761
    Epoch [32/50], Loss: 0.0764
    Epoch [33/50], Loss: 0.0764
    Epoch [34/50], Loss: 0.0765
    Epoch [35/50], Loss: 0.0768
    Epoch [36/50], Loss: 0.0773
    Epoch [37/50], Loss: 0.0781
    Epoch [38/50], Loss: 0.0782
    Epoch [39/50], Loss: 0.0789
    Epoch [40/50], Loss: 0.0793
    Epoch [41/50], Loss: 0.0800
    Epoch [42/50], Loss: 0.0803
    Epoch [43/50], Loss: 0.0807
    Epoch [44/50], Loss: 0.0807
    Epoch [45/50], Loss: 0.0812
    Epoch [46/50], Loss: 0.0810
    Epoch [47/50], Loss: 0.0808
    Epoch [48/50], Loss: 0.0816
    Epoch [49/50], Loss: 0.0818
    Epoch [50/50], Loss: 0.0819
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.071842
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 17 out of 128 (13.28%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 17 (13.28%)
    Active features: 111 (86.72%)
    Mean activation frequency: 0.1548
    Median activation frequency: 0.0660
    Mean activation strength (active features): 0.0278
    Epoch [1/50], Loss: 0.2229
    Epoch [2/50], Loss: 0.0761
    Epoch [3/50], Loss: 0.0618
    Epoch [4/50], Loss: 0.0555
    Epoch [5/50], Loss: 0.0713
    Epoch [6/50], Loss: 0.0663
    Epoch [7/50], Loss: 0.0523
    Epoch [8/50], Loss: 0.0471
    Epoch [9/50], Loss: 0.0516
    Epoch [10/50], Loss: 0.0597
    Epoch [11/50], Loss: 0.0583
    Epoch [12/50], Loss: 0.0582
    Epoch [13/50], Loss: 0.0581
    Epoch [14/50], Loss: 0.0583
    Epoch [15/50], Loss: 0.0582
    Epoch [16/50], Loss: 0.0585
    Epoch [17/50], Loss: 0.0589
    Epoch [18/50], Loss: 0.0596
    Epoch [19/50], Loss: 0.0598
    Epoch [20/50], Loss: 0.0599
    Epoch [21/50], Loss: 0.0599
    Epoch [22/50], Loss: 0.0599
    Epoch [23/50], Loss: 0.0595
    Epoch [24/50], Loss: 0.0596
    Epoch [25/50], Loss: 0.0597
    Epoch [26/50], Loss: 0.0598
    Epoch [27/50], Loss: 0.0597
    Epoch [28/50], Loss: 0.0599
    Epoch [29/50], Loss: 0.0600
    Epoch [30/50], Loss: 0.0602
    Epoch [31/50], Loss: 0.0604
    Epoch [32/50], Loss: 0.0605
    Epoch [33/50], Loss: 0.0607
    Epoch [34/50], Loss: 0.0608
    Epoch [35/50], Loss: 0.0610
    Epoch [36/50], Loss: 0.0613
    Epoch [37/50], Loss: 0.0615
    Epoch [38/50], Loss: 0.0616
    Epoch [39/50], Loss: 0.0617
    Epoch [40/50], Loss: 0.0618
    Epoch [41/50], Loss: 0.0616
    Epoch [42/50], Loss: 0.0617
    Epoch [43/50], Loss: 0.0619
    Epoch [44/50], Loss: 0.0621
    Epoch [45/50], Loss: 0.0623
    Epoch [46/50], Loss: 0.0625
    Epoch [47/50], Loss: 0.0626
    Epoch [48/50], Loss: 0.0628
    Epoch [49/50], Loss: 0.0629
    Epoch [50/50], Loss: 0.0630
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.052545
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 7 out of 128 (5.47%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 7 (5.47%)
    Active features: 121 (94.53%)
    Mean activation frequency: 0.2266
    Median activation frequency: 0.1722
    Mean activation strength (active features): 0.0348
    Epoch [1/50], Loss: 0.2278
    Epoch [2/50], Loss: 0.0710
    Epoch [3/50], Loss: 0.0549
    Epoch [4/50], Loss: 0.0479
    Epoch [5/50], Loss: 0.0636
    Epoch [6/50], Loss: 0.0523
    Epoch [7/50], Loss: 0.0419
    Epoch [8/50], Loss: 0.0402
    Epoch [9/50], Loss: 0.0393
    Epoch [10/50], Loss: 0.0387
    Epoch [11/50], Loss: 0.0599
    Epoch [12/50], Loss: 0.0408
    Epoch [13/50], Loss: 0.0385
    Epoch [14/50], Loss: 0.0380
    Epoch [15/50], Loss: 0.0405
    Epoch [16/50], Loss: 0.0551
    Epoch [17/50], Loss: 0.0519
    Epoch [18/50], Loss: 0.0516
    Epoch [19/50], Loss: 0.0514
    Epoch [20/50], Loss: 0.0512
    Epoch [21/50], Loss: 0.0509
    Epoch [22/50], Loss: 0.0509
    Epoch [23/50], Loss: 0.0508
    Epoch [24/50], Loss: 0.0507
    Epoch [25/50], Loss: 0.0507
    Epoch [26/50], Loss: 0.0506
    Epoch [27/50], Loss: 0.0505
    Epoch [28/50], Loss: 0.0504
    Epoch [29/50], Loss: 0.0503
    Epoch [30/50], Loss: 0.0502
    Epoch [31/50], Loss: 0.0502
    Epoch [32/50], Loss: 0.0501
    Epoch [33/50], Loss: 0.0501
    Epoch [34/50], Loss: 0.0501
    Epoch [35/50], Loss: 0.0501
    Epoch [36/50], Loss: 0.0501
    Epoch [37/50], Loss: 0.0501
    Epoch [38/50], Loss: 0.0501
    Epoch [39/50], Loss: 0.0501
    Epoch [40/50], Loss: 0.0501
    Epoch [41/50], Loss: 0.0502
    Epoch [42/50], Loss: 0.0502
    Epoch [43/50], Loss: 0.0502
    Epoch [44/50], Loss: 0.0502
    Epoch [45/50], Loss: 0.0502
    Epoch [46/50], Loss: 0.0502
    Epoch [47/50], Loss: 0.0502
    Epoch [48/50], Loss: 0.0502
    Epoch [49/50], Loss: 0.0503
    Epoch [50/50], Loss: 0.0503
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.040108
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 2 out of 128 (1.56%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 2 (1.56%)
    Active features: 126 (98.44%)
    Mean activation frequency: 0.2932
    Median activation frequency: 0.2991
    Mean activation strength (active features): 0.0382
    Epoch [1/50], Loss: 0.2210
    Epoch [2/50], Loss: 0.0672
    Epoch [3/50], Loss: 0.0509
    Epoch [4/50], Loss: 0.0442
    Epoch [5/50], Loss: 0.0404
    Epoch [6/50], Loss: 0.0677
    Epoch [7/50], Loss: 0.0416
    Epoch [8/50], Loss: 0.0370
    Epoch [9/50], Loss: 0.0355
    Epoch [10/50], Loss: 0.0346
    Epoch [11/50], Loss: 0.0517
    Epoch [12/50], Loss: 0.0361
    Epoch [13/50], Loss: 0.0341
    Epoch [14/50], Loss: 0.0334
    Epoch [15/50], Loss: 0.0330
    Epoch [16/50], Loss: 0.0497
    Epoch [17/50], Loss: 0.0475
    Epoch [18/50], Loss: 0.0470
    Epoch [19/50], Loss: 0.0466
    Epoch [20/50], Loss: 0.0463
    Epoch [21/50], Loss: 0.0460
    Epoch [22/50], Loss: 0.0459
    Epoch [23/50], Loss: 0.0457
    Epoch [24/50], Loss: 0.0456
    Epoch [25/50], Loss: 0.0455
    Epoch [26/50], Loss: 0.0454
    Epoch [27/50], Loss: 0.0454
    Epoch [28/50], Loss: 0.0453
    Epoch [29/50], Loss: 0.0453
    Epoch [30/50], Loss: 0.0452
    Epoch [31/50], Loss: 0.0452
    Epoch [32/50], Loss: 0.0452
    Epoch [33/50], Loss: 0.0452
    Epoch [34/50], Loss: 0.0452
    Epoch [35/50], Loss: 0.0452
    Epoch [36/50], Loss: 0.0452
    Epoch [37/50], Loss: 0.0452
    Epoch [38/50], Loss: 0.0452
    Epoch [39/50], Loss: 0.0452
    Epoch [40/50], Loss: 0.0452
    Epoch [41/50], Loss: 0.0453
    Epoch [42/50], Loss: 0.0453
    Epoch [43/50], Loss: 0.0453
    Epoch [44/50], Loss: 0.0453
    Epoch [45/50], Loss: 0.0453
    Epoch [46/50], Loss: 0.0453
    Epoch [47/50], Loss: 0.0453
    Epoch [48/50], Loss: 0.0453
    Epoch [49/50], Loss: 0.0453
    Epoch [50/50], Loss: 0.0453
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.035268
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 1 out of 128 (0.78%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 1 (0.78%)
    Active features: 127 (99.22%)
    Mean activation frequency: 0.3561
    Median activation frequency: 0.3817
    Mean activation strength (active features): 0.0419
    Epoch [1/50], Loss: 0.2149
    Epoch [2/50], Loss: 0.0660
    Epoch [3/50], Loss: 0.0485
    Epoch [4/50], Loss: 0.0418
    Epoch [5/50], Loss: 0.0381
    Epoch [6/50], Loss: 0.0358
    Epoch [7/50], Loss: 0.0342
    Epoch [8/50], Loss: 0.0330
    Epoch [9/50], Loss: 0.0321
    Epoch [10/50], Loss: 0.0313
    Epoch [11/50], Loss: 0.0308
    Epoch [12/50], Loss: 0.0303
    Epoch [13/50], Loss: 0.0299
    Epoch [14/50], Loss: 0.0295
    Epoch [15/50], Loss: 0.0292
    Epoch [16/50], Loss: 0.0290
    Epoch [17/50], Loss: 0.0288
    Epoch [18/50], Loss: 0.0286
    Epoch [19/50], Loss: 0.0284
    Epoch [20/50], Loss: 0.0283
    Epoch [21/50], Loss: 0.0281
    Epoch [22/50], Loss: 0.0280
    Epoch [23/50], Loss: 0.0279
    Epoch [24/50], Loss: 0.0278
    Epoch [25/50], Loss: 0.0278
    Epoch [26/50], Loss: 0.0277
    Epoch [27/50], Loss: 0.0276
    Epoch [28/50], Loss: 0.0275
    Epoch [29/50], Loss: 0.0274
    Epoch [30/50], Loss: 0.0274
    Epoch [31/50], Loss: 0.0274
    Epoch [32/50], Loss: 0.0273
    Epoch [33/50], Loss: 0.0273
    Epoch [34/50], Loss: 0.0272
    Epoch [35/50], Loss: 0.0272
    Epoch [36/50], Loss: 0.0271
    Epoch [37/50], Loss: 0.0271
    Epoch [38/50], Loss: 0.0271
    Epoch [39/50], Loss: 0.0271
    Epoch [40/50], Loss: 0.0270
    Epoch [41/50], Loss: 0.0270
    Epoch [42/50], Loss: 0.0270
    Epoch [43/50], Loss: 0.0269
    Epoch [44/50], Loss: 0.0269
    Epoch [45/50], Loss: 0.0269
    Epoch [46/50], Loss: 0.0269
    Epoch [47/50], Loss: 0.0269
    Epoch [48/50], Loss: 0.0268
    Epoch [49/50], Loss: 0.0268
    Epoch [50/50], Loss: 0.0268
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.026110
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.5000
    Median activation frequency: 0.5205
    Mean activation strength (active features): 0.0632
    Epoch [1/50], Loss: 0.2366
    Epoch [2/50], Loss: 0.0679
    Epoch [3/50], Loss: 0.0463
    Epoch [4/50], Loss: 0.0374
    Epoch [5/50], Loss: 0.0331
    Epoch [6/50], Loss: 0.0306
    Epoch [7/50], Loss: 0.0290
    Epoch [8/50], Loss: 0.0278
    Epoch [9/50], Loss: 0.0270
    Epoch [10/50], Loss: 0.0264
    Epoch [11/50], Loss: 0.0260
    Epoch [12/50], Loss: 0.0256
    Epoch [13/50], Loss: 0.0253
    Epoch [14/50], Loss: 0.0250
    Epoch [15/50], Loss: 0.0249
    Epoch [16/50], Loss: 0.0247
    Epoch [17/50], Loss: 0.0246
    Epoch [18/50], Loss: 0.0244
    Epoch [19/50], Loss: 0.0243
    Epoch [20/50], Loss: 0.0242
    Epoch [21/50], Loss: 0.0241
    Epoch [22/50], Loss: 0.0240
    Epoch [23/50], Loss: 0.0240
    Epoch [24/50], Loss: 0.0239
    Epoch [25/50], Loss: 0.0239
    Epoch [26/50], Loss: 0.0238
    Epoch [27/50], Loss: 0.0237
    Epoch [28/50], Loss: 0.0237
    Epoch [29/50], Loss: 0.0236
    Epoch [30/50], Loss: 0.0236
    Epoch [31/50], Loss: 0.0236
    Epoch [32/50], Loss: 0.0235
    Epoch [33/50], Loss: 0.0235
    Epoch [34/50], Loss: 0.0235
    Epoch [35/50], Loss: 0.0234
    Epoch [36/50], Loss: 0.0234
    Epoch [37/50], Loss: 0.0234
    Epoch [38/50], Loss: 0.0234
    Epoch [39/50], Loss: 0.0234
    Epoch [40/50], Loss: 0.0233
    Epoch [41/50], Loss: 0.0233
    Epoch [42/50], Loss: 0.0233
    Epoch [43/50], Loss: 0.0233
    Epoch [44/50], Loss: 0.0233
    Epoch [45/50], Loss: 0.0232
    Epoch [46/50], Loss: 0.0233
    Epoch [47/50], Loss: 0.0233
    Epoch [48/50], Loss: 0.0233
    Epoch [49/50], Loss: 0.0232
    Epoch [50/50], Loss: 0.0232
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.022496
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.7811
    Median activation frequency: 0.9447
    Mean activation strength (active features): 0.0920
    Epoch [1/50], Loss: 0.2227
    Epoch [2/50], Loss: 0.0655
    Epoch [3/50], Loss: 0.0458
    Epoch [4/50], Loss: 0.0369
    Epoch [5/50], Loss: 0.0319
    Epoch [6/50], Loss: 0.0289
    Epoch [7/50], Loss: 0.0271
    Epoch [8/50], Loss: 0.0258
    Epoch [9/50], Loss: 0.0250
    Epoch [10/50], Loss: 0.0244
    Epoch [11/50], Loss: 0.0239
    Epoch [12/50], Loss: 0.0236
    Epoch [13/50], Loss: 0.0233
    Epoch [14/50], Loss: 0.0231
    Epoch [15/50], Loss: 0.0229
    Epoch [16/50], Loss: 0.0227
    Epoch [17/50], Loss: 0.0226
    Epoch [18/50], Loss: 0.0225
    Epoch [19/50], Loss: 0.0224
    Epoch [20/50], Loss: 0.0223
    Epoch [21/50], Loss: 0.0222
    Epoch [22/50], Loss: 0.0222
    Epoch [23/50], Loss: 0.0221
    Epoch [24/50], Loss: 0.0221
    Epoch [25/50], Loss: 0.0221
    Epoch [26/50], Loss: 0.0220
    Epoch [27/50], Loss: 0.0220
    Epoch [28/50], Loss: 0.0219
    Epoch [29/50], Loss: 0.0219
    Epoch [30/50], Loss: 0.0219
    Epoch [31/50], Loss: 0.0219
    Epoch [32/50], Loss: 0.0218
    Epoch [33/50], Loss: 0.0219
    Epoch [34/50], Loss: 0.0218
    Epoch [35/50], Loss: 0.0218
    Epoch [36/50], Loss: 0.0218
    Epoch [37/50], Loss: 0.0217
    Epoch [38/50], Loss: 0.0218
    Epoch [39/50], Loss: 0.0217
    Epoch [40/50], Loss: 0.0218
    Epoch [41/50], Loss: 0.0217
    Epoch [42/50], Loss: 0.0218
    Epoch [43/50], Loss: 0.0217
    Epoch [44/50], Loss: 0.0217
    Epoch [45/50], Loss: 0.0217
    Epoch [46/50], Loss: 0.0217
    Epoch [47/50], Loss: 0.0216
    Epoch [48/50], Loss: 0.0217
    Epoch [49/50], Loss: 0.0218
    Epoch [50/50], Loss: 0.0216
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.020911
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.9923
    Median activation frequency: 0.9927
    Mean activation strength (active features): 0.1064
    Epoch [1/50], Loss: 0.0764
    Epoch [2/50], Loss: 0.0401
    Epoch [3/50], Loss: 0.0248
    Epoch [4/50], Loss: 0.0170
    Epoch [5/50], Loss: 0.0124
    Epoch [6/50], Loss: 0.0093
    Epoch [7/50], Loss: 0.0074
    Epoch [8/50], Loss: 0.0061
    Epoch [9/50], Loss: 0.0053
    Epoch [10/50], Loss: 0.0048
    Epoch [11/50], Loss: 0.0044
    Epoch [12/50], Loss: 0.0042
    Epoch [13/50], Loss: 0.0040
    Epoch [14/50], Loss: 0.0038
    Epoch [15/50], Loss: 0.0037
    Epoch [16/50], Loss: 0.0035
    Epoch [17/50], Loss: 0.0035
    Epoch [18/50], Loss: 0.0034
    Epoch [19/50], Loss: 0.0033
    Epoch [20/50], Loss: 0.0032
    Epoch [21/50], Loss: 0.0032
    Epoch [22/50], Loss: 0.0031
    Epoch [23/50], Loss: 0.0031
    Epoch [24/50], Loss: 0.0030
    Epoch [25/50], Loss: 0.0030
    Epoch [26/50], Loss: 0.0029
    Epoch [27/50], Loss: 0.0029
    Epoch [28/50], Loss: 0.0029
    Epoch [29/50], Loss: 0.0029
    Epoch [30/50], Loss: 0.0028
    Epoch [31/50], Loss: 0.0028
    Epoch [32/50], Loss: 0.0028
    Epoch [33/50], Loss: 0.0027
    Epoch [34/50], Loss: 0.0027
    Epoch [35/50], Loss: 0.0027
    Epoch [36/50], Loss: 0.0027
    Epoch [37/50], Loss: 0.0027
    Epoch [38/50], Loss: 0.0027
    Epoch [39/50], Loss: 0.0026
    Epoch [40/50], Loss: 0.0026
    Epoch [41/50], Loss: 0.0026
    Epoch [42/50], Loss: 0.0026
    Epoch [43/50], Loss: 0.0026
    Epoch [44/50], Loss: 0.0026
    Epoch [45/50], Loss: 0.0025
    Epoch [46/50], Loss: 0.0025
    Epoch [47/50], Loss: 0.0025
    Epoch [48/50], Loss: 0.0025
    Epoch [49/50], Loss: 0.0025
    Epoch [50/50], Loss: 0.0025
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.002822
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 89 out of 128 (69.53%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 89 (69.53%)
    Active features: 39 (30.47%)
    Mean activation frequency: 0.0391
    Median activation frequency: 0.0000
    Mean activation strength (active features): 0.0576
    Epoch [1/50], Loss: 0.0713
    Epoch [2/50], Loss: 0.0357
    Epoch [3/50], Loss: 0.0230
    Epoch [4/50], Loss: 0.0163
    Epoch [5/50], Loss: 0.0119
    Epoch [6/50], Loss: 0.0090
    Epoch [7/50], Loss: 0.0071
    Epoch [8/50], Loss: 0.0059
    Epoch [9/50], Loss: 0.0051
    Epoch [10/50], Loss: 0.0045
    Epoch [11/50], Loss: 0.0041
    Epoch [12/50], Loss: 0.0039
    Epoch [13/50], Loss: 0.0036
    Epoch [14/50], Loss: 0.0035
    Epoch [15/50], Loss: 0.0033
    Epoch [16/50], Loss: 0.0032
    Epoch [17/50], Loss: 0.0031
    Epoch [18/50], Loss: 0.0030
    Epoch [19/50], Loss: 0.0029
    Epoch [20/50], Loss: 0.0028
    Epoch [21/50], Loss: 0.0028
    Epoch [22/50], Loss: 0.0027
    Epoch [23/50], Loss: 0.0026
    Epoch [24/50], Loss: 0.0026
    Epoch [25/50], Loss: 0.0025
    Epoch [26/50], Loss: 0.0025
    Epoch [27/50], Loss: 0.0025
    Epoch [28/50], Loss: 0.0024
    Epoch [29/50], Loss: 0.0024
    Epoch [30/50], Loss: 0.0023
    Epoch [31/50], Loss: 0.0023
    Epoch [32/50], Loss: 0.0023
    Epoch [33/50], Loss: 0.0023
    Epoch [34/50], Loss: 0.0022
    Epoch [35/50], Loss: 0.0022
    Epoch [36/50], Loss: 0.0022
    Epoch [37/50], Loss: 0.0022
    Epoch [38/50], Loss: 0.0022
    Epoch [39/50], Loss: 0.0021
    Epoch [40/50], Loss: 0.0021
    Epoch [41/50], Loss: 0.0021
    Epoch [42/50], Loss: 0.0021
    Epoch [43/50], Loss: 0.0021
    Epoch [44/50], Loss: 0.0020
    Epoch [45/50], Loss: 0.0020
    Epoch [46/50], Loss: 0.0020
    Epoch [47/50], Loss: 0.0020
    Epoch [48/50], Loss: 0.0020
    Epoch [49/50], Loss: 0.0020
    Epoch [50/50], Loss: 0.0020
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.002437
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 67 out of 128 (52.34%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 67 (52.34%)
    Active features: 61 (47.66%)
    Mean activation frequency: 0.0781
    Median activation frequency: 0.0000
    Mean activation strength (active features): 0.0506
    Epoch [1/50], Loss: 0.0685
    Epoch [2/50], Loss: 0.0345
    Epoch [3/50], Loss: 0.0232
    Epoch [4/50], Loss: 0.0165
    Epoch [5/50], Loss: 0.0120
    Epoch [6/50], Loss: 0.0091
    Epoch [7/50], Loss: 0.0071
    Epoch [8/50], Loss: 0.0058
    Epoch [9/50], Loss: 0.0049
    Epoch [10/50], Loss: 0.0043
    Epoch [11/50], Loss: 0.0039
    Epoch [12/50], Loss: 0.0036
    Epoch [13/50], Loss: 0.0033
    Epoch [14/50], Loss: 0.0031
    Epoch [15/50], Loss: 0.0029
    Epoch [16/50], Loss: 0.0028
    Epoch [17/50], Loss: 0.0027
    Epoch [18/50], Loss: 0.0026
    Epoch [19/50], Loss: 0.0025
    Epoch [20/50], Loss: 0.0024
    Epoch [21/50], Loss: 0.0023
    Epoch [22/50], Loss: 0.0022
    Epoch [23/50], Loss: 0.0022
    Epoch [24/50], Loss: 0.0021
    Epoch [25/50], Loss: 0.0021
    Epoch [26/50], Loss: 0.0020
    Epoch [27/50], Loss: 0.0020
    Epoch [28/50], Loss: 0.0019
    Epoch [29/50], Loss: 0.0019
    Epoch [30/50], Loss: 0.0019
    Epoch [31/50], Loss: 0.0019
    Epoch [32/50], Loss: 0.0018
    Epoch [33/50], Loss: 0.0018
    Epoch [34/50], Loss: 0.0018
    Epoch [35/50], Loss: 0.0018
    Epoch [36/50], Loss: 0.0019
    Epoch [37/50], Loss: 0.0019
    Epoch [38/50], Loss: 0.0019
    Epoch [39/50], Loss: 0.0018
    Epoch [40/50], Loss: 0.0018
    Epoch [41/50], Loss: 0.0017
    Epoch [42/50], Loss: 0.0017
    Epoch [43/50], Loss: 0.0017
    Epoch [44/50], Loss: 0.0016
    Epoch [45/50], Loss: 0.0016
    Epoch [46/50], Loss: 0.0016
    Epoch [47/50], Loss: 0.0016
    Epoch [48/50], Loss: 0.0016
    Epoch [49/50], Loss: 0.0016
    Epoch [50/50], Loss: 0.0016
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.002178
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 26 out of 128 (20.31%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 26 (20.31%)
    Active features: 102 (79.69%)
    Mean activation frequency: 0.1562
    Median activation frequency: 0.1719
    Mean activation strength (active features): 0.0428
    Epoch [1/50], Loss: 0.0690
    Epoch [2/50], Loss: 0.0345
    Epoch [3/50], Loss: 0.0234
    Epoch [4/50], Loss: 0.0167
    Epoch [5/50], Loss: 0.0122
    Epoch [6/50], Loss: 0.0092
    Epoch [7/50], Loss: 0.0072
    Epoch [8/50], Loss: 0.0058
    Epoch [9/50], Loss: 0.0049
    Epoch [10/50], Loss: 0.0043
    Epoch [11/50], Loss: 0.0039
    Epoch [12/50], Loss: 0.0035
    Epoch [13/50], Loss: 0.0033
    Epoch [14/50], Loss: 0.0031
    Epoch [15/50], Loss: 0.0029
    Epoch [16/50], Loss: 0.0027
    Epoch [17/50], Loss: 0.0026
    Epoch [18/50], Loss: 0.0025
    Epoch [19/50], Loss: 0.0024
    Epoch [20/50], Loss: 0.0023
    Epoch [21/50], Loss: 0.0022
    Epoch [22/50], Loss: 0.0022
    Epoch [23/50], Loss: 0.0021
    Epoch [24/50], Loss: 0.0020
    Epoch [25/50], Loss: 0.0020
    Epoch [26/50], Loss: 0.0019
    Epoch [27/50], Loss: 0.0019
    Epoch [28/50], Loss: 0.0019
    Epoch [29/50], Loss: 0.0018
    Epoch [30/50], Loss: 0.0018
    Epoch [31/50], Loss: 0.0018
    Epoch [32/50], Loss: 0.0018
    Epoch [33/50], Loss: 0.0017
    Epoch [34/50], Loss: 0.0017
    Epoch [35/50], Loss: 0.0017
    Epoch [36/50], Loss: 0.0017
    Epoch [37/50], Loss: 0.0016
    Epoch [38/50], Loss: 0.0016
    Epoch [39/50], Loss: 0.0016
    Epoch [40/50], Loss: 0.0016
    Epoch [41/50], Loss: 0.0016
    Epoch [42/50], Loss: 0.0016
    Epoch [43/50], Loss: 0.0016
    Epoch [44/50], Loss: 0.0016
    Epoch [45/50], Loss: 0.0016
    Epoch [46/50], Loss: 0.0016
    Epoch [47/50], Loss: 0.0017
    Epoch [48/50], Loss: 0.0016
    Epoch [49/50], Loss: 0.0016
    Epoch [50/50], Loss: 0.0015
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.001972
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 7 out of 128 (5.47%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 7 (5.47%)
    Active features: 121 (94.53%)
    Mean activation frequency: 0.2344
    Median activation frequency: 0.2562
    Mean activation strength (active features): 0.0422
    Epoch [1/50], Loss: 0.0689
    Epoch [2/50], Loss: 0.0344
    Epoch [3/50], Loss: 0.0232
    Epoch [4/50], Loss: 0.0166
    Epoch [5/50], Loss: 0.0122
    Epoch [6/50], Loss: 0.0092
    Epoch [7/50], Loss: 0.0072
    Epoch [8/50], Loss: 0.0059
    Epoch [9/50], Loss: 0.0049
    Epoch [10/50], Loss: 0.0043
    Epoch [11/50], Loss: 0.0039
    Epoch [12/50], Loss: 0.0035
    Epoch [13/50], Loss: 0.0033
    Epoch [14/50], Loss: 0.0031
    Epoch [15/50], Loss: 0.0029
    Epoch [16/50], Loss: 0.0027
    Epoch [17/50], Loss: 0.0026
    Epoch [18/50], Loss: 0.0025
    Epoch [19/50], Loss: 0.0024
    Epoch [20/50], Loss: 0.0023
    Epoch [21/50], Loss: 0.0022
    Epoch [22/50], Loss: 0.0021
    Epoch [23/50], Loss: 0.0021
    Epoch [24/50], Loss: 0.0020
    Epoch [25/50], Loss: 0.0019
    Epoch [26/50], Loss: 0.0019
    Epoch [27/50], Loss: 0.0019
    Epoch [28/50], Loss: 0.0018
    Epoch [29/50], Loss: 0.0018
    Epoch [30/50], Loss: 0.0018
    Epoch [31/50], Loss: 0.0017
    Epoch [32/50], Loss: 0.0017
    Epoch [33/50], Loss: 0.0017
    Epoch [34/50], Loss: 0.0017
    Epoch [35/50], Loss: 0.0017
    Epoch [36/50], Loss: 0.0017
    Epoch [37/50], Loss: 0.0016
    Epoch [38/50], Loss: 0.0016
    Epoch [39/50], Loss: 0.0016
    Epoch [40/50], Loss: 0.0016
    Epoch [41/50], Loss: 0.0017
    Epoch [42/50], Loss: 0.0016
    Epoch [43/50], Loss: 0.0016
    Epoch [44/50], Loss: 0.0016
    Epoch [45/50], Loss: 0.0016
    Epoch [46/50], Loss: 0.0016
    Epoch [47/50], Loss: 0.0016
    Epoch [48/50], Loss: 0.0016
    Epoch [49/50], Loss: 0.0016
    Epoch [50/50], Loss: 0.0016
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.001934
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 3 out of 128 (2.34%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 3 (2.34%)
    Active features: 125 (97.66%)
    Mean activation frequency: 0.3112
    Median activation frequency: 0.3266
    Mean activation strength (active features): 0.0468
    Epoch [1/50], Loss: 0.0678
    Epoch [2/50], Loss: 0.0330
    Epoch [3/50], Loss: 0.0227
    Epoch [4/50], Loss: 0.0164
    Epoch [5/50], Loss: 0.0120
    Epoch [6/50], Loss: 0.0091
    Epoch [7/50], Loss: 0.0071
    Epoch [8/50], Loss: 0.0058
    Epoch [9/50], Loss: 0.0049
    Epoch [10/50], Loss: 0.0043
    Epoch [11/50], Loss: 0.0039
    Epoch [12/50], Loss: 0.0035
    Epoch [13/50], Loss: 0.0033
    Epoch [14/50], Loss: 0.0031
    Epoch [15/50], Loss: 0.0029
    Epoch [16/50], Loss: 0.0027
    Epoch [17/50], Loss: 0.0026
    Epoch [18/50], Loss: 0.0025
    Epoch [19/50], Loss: 0.0024
    Epoch [20/50], Loss: 0.0023
    Epoch [21/50], Loss: 0.0022
    Epoch [22/50], Loss: 0.0021
    Epoch [23/50], Loss: 0.0020
    Epoch [24/50], Loss: 0.0020
    Epoch [25/50], Loss: 0.0019
    Epoch [26/50], Loss: 0.0019
    Epoch [27/50], Loss: 0.0018
    Epoch [28/50], Loss: 0.0018
    Epoch [29/50], Loss: 0.0018
    Epoch [30/50], Loss: 0.0017
    Epoch [31/50], Loss: 0.0017
    Epoch [32/50], Loss: 0.0017
    Epoch [33/50], Loss: 0.0018
    Epoch [34/50], Loss: 0.0018
    Epoch [35/50], Loss: 0.0019
    Epoch [36/50], Loss: 0.0020
    Epoch [37/50], Loss: 0.0019
    Epoch [38/50], Loss: 0.0018
    Epoch [39/50], Loss: 0.0018
    Epoch [40/50], Loss: 0.0017
    Epoch [41/50], Loss: 0.0016
    Epoch [42/50], Loss: 0.0016
    Epoch [43/50], Loss: 0.0015
    Epoch [44/50], Loss: 0.0014
    Epoch [45/50], Loss: 0.0014
    Epoch [46/50], Loss: 0.0014
    Epoch [47/50], Loss: 0.0014
    Epoch [48/50], Loss: 0.0014
    Epoch [49/50], Loss: 0.0014
    Epoch [50/50], Loss: 0.0013
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.001803
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.3730
    Median activation frequency: 0.3688
    Mean activation strength (active features): 0.0455
    Epoch [1/50], Loss: 0.0681
    Epoch [2/50], Loss: 0.0337
    Epoch [3/50], Loss: 0.0232
    Epoch [4/50], Loss: 0.0168
    Epoch [5/50], Loss: 0.0124
    Epoch [6/50], Loss: 0.0095
    Epoch [7/50], Loss: 0.0075
    Epoch [8/50], Loss: 0.0061
    Epoch [9/50], Loss: 0.0051
    Epoch [10/50], Loss: 0.0045
    Epoch [11/50], Loss: 0.0040
    Epoch [12/50], Loss: 0.0037
    Epoch [13/50], Loss: 0.0034
    Epoch [14/50], Loss: 0.0032
    Epoch [15/50], Loss: 0.0030
    Epoch [16/50], Loss: 0.0029
    Epoch [17/50], Loss: 0.0027
    Epoch [18/50], Loss: 0.0026
    Epoch [19/50], Loss: 0.0025
    Epoch [20/50], Loss: 0.0024
    Epoch [21/50], Loss: 0.0023
    Epoch [22/50], Loss: 0.0022
    Epoch [23/50], Loss: 0.0021
    Epoch [24/50], Loss: 0.0021
    Epoch [25/50], Loss: 0.0020
    Epoch [26/50], Loss: 0.0019
    Epoch [27/50], Loss: 0.0019
    Epoch [28/50], Loss: 0.0018
    Epoch [29/50], Loss: 0.0018
    Epoch [30/50], Loss: 0.0017
    Epoch [31/50], Loss: 0.0017
    Epoch [32/50], Loss: 0.0017
    Epoch [33/50], Loss: 0.0016
    Epoch [34/50], Loss: 0.0016
    Epoch [35/50], Loss: 0.0016
    Epoch [36/50], Loss: 0.0016
    Epoch [37/50], Loss: 0.0016
    Epoch [38/50], Loss: 0.0017
    Epoch [39/50], Loss: 0.0017
    Epoch [40/50], Loss: 0.0017
    Epoch [41/50], Loss: 0.0017
    Epoch [42/50], Loss: 0.0016
    Epoch [43/50], Loss: 0.0016
    Epoch [44/50], Loss: 0.0016
    Epoch [45/50], Loss: 0.0015
    Epoch [46/50], Loss: 0.0015
    Epoch [47/50], Loss: 0.0015
    Epoch [48/50], Loss: 0.0015
    Epoch [49/50], Loss: 0.0014
    Epoch [50/50], Loss: 0.0013
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.001790
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.4183
    Median activation frequency: 0.4313
    Mean activation strength (active features): 0.0450
    Epoch [1/50], Loss: 0.0676
    Epoch [2/50], Loss: 0.0335
    Epoch [3/50], Loss: 0.0231
    Epoch [4/50], Loss: 0.0169
    Epoch [5/50], Loss: 0.0125
    Epoch [6/50], Loss: 0.0095
    Epoch [7/50], Loss: 0.0075
    Epoch [8/50], Loss: 0.0061
    Epoch [9/50], Loss: 0.0052
    Epoch [10/50], Loss: 0.0045
    Epoch [11/50], Loss: 0.0041
    Epoch [12/50], Loss: 0.0037
    Epoch [13/50], Loss: 0.0034
    Epoch [14/50], Loss: 0.0032
    Epoch [15/50], Loss: 0.0030
    Epoch [16/50], Loss: 0.0028
    Epoch [17/50], Loss: 0.0027
    Epoch [18/50], Loss: 0.0026
    Epoch [19/50], Loss: 0.0024
    Epoch [20/50], Loss: 0.0023
    Epoch [21/50], Loss: 0.0022
    Epoch [22/50], Loss: 0.0022
    Epoch [23/50], Loss: 0.0021
    Epoch [24/50], Loss: 0.0020
    Epoch [25/50], Loss: 0.0020
    Epoch [26/50], Loss: 0.0019
    Epoch [27/50], Loss: 0.0019
    Epoch [28/50], Loss: 0.0018
    Epoch [29/50], Loss: 0.0018
    Epoch [30/50], Loss: 0.0017
    Epoch [31/50], Loss: 0.0017
    Epoch [32/50], Loss: 0.0017
    Epoch [33/50], Loss: 0.0016
    Epoch [34/50], Loss: 0.0016
    Epoch [35/50], Loss: 0.0016
    Epoch [36/50], Loss: 0.0015
    Epoch [37/50], Loss: 0.0015
    Epoch [38/50], Loss: 0.0015
    Epoch [39/50], Loss: 0.0015
    Epoch [40/50], Loss: 0.0015
    Epoch [41/50], Loss: 0.0016
    Epoch [42/50], Loss: 0.0016
    Epoch [43/50], Loss: 0.0018
    Epoch [44/50], Loss: 0.0018
    Epoch [45/50], Loss: 0.0018
    Epoch [46/50], Loss: 0.0018
    Epoch [47/50], Loss: 0.0018
    Epoch [48/50], Loss: 0.0017
    Epoch [49/50], Loss: 0.0016
    Epoch [50/50], Loss: 0.0015
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.001884
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.4063
    Median activation frequency: 0.4125
    Mean activation strength (active features): 0.0452
    Epoch [1/50], Loss: 0.0678
    Epoch [2/50], Loss: 0.0340
    Epoch [3/50], Loss: 0.0233
    Epoch [4/50], Loss: 0.0168
    Epoch [5/50], Loss: 0.0125
    Epoch [6/50], Loss: 0.0094
    Epoch [7/50], Loss: 0.0074
    Epoch [8/50], Loss: 0.0061
    Epoch [9/50], Loss: 0.0052
    Epoch [10/50], Loss: 0.0046
    Epoch [11/50], Loss: 0.0041
    Epoch [12/50], Loss: 0.0038
    Epoch [13/50], Loss: 0.0035
    Epoch [14/50], Loss: 0.0033
    Epoch [15/50], Loss: 0.0031
    Epoch [16/50], Loss: 0.0029
    Epoch [17/50], Loss: 0.0028
    Epoch [18/50], Loss: 0.0026
    Epoch [19/50], Loss: 0.0025
    Epoch [20/50], Loss: 0.0024
    Epoch [21/50], Loss: 0.0023
    Epoch [22/50], Loss: 0.0022
    Epoch [23/50], Loss: 0.0022
    Epoch [24/50], Loss: 0.0021
    Epoch [25/50], Loss: 0.0020
    Epoch [26/50], Loss: 0.0020
    Epoch [27/50], Loss: 0.0019
    Epoch [28/50], Loss: 0.0019
    Epoch [29/50], Loss: 0.0018
    Epoch [30/50], Loss: 0.0018
    Epoch [31/50], Loss: 0.0017
    Epoch [32/50], Loss: 0.0017
    Epoch [33/50], Loss: 0.0017
    Epoch [34/50], Loss: 0.0016
    Epoch [35/50], Loss: 0.0016
    Epoch [36/50], Loss: 0.0016
    Epoch [37/50], Loss: 0.0015
    Epoch [38/50], Loss: 0.0015
    Epoch [39/50], Loss: 0.0015
    Epoch [40/50], Loss: 0.0014
    Epoch [41/50], Loss: 0.0014
    Epoch [42/50], Loss: 0.0014
    Epoch [43/50], Loss: 0.0014
    Epoch [44/50], Loss: 0.0014
    Epoch [45/50], Loss: 0.0014
    Epoch [46/50], Loss: 0.0014
    Epoch [47/50], Loss: 0.0014
    Epoch [48/50], Loss: 0.0014
    Epoch [49/50], Loss: 0.0015
    Epoch [50/50], Loss: 0.0016
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.001882
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.4299
    Median activation frequency: 0.4313
    Mean activation strength (active features): 0.0423
    Original LFW shape: (3023, 125, 94)
    Resized LFW shape: (3023, 4096)
    LFW Dataset loaded: 2418 train, 605 test
    Image size: 64×64, Input dimension: 4096
    Epoch [1/50], Loss: 0.2398
    Epoch [2/50], Loss: 0.0927
    Epoch [3/50], Loss: 0.0505
    Epoch [4/50], Loss: 0.0322
    Epoch [5/50], Loss: 0.0241
    Epoch [6/50], Loss: 0.0204
    Epoch [7/50], Loss: 0.0187
    Epoch [8/50], Loss: 0.0176
    Epoch [9/50], Loss: 0.0169
    Epoch [10/50], Loss: 0.0164
    Epoch [11/50], Loss: 0.0160
    Epoch [12/50], Loss: 0.0157
    Epoch [13/50], Loss: 0.0155
    Epoch [14/50], Loss: 0.0153
    Epoch [15/50], Loss: 0.0151
    Epoch [16/50], Loss: 0.0150
    Epoch [17/50], Loss: 0.0149
    Epoch [18/50], Loss: 0.0148
    Epoch [19/50], Loss: 0.0147
    Epoch [20/50], Loss: 0.0146
    Epoch [21/50], Loss: 0.0145
    Epoch [22/50], Loss: 0.0145
    Epoch [23/50], Loss: 0.0144
    Epoch [24/50], Loss: 0.0144
    Epoch [25/50], Loss: 0.0143
    Epoch [26/50], Loss: 0.0142
    Epoch [27/50], Loss: 0.0142
    Epoch [28/50], Loss: 0.0142
    Epoch [29/50], Loss: 0.0141
    Epoch [30/50], Loss: 0.0140
    Epoch [31/50], Loss: 0.0140
    Epoch [32/50], Loss: 0.0140
    Epoch [33/50], Loss: 0.0139
    Epoch [34/50], Loss: 0.0139
    Epoch [35/50], Loss: 0.0138
    Epoch [36/50], Loss: 0.0138
    Epoch [37/50], Loss: 0.0138
    Epoch [38/50], Loss: 0.0138
    Epoch [39/50], Loss: 0.0137
    Epoch [40/50], Loss: 0.0137
    Epoch [41/50], Loss: 0.0136
    Epoch [42/50], Loss: 0.0136
    Epoch [43/50], Loss: 0.0136
    Epoch [44/50], Loss: 0.0136
    Epoch [45/50], Loss: 0.0135
    Epoch [46/50], Loss: 0.0135
    Epoch [47/50], Loss: 0.0135
    Epoch [48/50], Loss: 0.0135
    Epoch [49/50], Loss: 0.0135
    Epoch [50/50], Loss: 0.0134
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.014583
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 75 out of 128 (58.59%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 75 (58.59%)
    Active features: 53 (41.41%)
    Mean activation frequency: 0.0391
    Median activation frequency: 0.0000
    Mean activation strength (active features): 0.0354
    Epoch [1/50], Loss: 0.2177
    Epoch [2/50], Loss: 0.0843
    Epoch [3/50], Loss: 0.0464
    Epoch [4/50], Loss: 0.0296
    Epoch [5/50], Loss: 0.0222
    Epoch [6/50], Loss: 0.0188
    Epoch [7/50], Loss: 0.0171
    Epoch [8/50], Loss: 0.0160
    Epoch [9/50], Loss: 0.0152
    Epoch [10/50], Loss: 0.0146
    Epoch [11/50], Loss: 0.0142
    Epoch [12/50], Loss: 0.0138
    Epoch [13/50], Loss: 0.0135
    Epoch [14/50], Loss: 0.0133
    Epoch [15/50], Loss: 0.0130
    Epoch [16/50], Loss: 0.0129
    Epoch [17/50], Loss: 0.0127
    Epoch [18/50], Loss: 0.0126
    Epoch [19/50], Loss: 0.0125
    Epoch [20/50], Loss: 0.0123
    Epoch [21/50], Loss: 0.0122
    Epoch [22/50], Loss: 0.0121
    Epoch [23/50], Loss: 0.0120
    Epoch [24/50], Loss: 0.0119
    Epoch [25/50], Loss: 0.0118
    Epoch [26/50], Loss: 0.0118
    Epoch [27/50], Loss: 0.0117
    Epoch [28/50], Loss: 0.0116
    Epoch [29/50], Loss: 0.0116
    Epoch [30/50], Loss: 0.0115
    Epoch [31/50], Loss: 0.0114
    Epoch [32/50], Loss: 0.0114
    Epoch [33/50], Loss: 0.0113
    Epoch [34/50], Loss: 0.0113
    Epoch [35/50], Loss: 0.0112
    Epoch [36/50], Loss: 0.0112
    Epoch [37/50], Loss: 0.0112
    Epoch [38/50], Loss: 0.0111
    Epoch [39/50], Loss: 0.0111
    Epoch [40/50], Loss: 0.0110
    Epoch [41/50], Loss: 0.0110
    Epoch [42/50], Loss: 0.0110
    Epoch [43/50], Loss: 0.0110
    Epoch [44/50], Loss: 0.0109
    Epoch [45/50], Loss: 0.0109
    Epoch [46/50], Loss: 0.0109
    Epoch [47/50], Loss: 0.0108
    Epoch [48/50], Loss: 0.0108
    Epoch [49/50], Loss: 0.0108
    Epoch [50/50], Loss: 0.0108
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.012198
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 54 out of 128 (42.19%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 54 (42.19%)
    Active features: 74 (57.81%)
    Mean activation frequency: 0.0781
    Median activation frequency: 0.0680
    Mean activation strength (active features): 0.0362
    Epoch [1/50], Loss: 0.2108
    Epoch [2/50], Loss: 0.0833
    Epoch [3/50], Loss: 0.0463
    Epoch [4/50], Loss: 0.0291
    Epoch [5/50], Loss: 0.0213
    Epoch [6/50], Loss: 0.0177
    Epoch [7/50], Loss: 0.0157
    Epoch [8/50], Loss: 0.0145
    Epoch [9/50], Loss: 0.0136
    Epoch [10/50], Loss: 0.0130
    Epoch [11/50], Loss: 0.0125
    Epoch [12/50], Loss: 0.0121
    Epoch [13/50], Loss: 0.0117
    Epoch [14/50], Loss: 0.0114
    Epoch [15/50], Loss: 0.0112
    Epoch [16/50], Loss: 0.0110
    Epoch [17/50], Loss: 0.0108
    Epoch [18/50], Loss: 0.0106
    Epoch [19/50], Loss: 0.0105
    Epoch [20/50], Loss: 0.0103
    Epoch [21/50], Loss: 0.0102
    Epoch [22/50], Loss: 0.0101
    Epoch [23/50], Loss: 0.0100
    Epoch [24/50], Loss: 0.0099
    Epoch [25/50], Loss: 0.0098
    Epoch [26/50], Loss: 0.0097
    Epoch [27/50], Loss: 0.0096
    Epoch [28/50], Loss: 0.0095
    Epoch [29/50], Loss: 0.0094
    Epoch [30/50], Loss: 0.0094
    Epoch [31/50], Loss: 0.0093
    Epoch [32/50], Loss: 0.0093
    Epoch [33/50], Loss: 0.0092
    Epoch [34/50], Loss: 0.0091
    Epoch [35/50], Loss: 0.0091
    Epoch [36/50], Loss: 0.0090
    Epoch [37/50], Loss: 0.0090
    Epoch [38/50], Loss: 0.0089
    Epoch [39/50], Loss: 0.0089
    Epoch [40/50], Loss: 0.0089
    Epoch [41/50], Loss: 0.0088
    Epoch [42/50], Loss: 0.0088
    Epoch [43/50], Loss: 0.0088
    Epoch [44/50], Loss: 0.0088
    Epoch [45/50], Loss: 0.0087
    Epoch [46/50], Loss: 0.0087
    Epoch [47/50], Loss: 0.0087
    Epoch [48/50], Loss: 0.0087
    Epoch [49/50], Loss: 0.0087
    Epoch [50/50], Loss: 0.0088
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.010325
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 16 out of 128 (12.50%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 16 (12.50%)
    Active features: 112 (87.50%)
    Mean activation frequency: 0.1562
    Median activation frequency: 0.1644
    Mean activation strength (active features): 0.0324
    Epoch [1/50], Loss: 0.2122
    Epoch [2/50], Loss: 0.0840
    Epoch [3/50], Loss: 0.0466
    Epoch [4/50], Loss: 0.0290
    Epoch [5/50], Loss: 0.0210
    Epoch [6/50], Loss: 0.0174
    Epoch [7/50], Loss: 0.0154
    Epoch [8/50], Loss: 0.0140
    Epoch [9/50], Loss: 0.0131
    Epoch [10/50], Loss: 0.0124
    Epoch [11/50], Loss: 0.0118
    Epoch [12/50], Loss: 0.0114
    Epoch [13/50], Loss: 0.0110
    Epoch [14/50], Loss: 0.0107
    Epoch [15/50], Loss: 0.0104
    Epoch [16/50], Loss: 0.0102
    Epoch [17/50], Loss: 0.0100
    Epoch [18/50], Loss: 0.0098
    Epoch [19/50], Loss: 0.0096
    Epoch [20/50], Loss: 0.0094
    Epoch [21/50], Loss: 0.0093
    Epoch [22/50], Loss: 0.0092
    Epoch [23/50], Loss: 0.0091
    Epoch [24/50], Loss: 0.0089
    Epoch [25/50], Loss: 0.0088
    Epoch [26/50], Loss: 0.0088
    Epoch [27/50], Loss: 0.0087
    Epoch [28/50], Loss: 0.0086
    Epoch [29/50], Loss: 0.0085
    Epoch [30/50], Loss: 0.0084
    Epoch [31/50], Loss: 0.0084
    Epoch [32/50], Loss: 0.0083
    Epoch [33/50], Loss: 0.0082
    Epoch [34/50], Loss: 0.0082
    Epoch [35/50], Loss: 0.0081
    Epoch [36/50], Loss: 0.0081
    Epoch [37/50], Loss: 0.0080
    Epoch [38/50], Loss: 0.0080
    Epoch [39/50], Loss: 0.0079
    Epoch [40/50], Loss: 0.0079
    Epoch [41/50], Loss: 0.0078
    Epoch [42/50], Loss: 0.0078
    Epoch [43/50], Loss: 0.0078
    Epoch [44/50], Loss: 0.0078
    Epoch [45/50], Loss: 0.0077
    Epoch [46/50], Loss: 0.0077
    Epoch [47/50], Loss: 0.0078
    Epoch [48/50], Loss: 0.0078
    Epoch [49/50], Loss: 0.0079
    Epoch [50/50], Loss: 0.0079
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.009213
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 3 out of 128 (2.34%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 3 (2.34%)
    Active features: 125 (97.66%)
    Mean activation frequency: 0.2344
    Median activation frequency: 0.2517
    Mean activation strength (active features): 0.0345
    Epoch [1/50], Loss: 0.2113
    Epoch [2/50], Loss: 0.0844
    Epoch [3/50], Loss: 0.0469
    Epoch [4/50], Loss: 0.0292
    Epoch [5/50], Loss: 0.0212
    Epoch [6/50], Loss: 0.0174
    Epoch [7/50], Loss: 0.0154
    Epoch [8/50], Loss: 0.0140
    Epoch [9/50], Loss: 0.0130
    Epoch [10/50], Loss: 0.0122
    Epoch [11/50], Loss: 0.0115
    Epoch [12/50], Loss: 0.0110
    Epoch [13/50], Loss: 0.0106
    Epoch [14/50], Loss: 0.0102
    Epoch [15/50], Loss: 0.0099
    Epoch [16/50], Loss: 0.0097
    Epoch [17/50], Loss: 0.0094
    Epoch [18/50], Loss: 0.0092
    Epoch [19/50], Loss: 0.0090
    Epoch [20/50], Loss: 0.0088
    Epoch [21/50], Loss: 0.0087
    Epoch [22/50], Loss: 0.0086
    Epoch [23/50], Loss: 0.0084
    Epoch [24/50], Loss: 0.0083
    Epoch [25/50], Loss: 0.0082
    Epoch [26/50], Loss: 0.0081
    Epoch [27/50], Loss: 0.0080
    Epoch [28/50], Loss: 0.0079
    Epoch [29/50], Loss: 0.0078
    Epoch [30/50], Loss: 0.0077
    Epoch [31/50], Loss: 0.0077
    Epoch [32/50], Loss: 0.0076
    Epoch [33/50], Loss: 0.0076
    Epoch [34/50], Loss: 0.0075
    Epoch [35/50], Loss: 0.0075
    Epoch [36/50], Loss: 0.0074
    Epoch [37/50], Loss: 0.0073
    Epoch [38/50], Loss: 0.0073
    Epoch [39/50], Loss: 0.0073
    Epoch [40/50], Loss: 0.0072
    Epoch [41/50], Loss: 0.0072
    Epoch [42/50], Loss: 0.0071
    Epoch [43/50], Loss: 0.0071
    Epoch [44/50], Loss: 0.0070
    Epoch [45/50], Loss: 0.0070
    Epoch [46/50], Loss: 0.0070
    Epoch [47/50], Loss: 0.0069
    Epoch [48/50], Loss: 0.0069
    Epoch [49/50], Loss: 0.0070
    Epoch [50/50], Loss: 0.0071
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.008287
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 4 out of 128 (3.12%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 4 (3.12%)
    Active features: 124 (96.88%)
    Mean activation frequency: 0.3125
    Median activation frequency: 0.3230
    Mean activation strength (active features): 0.0389
    Epoch [1/50], Loss: 0.2155
    Epoch [2/50], Loss: 0.0880
    Epoch [3/50], Loss: 0.0488
    Epoch [4/50], Loss: 0.0303
    Epoch [5/50], Loss: 0.0217
    Epoch [6/50], Loss: 0.0177
    Epoch [7/50], Loss: 0.0155
    Epoch [8/50], Loss: 0.0141
    Epoch [9/50], Loss: 0.0130
    Epoch [10/50], Loss: 0.0121
    Epoch [11/50], Loss: 0.0115
    Epoch [12/50], Loss: 0.0109
    Epoch [13/50], Loss: 0.0105
    Epoch [14/50], Loss: 0.0101
    Epoch [15/50], Loss: 0.0097
    Epoch [16/50], Loss: 0.0094
    Epoch [17/50], Loss: 0.0092
    Epoch [18/50], Loss: 0.0090
    Epoch [19/50], Loss: 0.0087
    Epoch [20/50], Loss: 0.0085
    Epoch [21/50], Loss: 0.0084
    Epoch [22/50], Loss: 0.0082
    Epoch [23/50], Loss: 0.0081
    Epoch [24/50], Loss: 0.0079
    Epoch [25/50], Loss: 0.0078
    Epoch [26/50], Loss: 0.0077
    Epoch [27/50], Loss: 0.0076
    Epoch [28/50], Loss: 0.0075
    Epoch [29/50], Loss: 0.0074
    Epoch [30/50], Loss: 0.0073
    Epoch [31/50], Loss: 0.0072
    Epoch [32/50], Loss: 0.0072
    Epoch [33/50], Loss: 0.0071
    Epoch [34/50], Loss: 0.0070
    Epoch [35/50], Loss: 0.0070
    Epoch [36/50], Loss: 0.0069
    Epoch [37/50], Loss: 0.0069
    Epoch [38/50], Loss: 0.0069
    Epoch [39/50], Loss: 0.0068
    Epoch [40/50], Loss: 0.0067
    Epoch [41/50], Loss: 0.0067
    Epoch [42/50], Loss: 0.0067
    Epoch [43/50], Loss: 0.0066
    Epoch [44/50], Loss: 0.0066
    Epoch [45/50], Loss: 0.0065
    Epoch [46/50], Loss: 0.0065
    Epoch [47/50], Loss: 0.0065
    Epoch [48/50], Loss: 0.0066
    Epoch [49/50], Loss: 0.0066
    Epoch [50/50], Loss: 0.0069
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.007992
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.3906
    Median activation frequency: 0.3881
    Mean activation strength (active features): 0.0424
    Epoch [1/50], Loss: 0.2047
    Epoch [2/50], Loss: 0.0821
    Epoch [3/50], Loss: 0.0460
    Epoch [4/50], Loss: 0.0290
    Epoch [5/50], Loss: 0.0212
    Epoch [6/50], Loss: 0.0174
    Epoch [7/50], Loss: 0.0153
    Epoch [8/50], Loss: 0.0139
    Epoch [9/50], Loss: 0.0128
    Epoch [10/50], Loss: 0.0120
    Epoch [11/50], Loss: 0.0113
    Epoch [12/50], Loss: 0.0108
    Epoch [13/50], Loss: 0.0103
    Epoch [14/50], Loss: 0.0099
    Epoch [15/50], Loss: 0.0095
    Epoch [16/50], Loss: 0.0092
    Epoch [17/50], Loss: 0.0089
    Epoch [18/50], Loss: 0.0087
    Epoch [19/50], Loss: 0.0084
    Epoch [20/50], Loss: 0.0082
    Epoch [21/50], Loss: 0.0080
    Epoch [22/50], Loss: 0.0079
    Epoch [23/50], Loss: 0.0077
    Epoch [24/50], Loss: 0.0076
    Epoch [25/50], Loss: 0.0074
    Epoch [26/50], Loss: 0.0073
    Epoch [27/50], Loss: 0.0072
    Epoch [28/50], Loss: 0.0071
    Epoch [29/50], Loss: 0.0070
    Epoch [30/50], Loss: 0.0069
    Epoch [31/50], Loss: 0.0068
    Epoch [32/50], Loss: 0.0067
    Epoch [33/50], Loss: 0.0067
    Epoch [34/50], Loss: 0.0067
    Epoch [35/50], Loss: 0.0066
    Epoch [36/50], Loss: 0.0065
    Epoch [37/50], Loss: 0.0064
    Epoch [38/50], Loss: 0.0063
    Epoch [39/50], Loss: 0.0063
    Epoch [40/50], Loss: 0.0062
    Epoch [41/50], Loss: 0.0062
    Epoch [42/50], Loss: 0.0062
    Epoch [43/50], Loss: 0.0061
    Epoch [44/50], Loss: 0.0062
    Epoch [45/50], Loss: 0.0063
    Epoch [46/50], Loss: 0.0061
    Epoch [47/50], Loss: 0.0061
    Epoch [48/50], Loss: 0.0060
    Epoch [49/50], Loss: 0.0060
    Epoch [50/50], Loss: 0.0060
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.006773
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.5000
    Median activation frequency: 0.5041
    Mean activation strength (active features): 0.0477
    Epoch [1/50], Loss: 0.2115
    Epoch [2/50], Loss: 0.0861
    Epoch [3/50], Loss: 0.0479
    Epoch [4/50], Loss: 0.0301
    Epoch [5/50], Loss: 0.0219
    Epoch [6/50], Loss: 0.0179
    Epoch [7/50], Loss: 0.0157
    Epoch [8/50], Loss: 0.0142
    Epoch [9/50], Loss: 0.0131
    Epoch [10/50], Loss: 0.0122
    Epoch [11/50], Loss: 0.0115
    Epoch [12/50], Loss: 0.0110
    Epoch [13/50], Loss: 0.0105
    Epoch [14/50], Loss: 0.0100
    Epoch [15/50], Loss: 0.0096
    Epoch [16/50], Loss: 0.0093
    Epoch [17/50], Loss: 0.0089
    Epoch [18/50], Loss: 0.0087
    Epoch [19/50], Loss: 0.0084
    Epoch [20/50], Loss: 0.0082
    Epoch [21/50], Loss: 0.0080
    Epoch [22/50], Loss: 0.0078
    Epoch [23/50], Loss: 0.0076
    Epoch [24/50], Loss: 0.0074
    Epoch [25/50], Loss: 0.0073
    Epoch [26/50], Loss: 0.0072
    Epoch [27/50], Loss: 0.0071
    Epoch [28/50], Loss: 0.0069
    Epoch [29/50], Loss: 0.0067
    Epoch [30/50], Loss: 0.0066
    Epoch [31/50], Loss: 0.0064
    Epoch [32/50], Loss: 0.0065
    Epoch [33/50], Loss: 0.0064
    Epoch [34/50], Loss: 0.0063
    Epoch [35/50], Loss: 0.0065
    Epoch [36/50], Loss: 0.0063
    Epoch [37/50], Loss: 0.0060
    Epoch [38/50], Loss: 0.0059
    Epoch [39/50], Loss: 0.0058
    Epoch [40/50], Loss: 0.0057
    Epoch [41/50], Loss: 0.0056
    Epoch [42/50], Loss: 0.0056
    Epoch [43/50], Loss: 0.0055
    Epoch [44/50], Loss: 0.0055
    Epoch [45/50], Loss: 0.0055
    Epoch [46/50], Loss: 0.0056
    Epoch [47/50], Loss: 0.0056
    Epoch [48/50], Loss: 0.0055
    Epoch [49/50], Loss: 0.0053
    Epoch [50/50], Loss: 0.0052
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.005632
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.7685
    Median activation frequency: 0.8019
    Mean activation strength (active features): 0.0591
    Epoch [1/50], Loss: 0.2101
    Epoch [2/50], Loss: 0.0857
    Epoch [3/50], Loss: 0.0480
    Epoch [4/50], Loss: 0.0302
    Epoch [5/50], Loss: 0.0220
    Epoch [6/50], Loss: 0.0180
    Epoch [7/50], Loss: 0.0158
    Epoch [8/50], Loss: 0.0143
    Epoch [9/50], Loss: 0.0131
    Epoch [10/50], Loss: 0.0122
    Epoch [11/50], Loss: 0.0115
    Epoch [12/50], Loss: 0.0109
    Epoch [13/50], Loss: 0.0104
    Epoch [14/50], Loss: 0.0099
    Epoch [15/50], Loss: 0.0096
    Epoch [16/50], Loss: 0.0092
    Epoch [17/50], Loss: 0.0089
    Epoch [18/50], Loss: 0.0086
    Epoch [19/50], Loss: 0.0084
    Epoch [20/50], Loss: 0.0082
    Epoch [21/50], Loss: 0.0079
    Epoch [22/50], Loss: 0.0077
    Epoch [23/50], Loss: 0.0075
    Epoch [24/50], Loss: 0.0074
    Epoch [25/50], Loss: 0.0072
    Epoch [26/50], Loss: 0.0071
    Epoch [27/50], Loss: 0.0070
    Epoch [28/50], Loss: 0.0068
    Epoch [29/50], Loss: 0.0067
    Epoch [30/50], Loss: 0.0066
    Epoch [31/50], Loss: 0.0065
    Epoch [32/50], Loss: 0.0064
    Epoch [33/50], Loss: 0.0066
    Epoch [34/50], Loss: 0.0064
    Epoch [35/50], Loss: 0.0061
    Epoch [36/50], Loss: 0.0060
    Epoch [37/50], Loss: 0.0059
    Epoch [38/50], Loss: 0.0058
    Epoch [39/50], Loss: 0.0057
    Epoch [40/50], Loss: 0.0058
    Epoch [41/50], Loss: 0.0060
    Epoch [42/50], Loss: 0.0058
    Epoch [43/50], Loss: 0.0057
    Epoch [44/50], Loss: 0.0062
    Epoch [45/50], Loss: 0.0059
    Epoch [46/50], Loss: 0.0054
    Epoch [47/50], Loss: 0.0053
    Epoch [48/50], Loss: 0.0052
    Epoch [49/50], Loss: 0.0051
    Epoch [50/50], Loss: 0.0051
    Finished Training
    Test Loss for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0.005525
    Number of dead neurons in Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss: 0 out of 128 (0.00%)
    
    === Activation Statistics for Sparse Autoencoder with weight init., JumpReLU and Auxiliary Loss ===
    Total features: 128
    Dead features: 0 (0.00%)
    Active features: 128 (100.00%)
    Mean activation frequency: 0.8046
    Median activation frequency: 0.8385
    Mean activation strength (active features): 0.0613


Plotting all those results


```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_k_sensitivity_results(results, save_figs=True):
    """
    Comprehensive plotting function for K sensitivity sweep results.

    Parameters:
    -----------
    results : list of dict
        Results from K sensitivity sweep, each dict containing:
        - dataset: str (e.g., 'mnist', 'olivetti', 'lfw')
        - k: int (top-K value)
        - sparsity_ratio: float (k/hidden_size)
        - test_mse: float
        - dead_neurons: int
        - active_features: int
        - mean_activation_freq: float

    save_figs : bool
        Whether to save figures to disk
    """

    df = pd.DataFrame(results)
    datasets = df['dataset'].unique()
    colors = {'mnist': '#1f77b4', 'olivetti': '#2ca02c', 'lfw': '#d62728'}
    markers = {'mnist': 'o', 'olivetti': 's', 'lfw': '^'}

    print(datasets)
    # ===== MAIN FIGURE: 4-panel comparison =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Sparsity Sensitivity Analysis Across Datasets',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Test MSE vs K
    for dataset in datasets:
        data = df[df['dataset'] == dataset].sort_values('k')
        axes[0, 0].plot(data['k'], data['test_mse'],
                       marker=markers[dataset], color=colors[dataset],
                       label=dataset.upper(), linewidth=2.5, markersize=8, alpha=0.8)
    axes[0, 0].set_xlabel('K (Top-K Activations)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Test MSE', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Reconstruction Error', fontsize=12)
    axes[0, 0].legend(framealpha=0.9)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Dead Neurons vs K
    for dataset in datasets:
        data = df[df['dataset'] == dataset].sort_values('k')
        axes[0, 1].plot(data['k'], data['dead_neurons'],
                       marker=markers[dataset], color=colors[dataset],
                       label=dataset.upper(), linewidth=2.5, markersize=8, alpha=0.8)
    axes[0, 1].set_xlabel('K (Top-K Activations)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Dead Neurons', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Feature Utilization', fontsize=12)
    axes[0, 1].legend(framealpha=0.9)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')

    # Plot 3: Active Features vs K
    for dataset in datasets:
        data = df[df['dataset'] == dataset].sort_values('k')
        axes[1, 0].plot(data['k'], data['active_features'],
                       marker=markers[dataset], color=colors[dataset],
                       label=dataset.upper(), linewidth=2.5, markersize=8, alpha=0.8)
    axes[1, 0].set_xlabel('K (Top-K Activations)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Active Features', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Feature Diversity', fontsize=12)
    axes[1, 0].legend(framealpha=0.9)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')

    # Plot 4: Mean Activation Frequency vs K
    for dataset in datasets:
        data = df[df['dataset'] == dataset].sort_values('k')
        axes[1, 1].plot(data['k'], data['mean_activation_freq'],
                       marker=markers[dataset], color=colors[dataset],
                       label=dataset.upper(), linewidth=2.5, markersize=8, alpha=0.8)
    axes[1, 1].set_xlabel('K (Top-K Activations)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Mean Activation Frequency', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Activation Distribution', fontsize=12)
    axes[1, 1].legend(framealpha=0.9)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    if save_figs:
        plt.savefig('k_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ===== SPARSITY RATIO FIGURE =====
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    for dataset in datasets:
        data = df[df['dataset'] == dataset].sort_values('sparsity_ratio')
        ax.plot(data['sparsity_ratio'], data['test_mse'],
               marker=markers[dataset], color=colors[dataset],
               label=dataset.upper(), linewidth=2.5, markersize=8, alpha=0.8)

    ax.set_xlabel('Sparsity Ratio (K/Hidden Size)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
    ax.set_title('Reconstruction Error vs Sparsity Ratio', fontsize=14, fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_figs:
        plt.savefig('sparsity_ratio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ===== PER-DATASET DETAILED ANALYSIS =====
    for dataset in datasets:
        data = df[df['dataset'] == dataset].sort_values('k')

        fig3, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig3.suptitle(f'{dataset.upper()} Dataset: Detailed K Analysis',
                     fontsize=14, fontweight='bold')

        # MSE trend
        axes[0, 0].plot(data['k'], data['test_mse'],
                       marker='o', color=colors[dataset], linewidth=2.5, markersize=8)
        axes[0, 0].fill_between(data['k'], data['test_mse'], alpha=0.2, color=colors[dataset])
        axes[0, 0].set_xlabel('K', fontsize=10)
        axes[0, 0].set_ylabel('Test MSE', fontsize=10)
        axes[0, 0].set_title('Reconstruction Quality', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # Feature utilization
        axes[0, 1].plot(data['k'], data['dead_neurons'],
                       marker='s', color='red', linewidth=2.5, markersize=8, label='Dead')
        axes[0, 1].plot(data['k'], data['active_features'],
                       marker='^', color='green', linewidth=2.5, markersize=8, label='Active')
        axes[0, 1].set_xlabel('K', fontsize=10)
        axes[0, 1].set_ylabel('Count', fontsize=10)
        axes[0, 1].set_title('Feature Statistics', fontsize=11)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Sparsity relationship
        axes[1, 0].scatter(data['sparsity_ratio'], data['test_mse'],
                          s=100, alpha=0.6, color=colors[dataset])
        axes[1, 0].plot(data['sparsity_ratio'], data['test_mse'],
                       linewidth=2, alpha=0.5, color=colors[dataset])
        axes[1, 0].set_xlabel('Sparsity Ratio', fontsize=10)
        axes[1, 0].set_ylabel('Test MSE', fontsize=10)
        axes[1, 0].set_title('Sparsity vs Error', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        # Activation frequency
        axes[1, 1].bar(range(len(data)), data['mean_activation_freq'],
                      color=colors[dataset], alpha=0.7)
        axes[1, 1].set_xlabel('K Value Index', fontsize=10)
        axes[1, 1].set_ylabel('Mean Activation Frequency', fontsize=10)
        axes[1, 1].set_title('Activation Distribution', fontsize=11)
        axes[1, 1].set_xticks(range(len(data)))
        axes[1, 1].set_xticklabels(data['k'].values, rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_figs:
            plt.savefig(f'{dataset}_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

```


```python
plot_k_sensitivity_results(results_complete, save_figs=True)
```

    ['mnist' 'olivetti' 'lfw']



    
![png](Sparse%20Autoencoder%20Research_files/Sparse%20Autoencoder%20Research_38_1.png)
    



    
![png](Sparse%20Autoencoder%20Research_files/Sparse%20Autoencoder%20Research_38_2.png)
    



    
![png](Sparse%20Autoencoder%20Research_files/Sparse%20Autoencoder%20Research_38_3.png)
    



    
![png](Sparse%20Autoencoder%20Research_files/Sparse%20Autoencoder%20Research_38_4.png)
    



    
![png](Sparse%20Autoencoder%20Research_files/Sparse%20Autoencoder%20Research_38_5.png)
    

