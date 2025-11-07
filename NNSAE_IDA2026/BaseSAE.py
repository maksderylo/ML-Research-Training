# Maks Derylo

#%% md
# SAE implementation - Most basic TopK
#%%
import torch.nn as nn
from typing import Callable
import math
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, k_top: int,
                 activation: Callable[[torch.tensor], torch.tensor]):
        super(SparseAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k_top = k_top
        self.activation = activation
        # Encoder maps input to hidden representation
        self.encoder = nn.Linear(input_size, hidden_size)

        # Decoder maps hidden representation back to input space
        self.decoder = nn.Linear(hidden_size, input_size)

        # Initialize encoder weights first with random directions
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        # Initialize the decoder to be the transpose of the encoder weights
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())


    def _topk_mask(self, activations: torch.Tensor) -> torch.Tensor:
        # activations: (batch, hidden)
        k = max(0, min(self.k_top, activations.size(1)))
        _, idx = torch.topk(activations, k, dim=1)
        mask = torch.zeros_like(activations)
        mask.scatter_(1, idx, 1.0)
        return mask

    def forward(self, x):
        pre_activations = self.encoder(x)
        pre_activations = self.activation(pre_activations)
        mask = self._topk_mask(pre_activations)
        h = pre_activations * mask
        x_hat = self.decoder(h)
        return h, x_hat


    def compute_loss(self, x, h, x_hat):
        # Compute sum of squares and normalize by batch size
        recon_loss = torch.sum((x - x_hat) ** 2) / (x.size(0))

        return recon_loss



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
