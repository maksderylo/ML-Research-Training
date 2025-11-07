import torch
from torch import nn

from ..BaseSAE import SparseAutoencoder
from ..DataLoader import load_data
from ..SAEtraining import train_sparse_autoencoder

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SparseAutoencoder(input_size=784, hidden_size=50, k_top=10, activation=nn.ReLU())
    model.to(device)

    train_loader, test_loader, _ = load_data('mnist', batch_size=128)
    trained_model = train_sparse_autoencoder(train_loader, 'mnist', model, num_epochs=50, learning_rate=0.001,
                                            warmup_epochs=10, use_weight_clamping=True)
