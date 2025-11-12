import torch
from torch import nn

from NNSAE_IDA2026.BaseSAE import SparseAutoencoder
from NNSAE_IDA2026.DataLoader import load_data
from NNSAE_IDA2026.SAEtraining import train_sparse_autoencoder
from NNSAE_IDA2026.ModelVisualizations import SAEVisualizer

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    folder = 'basic_training/50epochs_clamping_10ktop_relu_50hidden_bigbatch/'

    # --- Hyperparameters ---
    hyperparameters = {
        'input_size': 784,
        'hidden_size': 50,
        'k_top': 10,
        'activation': 'ReLU',
        'dataset': 'mnist',
        'batch_size': 1024,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'use_weight_clamping': True,
    }

    model = SparseAutoencoder(
        input_size=hyperparameters['input_size'],
        hidden_size=hyperparameters['hidden_size'],
        k_top=hyperparameters['k_top'],
        activation=nn.ReLU()
    )
    model.to(device)

    train_loader, test_loader = load_data(hyperparameters['dataset'], batch_size=hyperparameters['batch_size'])
    trained_model, training_data = train_sparse_autoencoder(
        train_loader,
        hyperparameters['dataset'],
        model,
        num_epochs=hyperparameters['num_epochs'],
        learning_rate=hyperparameters['learning_rate'],
        use_weight_clamping=hyperparameters['use_weight_clamping']
    )

    visualizer = SAEVisualizer(trained_model)

    visualizer.plot_training_history(training_data, folder + "training_history.png", hyperparameters=hyperparameters)
    visualizer.visualize_reconstructions(test_loader, "mnist", 10, folder + "example_reconstructions.png")
    visualizer.visualize_decoder_weights(64, None, folder + "visualized_decoder_longer_weights.png", cluster_by_similarity=True, dendrogram_path=folder + "decoder_dendrogram.png")
