import torch
from torch import optim
import inspect


def train_sparse_autoencoder(train_loader, dataset_type, model, num_epochs=50,
                             learning_rate=0.001, use_weight_clamping=True,
                             auxk_coef=1 / 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_data = []

    for epoch in range(num_epochs):
        running_recon_loss = 0.0
        running_auxk_loss = 0.0
        active_neurons = set()
        total_dead_features = 0

        for batch_idx, data in enumerate(train_loader):
            # Handle different data loader formats
            if dataset_type in ['olivetti', 'lfw']:
                inputs, = data
                inputs = inputs.to(device)
            elif dataset_type in ['mnist', 'imagenet']:
                inputs, _ = data
                inputs = inputs.to(device)
            else:
                raise ValueError(f"Unknown dataset_type: {dataset_type}")

            optimizer.zero_grad()

            h, x_hat, pre_activations = model(inputs)

            loss, loss_dict = model.compute_loss(inputs, h, x_hat, pre_activations, auxk_coef=auxk_coef)

            loss.backward()
            optimizer.step()

            # Weight clamping for nonnegativity (optional)
            if use_weight_clamping:
                with torch.no_grad():
                    model.decoder.weight.clamp_(0.0)

            running_recon_loss += loss_dict['recon_loss']
            running_auxk_loss += loss_dict.get('auxk_loss', 0.0)
            total_dead_features += loss_dict.get('num_dead', 0)

            # Track active neurons
            active_indices = torch.where(h.abs() > 1e-5)[1].cpu().numpy()
            active_neurons.update(active_indices)

        # Gather training data
        avg_recon_loss = running_recon_loss / len(train_loader)
        avg_auxk_loss = running_auxk_loss / len(train_loader)
        num_active_neurons = len(active_neurons)
        avg_dead_features = total_dead_features / len(train_loader)

        training_data.append((epoch, avg_recon_loss, num_active_neurons, avg_auxk_loss, avg_dead_features))

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Recon: {avg_recon_loss:.4f}, '
              f'AuxK: {avg_auxk_loss:.4f}, '
              f'Active: {num_active_neurons}/{model.hidden_size if hasattr(model, "hidden_size") else "?"} , '
              f'Dead: {avg_dead_features:.1f}')

    return model, training_data
