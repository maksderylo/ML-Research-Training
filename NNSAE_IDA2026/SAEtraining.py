import torch
from torch import optim

# python
def train_sparse_autoencoder(train_loader, dataset_type, model, num_epochs=50, learning_rate=0.001, use_weight_clamping=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_data = []

    for epoch in range(num_epochs):
        running_recon_loss = 0.0
        active_neurons = set()

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

            inputs = inputs.to(device)
            optimizer.zero_grad()

            h, outputs = model(inputs)

            loss = model.compute_loss(inputs, h, outputs)
            loss.backward()
            optimizer.step()

            # Weight clamping (optional)
            if use_weight_clamping:
                with torch.no_grad():
                    model.decoder.weight.clamp_(0.0)

            running_recon_loss += loss.item()
            #running_sparsity_loss += sparsity_loss.item()
            active_neurons.update(torch.where(h > 0)[1].cpu().numpy())

        # gather training data
        avg_recon_loss = running_recon_loss / len(train_loader)
        num_active_neurons = len(active_neurons)
        training_data.append((epoch, avg_recon_loss, num_active_neurons))

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Recon: {avg_recon_loss:.4f}, '
              #f'L1: {running_sparsity_loss / len(train_loader):.2f}, '
              f'Active: {num_active_neurons}')

    return model, training_data
