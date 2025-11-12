# In NNSAE_IDA2026/KLSAETraining.py

import torch
from torch import optim

def train_kl_sae(train_loader, dataset_type, model, num_epochs=50, learning_rate=0.001, use_weight_clamping=False, warmup_epochs=0, auxk_coef=1/32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_data = []

    for epoch in range(num_epochs):
        running_recon_loss = 0.0
        running_sparse_loss = 0.0
        running_auxk_loss = 0.0
        total_dead_neurons = 0
        active_neurons = set()

        for data in train_loader:
            if dataset_type in ['olivetti', 'lfw']:
                inputs, = data
                inputs = inputs.to(device)
            elif dataset_type in ['mnist', 'imagenet']:
                inputs, _ = data
                inputs = inputs.to(device)
            else:
                raise ValueError(f"Unknown dataset_type: {dataset_type}")

            optimizer.zero_grad()

            # Updated forward call
            h, x_hat, pre_activations, h_aux = model(inputs)

            is_warmup = epoch < warmup_epochs
            loss, loss_dict = model.compute_loss(
                inputs, h, x_hat, pre_activations, h_aux,
                warmup=is_warmup,
                auxk_coef=auxk_coef
            )
            loss.backward()
            optimizer.step()

            if use_weight_clamping:
                with torch.no_grad():
                    model.decoder.weight.clamp_(0.0)

            running_recon_loss += loss_dict['recon_loss']
            running_sparse_loss += loss_dict.get('sparse_loss', 0.0)
            running_auxk_loss += loss_dict.get('auxk_loss', 0.0)
            total_dead_neurons += loss_dict.get('num_dead', 0)
            active_neurons.update(torch.where(h > 1e-5)[1].cpu().numpy())

        avg_recon_loss = running_recon_loss / len(train_loader)
        avg_sparse_loss = running_sparse_loss / len(train_loader)
        avg_auxk_loss = running_auxk_loss / len(train_loader)
        avg_dead_neurons = total_dead_neurons / len(train_loader)
        num_active_neurons = len(active_neurons)
        training_data.append((epoch, avg_recon_loss, num_active_neurons, avg_sparse_loss, avg_auxk_loss, avg_dead_neurons))

        print_str = f'Epoch [{epoch + 1}/{num_epochs}], Recon: {avg_recon_loss:.4f}, Active: {num_active_neurons}/{model.hidden_size}'
        if not is_warmup:
            penalty_name = 'L1' if model.penalty_type == 'l1' else 'KL'
            print_str += f', {penalty_name}: {avg_sparse_loss:.4f}'
            if model.penalty_type == 'l1':
                 print_str += f', AuxK: {avg_auxk_loss:.4f}, Dead: {avg_dead_neurons:.1f}'
        else:
            print_str += ' (Warmup)'
        print(print_str)

    return model, training_data
