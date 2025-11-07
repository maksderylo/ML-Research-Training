#!/usr/bin/env python3
"""
Experiment 4: Effect of K on Reconstruction Quality and Parts-Based Representation
Investigates the sparsity-reconstruction tradeoff by varying k_top parameter
Includes both quantitative (MSE) and qualitative (interpretability) analysis
"""

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for cluster

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from PIL import Image
import math
import os
import random


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class SparseAutoencoder(nn.Module):
    """TopK Sparse Autoencoder"""

    def __init__(self, input_size=784, hidden_size=64, k_top=20):
        super(SparseAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k_top = k_top
        self.name = f"TopK-SAE (k={k_top})"

        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

        # Initialize with tied weights
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

    def _topk_mask(self, activations):
        k = max(0, min(self.k_top, activations.size(1)))
        _, idx = torch.topk(activations, k, dim=1)
        mask = torch.zeros_like(activations)
        mask.scatter_(1, idx, 1.0)
        return mask

    def forward(self, x):
        pre_activations = self.encoder(x)
        pre_activations = torch.nn.functional.relu(pre_activations)
        mask = self._topk_mask(pre_activations)
        h = pre_activations * mask
        x_hat = self.decoder(h)
        return h, x_hat

    def compute_loss(self, x, h, x_hat):
        recon_loss = torch.mean((x - x_hat) ** 2)
        return recon_loss


class CompleteAutoencoder(nn.Module):
    """Dense autoencoder (no sparsity) for comparison"""

    def __init__(self, input_size=784, hidden_size=64):
        super(CompleteAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = "Complete (No Sparsity)"

        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

    def forward(self, x):
        h = torch.nn.functional.relu(self.encoder(x))
        x_hat = self.decoder(h)
        return h, x_hat

    def compute_loss(self, x, h, x_hat):
        return torch.mean((x - x_hat) ** 2)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mnist_data(batch_size=256):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_olivetti_data(batch_size=32):
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    data_tensor = torch.FloatTensor(faces.data)
    dataset = TensorDataset(data_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_lfw_data(batch_size=128, img_size=64):
    lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=False)

    resized_images = []
    for img in lfw_people.images:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((img_size, img_size), Image.LANCZOS)
        resized = np.array(pil_img).astype(np.float32) / 255.0
        resized_images.append(resized.flatten())

    data_tensor = torch.FloatTensor(np.array(resized_images))

    class LFWDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return (self.data[idx],)

    dataset = LFWDataset(data_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_loader, dataset_type, num_epochs=50, learning_rate=0.001):
    """Train sparse autoencoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data in train_loader:
            if dataset_type in ['olivetti', 'lfw']:
                inputs, = data
            else:
                inputs, _ = data

            inputs = inputs.to(device)
            if len(inputs.shape) == 4:
                inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            h, outputs = model(inputs)
            loss = model.compute_loss(inputs, h, outputs)
            loss.backward()
            optimizer.step()

            # Enforce non-negativity
            with torch.no_grad():
                model.encoder.weight.clamp_(0.0)
                model.decoder.weight.clamp_(0.0)

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.6f}')

    return model


def compute_reconstruction_metrics(model, test_loader, dataset_type):
    """Compute MSE and L0 (average sparsity)"""
    device = next(model.parameters()).device
    model.eval()

    total_mse = 0.0
    total_l0 = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            if dataset_type in ['olivetti', 'lfw']:
                inputs, = data
            else:
                inputs, _ = data

            inputs = inputs.to(device)
            if len(inputs.shape) == 4:
                inputs = inputs.view(inputs.size(0), -1)

            h, x_hat = model(inputs)

            # MSE
            mse = torch.mean((inputs - x_hat) ** 2, dim=1)
            total_mse += mse.sum().item()

            # L0: number of active features per sample
            l0 = (h > 0).sum(dim=1).float()
            total_l0 += l0.sum().item()

            total_samples += inputs.size(0)

    avg_mse = total_mse / total_samples
    avg_l0 = total_l0 / total_samples

    return avg_mse, avg_l0


def compute_explained_variance(model, test_loader, dataset_type):
    """Compute explained variance (1 - MSE/Var(X))"""
    device = next(model.parameters()).device
    model.eval()

    all_inputs = []
    all_reconstructions = []

    with torch.no_grad():
        for data in test_loader:
            if dataset_type in ['olivetti', 'lfw']:
                inputs, = data
            else:
                inputs, _ = data

            inputs = inputs.to(device)
            if len(inputs.shape) == 4:
                inputs = inputs.view(inputs.size(0), -1)

            h, x_hat = model(inputs)

            all_inputs.append(inputs.cpu())
            all_reconstructions.append(x_hat.cpu())

    all_inputs = torch.cat(all_inputs, dim=0)
    all_reconstructions = torch.cat(all_reconstructions, dim=0)

    var_x = torch.var(all_inputs)
    mse = torch.mean((all_inputs - all_reconstructions) ** 2)
    explained_var = 1.0 - (mse / var_x)

    return explained_var.item()


# ============================================================================
# INTERPRETABILITY ANALYSIS
# ============================================================================

def visualize_decoder_features_grid(models_dict, dataset_name, num_features_per_model=9):
    """
    Visualize decoder features for multiple models in a grid
    Each row shows features from one model (one K value)
    """
    num_models = len(models_dict)

    fig, axes = plt.subplots(num_models, num_features_per_model,
                             figsize=(num_features_per_model * 2, num_models * 2))

    if num_models == 1:
        axes = axes.reshape(1, -1)

    # Determine image shape
    if dataset_name == 'mnist':
        img_shape = (28, 28)
    else:
        img_shape = (64, 64)

    for row_idx, (model_name, model) in enumerate(sorted(models_dict.items())):
        decoder_weights = model.decoder.weight.data.cpu().numpy()

        # Select most representative features (highest L2 norm)
        feature_norms = np.linalg.norm(decoder_weights, axis=0)
        top_features = np.argsort(feature_norms)[-num_features_per_model:][::-1]

        for col_idx, feature_idx in enumerate(top_features):
            feature_vector = decoder_weights[:, feature_idx]
            feature_img = feature_vector.reshape(img_shape)

            # Normalize
            feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min() + 1e-8)

            axes[row_idx, col_idx].imshow(feature_img, cmap='gray')
            axes[row_idx, col_idx].axis('off')

            if col_idx == 0:
                # Extract k value from model name
                if 'k=' in model_name:
                    k_val = model_name.split('k=')[1].split(')')[0]
                    axes[row_idx, col_idx].set_ylabel(f'k={k_val}',
                                                      fontsize=12, rotation=0,
                                                      labelpad=30, va='center')
                else:
                    axes[row_idx, col_idx].set_ylabel('Dense',
                                                      fontsize=12, rotation=0,
                                                      labelpad=30, va='center')

    plt.suptitle(f'Learned Features Across Sparsity Levels: {dataset_name.upper()}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def compute_feature_interpretability_metrics(model, test_loader, dataset_type, top_k_examples=100):
    """
    Compute monosemanticity scores for all features
    Returns mean and std of monosemanticity across features
    """
    device = next(model.parameters()).device
    model.eval()

    # Collect all activations
    all_activations = []
    all_inputs = []

    with torch.no_grad():
        for data in test_loader:
            if dataset_type in ['olivetti', 'lfw']:
                inputs, = data
            else:
                inputs, _ = data

            inputs = inputs.to(device)
            if len(inputs.shape) == 4:
                inputs = inputs.view(inputs.size(0), -1)

            h, _ = model(inputs)

            all_activations.append(h.cpu())
            all_inputs.append(inputs.cpu())

    all_activations = torch.cat(all_activations, dim=0)  # (N, hidden_size)
    all_inputs = torch.cat(all_inputs, dim=0)  # (N, input_size)

    # For each feature, compute monosemanticity
    hidden_size = all_activations.size(1)
    monosemanticity_scores = []

    active_features = (all_activations.sum(dim=0) > 0).numpy()

    for feat_idx in range(hidden_size):
        if not active_features[feat_idx]:
            continue

        # Get activations for this feature
        feat_acts = all_activations[:, feat_idx]

        # Find top-k activating samples
        nonzero_mask = feat_acts > 0
        if nonzero_mask.sum() < 10:  # Too few activations
            continue

        nonzero_acts = feat_acts[nonzero_mask]
        nonzero_inputs = all_inputs[nonzero_mask]

        if len(nonzero_acts) < top_k_examples:
            top_k_actual = len(nonzero_acts)
        else:
            top_k_actual = top_k_examples

        top_k_indices = torch.argsort(nonzero_acts, descending=True)[:top_k_actual]
        top_inputs = nonzero_inputs[top_k_indices]

        # Compute pairwise cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            top_inputs.unsqueeze(1),
            top_inputs.unsqueeze(0),
            dim=2
        )

        # Average similarity (excluding diagonal)
        mask = ~torch.eye(similarities.size(0), dtype=torch.bool)
        avg_similarity = similarities[mask].mean().item()

        monosemanticity_scores.append(avg_similarity)

    if len(monosemanticity_scores) == 0:
        return 0.0, 0.0, 0

    return np.mean(monosemanticity_scores), np.std(monosemanticity_scores), len(monosemanticity_scores)


# ============================================================================
# EXPERIMENT 4: EFFECT OF K ON RECONSTRUCTION AND INTERPRETABILITY
# ============================================================================

def run_experiment_4():
    """
    Main experiment: Vary K and measure reconstruction quality + interpretability
    """
    print("=" * 80)
    print("EXPERIMENT 4: EFFECT OF K ON RECONSTRUCTION AND INTERPRETABILITY")
    print("=" * 80)

    # Configuration
    datasets = {
        'mnist': {'input_size': 784, 'batch_size': 256, 'img_shape': (28, 28)},
        'olivetti': {'input_size': 4096, 'batch_size': 32, 'img_shape': (64, 64)},
        'lfw': {'input_size': 4096, 'batch_size': 128, 'img_shape': (64, 64)}
    }

    hidden_size = 256
    num_epochs = 50
    learning_rate = 0.001

    # K values to test (sparsity levels from very sparse to dense)
    k_values = [5, 10, 20, 40, 64, 128, 256]  # 256 = dense (all features)

    results = defaultdict(list)
    trained_models = {}

    for dataset_name, config in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Processing {dataset_name.upper()} Dataset")
        print(f"{'=' * 80}")

        # Load data
        if dataset_name == 'mnist':
            train_loader, test_loader = load_mnist_data(config['batch_size'])
        elif dataset_name == 'olivetti':
            train_loader, test_loader = load_olivetti_data(config['batch_size'])
        else:
            train_loader, test_loader = load_lfw_data(config['batch_size'])

        dataset_models = {}

        # Train Complete model (no sparsity baseline)
        print(f"\nTraining Complete (Dense) Model...")
        set_seeds(42)
        complete_model = CompleteAutoencoder(input_size=config['input_size'],
                                             hidden_size=hidden_size)
        complete_model = train_model(complete_model, train_loader, dataset_name,
                                     num_epochs, learning_rate)

        mse, l0 = compute_reconstruction_metrics(complete_model, test_loader, dataset_name)
        explained_var = compute_explained_variance(complete_model, test_loader, dataset_name)
        mono_mean, mono_std, num_active = compute_feature_interpretability_metrics(
            complete_model, test_loader, dataset_name
        )

        results['dataset'].append(dataset_name)
        results['k'].append(hidden_size)  # All features active
        results['model_type'].append('Complete')
        results['mse'].append(mse)
        results['l0'].append(l0)
        results['explained_variance'].append(explained_var)
        results['monosemanticity_mean'].append(mono_mean)
        results['monosemanticity_std'].append(mono_std)
        results['num_active_features'].append(num_active)

        dataset_models['Complete'] = complete_model

        print(f"  MSE: {mse:.6f}, L0: {l0:.2f}, Explained Var: {explained_var:.4f}")
        print(f"  Monosemanticity: {mono_mean:.4f} ± {mono_std:.4f} ({num_active} features)")

        # Train TopK SAE for each K value
        for k in k_values:
            if k >= hidden_size:
                continue  # Skip if k >= hidden_size

            print(f"\nTraining TopK-SAE with k={k}...")
            set_seeds(42)

            model = SparseAutoencoder(input_size=config['input_size'],
                                      hidden_size=hidden_size,
                                      k_top=k)
            model = train_model(model, train_loader, dataset_name, num_epochs, learning_rate)

            # Evaluate
            mse, l0 = compute_reconstruction_metrics(model, test_loader, dataset_name)
            explained_var = compute_explained_variance(model, test_loader, dataset_name)
            mono_mean, mono_std, num_active = compute_feature_interpretability_metrics(
                model, test_loader, dataset_name
            )

            results['dataset'].append(dataset_name)
            results['k'].append(k)
            results['model_type'].append('TopK-SAE')
            results['mse'].append(mse)
            results['l0'].append(l0)
            results['explained_variance'].append(explained_var)
            results['monosemanticity_mean'].append(mono_mean)
            results['monosemanticity_std'].append(mono_std)
            results['num_active_features'].append(num_active)

            dataset_models[f'k={k}'] = model

            print(f"  MSE: {mse:.6f}, L0: {l0:.2f}, Explained Var: {explained_var:.4f}")
            print(f"  Monosemanticity: {mono_mean:.4f} ± {mono_std:.4f} ({num_active} features)")

        # Store models for visualization
        trained_models[dataset_name] = dataset_models

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('experiment4_k_effects_results.csv', index=False)
    print("\n✓ Results saved to 'experiment4_k_effects_results.csv'")

    return df, trained_models


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_figures(df, trained_models):
    """Create publication-quality figures"""

    # Figure 1: Sparsity-Reconstruction Tradeoff (L0 vs MSE)
    print("\nGenerating Figure 1: Sparsity-Reconstruction Tradeoff...")
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, dataset in enumerate(['mnist', 'olivetti', 'lfw']):
        df_dataset = df[df['dataset'] == dataset]

        # Plot TopK points
        topk_data = df_dataset[df_dataset['model_type'] == 'TopK-SAE']
        axes[idx].plot(topk_data['l0'], topk_data['mse'],
                       'o-', linewidth=2, markersize=8,
                       label='TopK-SAE', color='steelblue')

        # Plot Complete point
        complete_data = df_dataset[df_dataset['model_type'] == 'Complete']
        axes[idx].plot(complete_data['l0'], complete_data['mse'],
                       's', markersize=12, label='Dense (No Sparsity)',
                       color='coral', zorder=10)

        # Annotate K values
        for _, row in topk_data.iterrows():
            axes[idx].annotate(f"k={int(row['k'])}",
                               (row['l0'], row['mse']),
                               textcoords="offset points",
                               xytext=(0, 10), ha='center', fontsize=8)

        axes[idx].set_xlabel('Average L0 (Active Features)', fontsize=12)
        axes[idx].set_ylabel('MSE (Test)', fontsize=12)
        axes[idx].set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    fig1.savefig('figure_exp4_1_sparsity_reconstruction_tradeoff.png', dpi=300, bbox_inches='tight')
    fig1.savefig('figure_exp4_1_sparsity_reconstruction_tradeoff.pdf', bbox_inches='tight')
    print("  ✓ Saved: figure_exp4_1_sparsity_reconstruction_tradeoff.png/pdf")

    # Figure 2: Explained Variance vs Sparsity
    print("\nGenerating Figure 2: Explained Variance vs Sparsity...")
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, dataset in enumerate(['mnist', 'olivetti', 'lfw']):
        df_dataset = df[df['dataset'] == dataset]

        topk_data = df_dataset[df_dataset['model_type'] == 'TopK-SAE']
        axes[idx].plot(topk_data['k'], topk_data['explained_variance'],
                       'o-', linewidth=2, markersize=8,
                       label='TopK-SAE', color='forestgreen')

        complete_data = df_dataset[df_dataset['model_type'] == 'Complete']
        axes[idx].axhline(complete_data['explained_variance'].values[0],
                          linestyle='--', color='coral', linewidth=2,
                          label='Dense Baseline')

        axes[idx].set_xlabel('K (Sparsity Level)', fontsize=12)
        axes[idx].set_ylabel('Explained Variance', fontsize=12)
        axes[idx].set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(alpha=0.3)
        axes[idx].set_xscale('log')

    plt.tight_layout()
    fig2.savefig('figure_exp4_2_explained_variance.png', dpi=300, bbox_inches='tight')
    fig2.savefig('figure_exp4_2_explained_variance.pdf', bbox_inches='tight')
    print("  ✓ Saved: figure_exp4_2_explained_variance.png/pdf")

    # Figure 3: Monosemanticity vs Sparsity
    print("\nGenerating Figure 3: Monosemanticity vs Sparsity...")
    fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, dataset in enumerate(['mnist', 'olivetti', 'lfw']):
        df_dataset = df[df['dataset'] == dataset]

        topk_data = df_dataset[df_dataset['model_type'] == 'TopK-SAE']
        axes[idx].errorbar(topk_data['k'], topk_data['monosemanticity_mean'],
                           yerr=topk_data['monosemanticity_std'],
                           fmt='o-', linewidth=2, markersize=8, capsize=5,
                           label='TopK-SAE', color='purple')

        complete_data = df_dataset[df_dataset['model_type'] == 'Complete']
        axes[idx].axhline(complete_data['monosemanticity_mean'].values[0],
                          linestyle='--', color='coral', linewidth=2,
                          label='Dense Baseline')

        axes[idx].set_xlabel('K (Sparsity Level)', fontsize=12)
        axes[idx].set_ylabel('Monosemanticity Score', fontsize=12)
        axes[idx].set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(alpha=0.3)
        axes[idx].set_xscale('log')

    plt.tight_layout()
    fig3.savefig('figure_exp4_3_monosemanticity.png', dpi=300, bbox_inches='tight')
    fig3.savefig('figure_exp4_3_monosemanticity.pdf', bbox_inches='tight')
    print("  ✓ Saved: figure_exp4_3_monosemanticity.png/pdf")

    # Figure 4: Decoder Feature Visualization Grid
    print("\nGenerating Figure 4: Decoder Feature Visualizations...")
    for dataset_name, models_dict in trained_models.items():
        fig = visualize_decoder_features_grid(models_dict, dataset_name,
                                              num_features_per_model=9)
        fig.savefig(f'figure_exp4_4_features_{dataset_name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'figure_exp4_4_features_{dataset_name}.pdf', bbox_inches='tight')
        print(f"  ✓ Saved: figure_exp4_4_features_{dataset_name}.png/pdf")

    # Figure 5: Combined Multi-Panel Summary
    print("\nGenerating Figure 5: Comprehensive Summary...")
    fig5 = plt.figure(figsize=(20, 12))
    gs = fig5.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Sparsity-Reconstruction Tradeoff
    for col, dataset in enumerate(['mnist', 'olivetti', 'lfw']):
        ax = fig5.add_subplot(gs[0, col])
        df_dataset = df[df['dataset'] == dataset]
        topk_data = df_dataset[df_dataset['model_type'] == 'TopK-SAE']
        complete_data = df_dataset[df_dataset['model_type'] == 'Complete']

        ax.plot(topk_data['l0'], topk_data['mse'], 'o-', linewidth=2, markersize=8)
        ax.plot(complete_data['l0'], complete_data['mse'], 's', markersize=12, color='coral')
        ax.set_xlabel('L0 (Active Features)', fontsize=10)
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_title(f'{dataset.upper()}: MSE vs L0', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

    # Row 2: Monosemanticity
    for col, dataset in enumerate(['mnist', 'olivetti', 'lfw']):
        ax = fig5.add_subplot(gs[1, col])
        df_dataset = df[df['dataset'] == dataset]
        topk_data = df_dataset[df_dataset['model_type'] == 'TopK-SAE']

        ax.plot(topk_data['k'], topk_data['monosemanticity_mean'],
                'o-', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('K (Sparsity)', fontsize=10)
        ax.set_ylabel('Monosemanticity', fontsize=10)
        ax.set_title(f'{dataset.upper()}: Interpretability', fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(alpha=0.3)

    # Row 3: Reconstruction Quality vs Interpretability Trade-off
    for col, dataset in enumerate(['mnist', 'olivetti', 'lfw']):
        ax = fig5.add_subplot(gs[2, col])
        df_dataset = df[df['dataset'] == dataset]
        topk_data = df_dataset[df_dataset['model_type'] == 'TopK-SAE']

        scatter = ax.scatter(topk_data['mse'], topk_data['monosemanticity_mean'],
                             c=topk_data['k'], cmap='viridis', s=100, alpha=0.7)

        # Annotate k values
        for _, row in topk_data.iterrows():
            ax.annotate(f"{int(row['k'])}", (row['mse'], row['monosemanticity_mean']),
                        fontsize=8, ha='center')

        ax.set_xlabel('MSE (Lower = Better)', fontsize=10)
        ax.set_ylabel('Monosemanticity (Higher = Better)', fontsize=10)
        ax.set_title(f'{dataset.upper()}: Quality vs Interpretability', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

        if col == 2:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('K (Sparsity)', fontsize=10)

    plt.suptitle('Effect of K on Reconstruction and Interpretability',
                 fontsize=16, fontweight='bold', y=0.995)

    fig5.savefig('figure_exp4_5_comprehensive_summary.png', dpi=300, bbox_inches='tight')
    fig5.savefig('figure_exp4_5_comprehensive_summary.pdf', bbox_inches='tight')
    print("  ✓ Saved: figure_exp4_5_comprehensive_summary.png/pdf")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SPARSE AUTOENCODER: EFFECT OF K ON RECONSTRUCTION AND INTERPRETABILITY")
    print("=" * 80)

    # Run experiment
    df, trained_models = run_experiment_4()

    # Create figures
    create_comprehensive_figures(df, trained_models)

    print("\n" + "=" * 80)
    print("EXPERIMENT 4 COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("- experiment4_k_effects_results.csv")
    print("- figure_exp4_1_sparsity_reconstruction_tradeoff.png/pdf")
    print("- figure_exp4_2_explained_variance.png/pdf")
    print("- figure_exp4_3_monosemanticity.png/pdf")
    print("- figure_exp4_4_features_mnist.png/pdf")
    print("- figure_exp4_4_features_olivetti.png/pdf")
    print("- figure_exp4_4_features_lfw.png/pdf")
    print("- figure_exp4_5_comprehensive_summary.png/pdf")
    print("=" * 80)
