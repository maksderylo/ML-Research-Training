# In NNSAE_IDA2026/ModelVisualizations.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import os

# Add clustering imports with graceful fallback
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    _HAS_CLUSTER_LIBS = True
except Exception:
    _HAS_CLUSTER_LIBS = False


class SAEVisualizer:
    """
    Abstract visualizer for Sparse Autoencoders.
    Works with any model architecture and dataset.
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def _infer_image_shape(self, input_size: int) -> Union[Tuple[int, int], Tuple[int, int, 3]]:
        """Auto-detect image dimensions from input size"""
        if input_size == 784:
            return (28, 28)
        elif input_size == 4096:
            return (64, 64)
        elif input_size == 12288:  # 64x64x3
            return (64, 64, 3)
        else:
            side = int(np.sqrt(input_size))
            if side * side == input_size:
                return (side, side)
            # Find best rectangular factorization
            for h in range(int(np.sqrt(input_size)), 0, -1):
                if input_size % h == 0:
                    return (h, input_size // h)
        raise ValueError(f"Cannot infer image shape from input_size={input_size}")

    def _unpack_batch(self, data, dataset_type: str):
        """Handle different dataloader formats"""
        if dataset_type in ['olivetti', 'lfw']:
            inputs, = data
        else:
            inputs, _ = data
        return inputs.to(self.device)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] for display"""
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    # ============================================================================
    # FEATURE VISUALIZATION
    # ============================================================================

    def visualize_decoder_weights(self, num_features: int = 64,
                                  figsize: Optional[Tuple] = None,
                                  save_path: Optional[str] = None,
                                  cluster_by_similarity: bool = False,
                                  dendrogram_path: Optional[str] = None):
        """
        Visualize decoder weight vectors as images.

        New params:
          - cluster_by_similarity: if True, group features by cosine similarity.
          - dendrogram_path: optional path to save the clustering dendrogram plot.
        """
        weights = self.model.decoder.weight.data.cpu().numpy().T
        input_size = weights.shape[1]
        img_shape = self._infer_image_shape(input_size)

        n_features = weights.shape[0]
        num_features = min(num_features, n_features)

        # Optionally cluster / reorder features by cosine similarity
        ordering = np.arange(n_features)
        if cluster_by_similarity and n_features > 1:
            if _HAS_CLUSTER_LIBS:
                # compute cosine distances
                sim = cosine_similarity(weights)  # (n, n)
                dist = 1.0 - sim
                # Ensure numeric stability: zero diagonal
                np.fill_diagonal(dist, 0.0)
                # convert to condensed form for linkage
                cond = squareform(dist, checks=False)
                Z = linkage(cond, method='average')
                dend = dendrogram(Z, no_plot=True)
                leaves = np.array(dend['leaves'], dtype=int)
                ordering = leaves
                # Optionally save dendrogram
                if dendrogram_path:
                    fig_d, ax_d = plt.subplots(figsize=(8, max(4, n_features * 0.1)))
                    dendrogram(Z, ax=ax_d, labels=[f'F{i}' for i in range(n_features)], leaf_rotation=90)
                    fig_d.suptitle(f'{getattr(self.model, "name", "SAE")} - Decoder Weight Clustering', fontsize=12)
                    save_dir = os.path.dirname(dendrogram_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    plt.tight_layout()
                    fig_d.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
                    plt.close(fig_d)
            else:
                # Fallback: sort by first principal component projection (cheap heuristic)
                try:
                    u, s, vh = np.linalg.svd(weights - weights.mean(axis=0), full_matrices=False)
                    pc1 = (weights @ vh[0])
                    ordering = np.argsort(-pc1)  # descending
                except Exception:
                    ordering = np.arange(n_features)

        # Apply ordering and limit to requested number of features
        ordered_indices = ordering[:num_features]
        weights_ordered = weights[ordered_indices]

        grid_size = int(np.ceil(np.sqrt(num_features)))

        if figsize is None:
            figsize = (grid_size * 2, grid_size * 2)

        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten()

        model_name = getattr(self.model, 'name', 'SAE')
        fig.suptitle(f'{model_name} Decoder Weights', fontsize=14, fontweight='bold')

        for i in range(num_features):
            weight_img = weights_ordered[i].reshape(img_shape)
            weight_img = self._normalize_image(weight_img)

            cmap = 'gray' if len(img_shape) == 2 else None
            axes[i].imshow(weight_img, cmap=cmap, interpolation='nearest')
            axes[i].axis('off')
            original_idx = ordered_indices[i]
            axes[i].set_title(f'F{original_idx}', fontsize=8)

        for i in range(num_features, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_encoder_weights(self, num_features: int = 64,
                                  figsize: Optional[Tuple] = None,
                                  save_path: Optional[str] = None):
        """Visualize encoder weight vectors"""
        if hasattr(self.model.encoder, 'weight'):
            weights = self.model.encoder.weight.data.cpu().numpy()
        elif isinstance(self.model.encoder, torch.nn.Sequential):
            weights = self.model.encoder[0].weight.data.cpu().numpy()
        else:
            raise ValueError("Cannot extract encoder weights")

        input_size = weights.shape[1]
        img_shape = self._infer_image_shape(input_size)

        num_features = min(num_features, weights.shape[0])
        grid_size = int(np.ceil(np.sqrt(num_features)))

        if figsize is None:
            figsize = (grid_size * 2, grid_size * 2)

        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten()

        model_name = getattr(self.model, 'name', 'SAE')
        fig.suptitle(f'{model_name} Encoder Weights', fontsize=14, fontweight='bold')

        for i in range(num_features):
            weight_img = weights[i].reshape(img_shape)
            weight_img = self._normalize_image(weight_img)

            cmap = 'gray' if len(img_shape) == 2 else None
            axes[i].imshow(weight_img, cmap=cmap, interpolation='nearest')
            axes[i].axis('off')
            axes[i].set_title(f'F{i}', fontsize=8)

        for i in range(num_features, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ============================================================================
    # RECONSTRUCTION VISUALIZATION
    # ============================================================================

    def visualize_reconstructions(self, data_loader, dataset_type: str = 'mnist',
                                  num_samples: int = 10, save_path: Optional[str] = None):
        """Compare original vs reconstructed images"""
        data_iter = iter(data_loader)
        inputs = self._unpack_batch(next(data_iter), dataset_type)[:num_samples]

        input_size = inputs.shape[1]
        img_shape = self._infer_image_shape(input_size)

        from .KLSAE import KLSAE
        from .AuxiliarySAE import AuxiliarySparseAutoencoder
        is_kl_sae = isinstance(self.model, KLSAE)
        is_aux_sae = isinstance(self.model, AuxiliarySparseAutoencoder)

        with torch.no_grad():
            model_output = self.model(inputs)
            if is_kl_sae:
                h, reconstructions, _, _ = model_output
            elif is_aux_sae:
                h, reconstructions, _ = model_output
            else:
                h, reconstructions = model_output

        inputs = inputs.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()

        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        model_name = getattr(self.model, 'name', 'SAE')
        fig.suptitle(f'{model_name} Reconstructions', fontsize=14, fontweight='bold')

        cmap = 'gray' if len(img_shape) == 2 else None

        for i in range(num_samples):
            axes[0, i].imshow(inputs[i].reshape(img_shape), cmap=cmap, interpolation='nearest')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            axes[1, i].imshow(reconstructions[i].reshape(img_shape), cmap=cmap, interpolation='nearest')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

        plt.tight_layout()
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ============================================================================
    # TOP ACTIVATING EXAMPLES
    # ============================================================================

    def visualize_top_activating_examples(self, data_loader, dataset_type: str,
                                          feature_indices: List[int], top_k: int = 10,
                                          save_path: Optional[str] = None):
        """
        Show top-k inputs that activate each feature most strongly.
        Standard method from Anthropic's interpretability work.
        """
        top_activations = {idx: {'values': [], 'images': []} for idx in feature_indices}

        with torch.no_grad():
            for data in data_loader:
                inputs = self._unpack_batch(data, dataset_type)

                # Handle different model outputs
                from .KLSAE import KLSAE
                from .AuxiliarySAE import AuxiliarySparseAutoencoder
                is_kl_sae = isinstance(self.model, KLSAE)
                is_aux_sae = isinstance(self.model, AuxiliarySparseAutoencoder)

                model_output = self.model(inputs)
                if is_kl_sae:
                    h, _, _, _ = model_output
                elif is_aux_sae:
                    h, _, _ = model_output
                else:  # Assuming a default order
                    h, _ = model_output

                for i in range(inputs.size(0)):
                    for feature_idx in feature_indices:
                        activation_value = h[i, feature_idx].item()
                        if activation_value > 1e-5:  # Store only non-trivial activations
                            top_activations[feature_idx]['values'].append(activation_value)
                            top_activations[feature_idx]['images'].append(inputs[i].cpu())

        # Check if any images were found before proceeding
        if not any(top_activations[idx]['images'] for idx in feature_indices):
            print("Warning: No activating examples found for the given features. Skipping visualization.")
            return

        # Determine image shape
        sample_img = top_activations[feature_indices[0]]['images'][0].squeeze().numpy()
        img_shape = self._infer_image_shape(len(sample_img))

        # Plot
        num_features = len(feature_indices)
        fig, axes = plt.subplots(num_features, top_k, figsize=(top_k * 2, num_features * 2))
        if num_features == 1:
            axes = axes.reshape(1, -1)

        cmap = 'gray' if len(img_shape) == 2 else None

        for row, feature_idx in enumerate(feature_indices):
            values = np.array(top_activations[feature_idx]['values'])
            images = top_activations[feature_idx]['images']

            if len(values) > 0:
                top_k_indices = np.argsort(values)[-top_k:][::-1]

                for col, img_idx in enumerate(top_k_indices):
                    img = images[img_idx].squeeze().numpy().reshape(img_shape)
                    axes[row, col].imshow(img, cmap=cmap)
                    axes[row, col].axis('off')
                    axes[row, col].set_title(f'{values[img_idx]:.2f}', fontsize=8)

            axes[row, 0].set_ylabel(f'Feature {feature_idx}', fontsize=10)

        model_name = getattr(self.model, 'name', 'SAE')
        plt.suptitle(f'{model_name}: Top Activating Examples', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ============================================================================
    # DASHBOARD
    # ============================================================================
    def create_summary_dashboard(self, output_path: str,
                                 history_path: str, recons_path: str,
                                 weights_path: str, activations_path: str,
                                 hyperparameters: dict):
        """
        Creates a single dashboard image from pre-generated plot files.
        """
        fig = plt.figure(figsize=(20, 22))
        model_name = getattr(self.model, 'name', 'SAE')
        clamp_status = "ON" if hyperparameters.get('use_weight_clamping') else "OFF"
        fig.suptitle(f'{model_name} on {hyperparameters.get("dataset", "N/A")} | Decoder Clamping: {clamp_status}',
                     fontsize=20, fontweight='bold')

        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], width_ratios=[1, 1])

        # 1. Hyperparameters Text
        ax_text = fig.add_subplot(gs[0, 0])
        ax_text.axis('off')
        param_text = "Hyperparameters:\n" + "\n".join([f"- {k}: {v}" for k, v in hyperparameters.items()])
        ax_text.text(0.05, 0.95, param_text, va='top', ha='left', fontsize=12, family='monospace')

        # 2. Training History
        ax_history = fig.add_subplot(gs[0, 1])
        ax_history.set_title('Training History', fontsize=14)
        ax_history.axis('off')
        if os.path.exists(history_path):
            img = plt.imread(history_path)
            ax_history.imshow(img)

        # 3. Reconstructions
        ax_recons = fig.add_subplot(gs[1, 0])
        ax_recons.set_title('Reconstructions', fontsize=14)
        ax_recons.axis('off')
        if os.path.exists(recons_path):
            img = plt.imread(recons_path)
            ax_recons.imshow(img)

        # 4. Decoder Weights
        ax_weights = fig.add_subplot(gs[1, 1])
        ax_weights.set_title('Decoder Weights', fontsize=14)
        ax_weights.axis('off')
        if os.path.exists(weights_path):
            img = plt.imread(weights_path)
            ax_weights.imshow(img)

        # 5. Activation Histograms
        ax_activations = fig.add_subplot(gs[2, :]) # Span both columns
        ax_activations.set_title('Activation Statistics', fontsize=14)
        ax_activations.axis('off')
        if os.path.exists(activations_path):
            img = plt.imread(activations_path)
            ax_activations.imshow(img)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Summary dashboard saved to {output_path}")

    # ============================================================================
    # TRAINING VISUALIZATION
    # ============================================================================

    def _save_training_summary(self, save_path: str, hyperparameters: dict, training_data: List[Tuple[int, float, int]]):
        """Saves hyperparameters and training data to a text file."""
        summary_path = os.path.splitext(save_path)[0] + ".txt"
        # Ensure directory exists
        summary_dir = os.path.dirname(summary_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write("--- Hyperparameters ---\n")
            for key, value in hyperparameters.items():
                f.write(f"{key}: {value}\n")
            f.write("\n--- Training Data (Epoch, Recon Loss, Active Neurons) ---\n")
            for data_point in training_data:
                f.write(f"{data_point[0]+1}, {data_point[1]:.6f}, {data_point[2]}\n")
        print(f"✓ Training summary saved to {summary_path}")

    def plot_training_history(self, training_data: List[Tuple[int, float, int]],
                              save_path: Optional[str] = None,
                              hyperparameters: Optional[dict] = None):
        """
        Plots reconstruction loss and number of active neurons over epochs.
        `training_data` is a list of tuples: (epoch, avg_recon_loss, num_active_neurons)
        """
        if not training_data:
            print("Warning: training_data is empty. Skipping plot.")
            return

        epochs = [data[0] + 1 for data in training_data]
        recon_losses = [data[1] for data in training_data]
        active_neurons = [data[2] for data in training_data]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        model_name = getattr(self.model, 'name', 'SAE')
        fig.suptitle(f'{model_name} Training History', fontsize=16, fontweight='bold')

        # Plot Reconstruction Loss
        ax1.plot(epochs, recon_losses, 'b-o', label='Reconstruction Loss')
        ax1.set_ylabel('Average Reconstruction Loss')
        ax1.set_title('Reconstruction Loss per Epoch')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot Number of Active Neurons
        ax2.plot(epochs, active_neurons, 'r-o', label='Active Neurons')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Number of Unique Active Neurons')
        ax2.set_title('Active Neurons per Epoch')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        if save_path:
            # ensure directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if hyperparameters:
                self._save_training_summary(save_path, hyperparameters, training_data)
        plt.show()

    # ============================================================================
    # AGGREGATED VISUALIZATION
    # ============================================================================

    def plot_aggregated_training_history(self, all_training_data: List[List[Tuple[int, float, int]]],
                                         save_path: Optional[str] = None,
                                         hyperparameters: Optional[dict] = None):
        """
        Plots the mean and standard deviation of training metrics over multiple runs.
        `all_training_data` is a list of training_data lists.
        """
        if not all_training_data:
            print("Warning: all_training_data is empty. Skipping plot.")
            return

        # Find the minimum number of epochs across all runs
        min_epochs = min(len(run_data) for run_data in all_training_data)
        if min_epochs == 0:
            print("Warning: No epoch data found in one of the runs. Skipping plot.")
            return

        # Aggregate data
        recon_losses = np.array([ [epoch_data[1] for epoch_data in run_data[:min_epochs]] for run_data in all_training_data])
        active_neurons = np.array([ [epoch_data[2] for epoch_data in run_data[:min_epochs]] for run_data in all_training_data])

        mean_loss = np.mean(recon_losses, axis=0)
        std_loss = np.std(recon_losses, axis=0)
        mean_neurons = np.mean(active_neurons, axis=0)
        std_neurons = np.std(active_neurons, axis=0)

        epochs = np.arange(1, min_epochs + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        model_name = getattr(self.model, 'name', 'SAE')
        fig.suptitle(f'{model_name} Aggregated Training History ({len(all_training_data)} runs)', fontsize=16, fontweight='bold')

        # Plot Reconstruction Loss
        ax1.plot(epochs, mean_loss, 'b-o', label='Mean Reconstruction Loss')
        ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color='b', alpha=0.2, label='Std. Dev.')
        ax1.set_ylabel('Average Reconstruction Loss')
        ax1.set_title('Reconstruction Loss per Epoch')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot Number of Active Neurons
        ax2.plot(epochs, mean_neurons, 'r-o', label='Mean Active Neurons')
        ax2.fill_between(epochs, mean_neurons - std_neurons, mean_neurons + std_neurons, color='r', alpha=0.2, label='Std. Dev.')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Number of Unique Active Neurons')
        ax2.set_title('Active Neurons per Epoch')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ============================================================================
    # ACTIVATION STATISTICS
    # ============================================================================

    def plot_activation_histogram(self, data_loader, dataset_type: str = 'mnist',
                                  save_path: Optional[str] = None):
        """Plot distribution of feature activation frequencies and strengths"""
        stats = self.get_activation_statistics(data_loader, dataset_type)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        model_name = getattr(self.model, 'name', 'SAE')

        # Activation frequencies
        axes[0].hist(stats['activation_frequencies'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Activation Frequency')
        axes[0].set_ylabel('Number of Features')
        axes[0].set_title(f'{model_name}: Activation Frequencies')
        axes[0].axvline(stats['mean_activation_frequency'], color='r', linestyle='--',
                        label=f"Mean: {stats['mean_activation_frequency']:.4f}")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Activation strengths (active features only)
        active_strengths = stats['activation_strengths'][stats['activation_strengths'] > 0]
        axes[1].hist(active_strengths, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_xlabel('Mean Activation Strength')
        axes[1].set_ylabel('Number of Features')
        axes[1].set_title(f'{model_name}: Activation Strengths')
        axes[1].axvline(stats['mean_activation_strength'], color='r', linestyle='--',
                        label=f"Mean: {stats['mean_activation_strength']:.4f}")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_activation_statistics(self, data_loader, dataset_type: str = 'mnist'):
        """Compute comprehensive activation statistics"""
        activation_counts = torch.zeros(self.model.hidden_size).to(self.device)
        activation_sums = torch.zeros(self.model.hidden_size).to(self.device)
        total_samples = 0

        with torch.no_grad():
            for data in data_loader:
                inputs = self._unpack_batch(data, dataset_type)

                # Handle different model forward signatures
                from .KLSAE import KLSAE
                from .AuxiliarySAE import AuxiliarySparseAutoencoder
                from .BaseSAE import SparseAutoencoder

                model_output = self.model(inputs)

                if isinstance(self.model, KLSAE):
                    h, _, _, _ = model_output
                elif isinstance(self.model, AuxiliarySparseAutoencoder):
                    h, _, _ = model_output
                elif isinstance(self.model, SparseAutoencoder):
                    h, _ = model_output
                else:
                    # Fallback for unknown models, assuming first element is 'h'
                    h = model_output[0]

                activation_counts += (h > 0).sum(dim=0).float()
                activation_sums += h.sum(dim=0)
                total_samples += inputs.size(0)

        activation_counts = activation_counts.cpu().numpy()
        activation_sums = activation_sums.cpu().numpy()

        activation_freq = activation_counts / total_samples
        active_mask = activation_counts > 0
        mean_activation = np.zeros(self.model.hidden_size)
        mean_activation[active_mask] = activation_sums[active_mask] / activation_counts[active_mask]

        stats = {
            'total_features': self.model.hidden_size,
            'dead_features': np.sum(activation_counts == 0),
            'active_features': np.sum(activation_counts > 0),
            'mean_activation_frequency': np.mean(activation_freq),
            'median_activation_frequency': np.median(activation_freq),
            'mean_activation_strength': np.mean(mean_activation[active_mask]) if active_mask.any() else 0.0,
            'activation_frequencies': activation_freq,
            'activation_strengths': mean_activation,
            'activation_counts': activation_counts
        }

        return stats


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def visualize_sae(model, data_loader, dataset_type='mnist',
                  num_weights=64, num_reconstructions=10, device='cuda'):
    """
    All-in-one visualization function.

    Usage:
        from sae_visualizations import visualize_sae
        visualize_sae(model, train_loader, dataset_type='mnist')
    """
    viz = SAEVisualizer(model, device)

    print("Generating decoder weights...")
    viz.visualize_decoder_weights(num_features=num_weights)

    print("Generating reconstructions...")
    viz.visualize_reconstructions(data_loader, dataset_type, num_samples=num_reconstructions)

    print("Generating activation histograms...")
    viz.plot_activation_histogram(data_loader, dataset_type)

    print("✓ Visualization complete")


def visualize_nmf_results(nmf_model, data_loader, folder: str, n_samples: int = 10):
    """
    Visualizes the results of an NMF model.

    Args:
        nmf_model: A trained scikit-learn NMF model.
        data_loader: DataLoader providing the test data (non-normalized).
        folder (str): The directory to save the output plots.
        n_samples (int): Number of reconstruction samples to show.
    """
    print("\n--- Generating NMF Visualizations ---")
    os.makedirs(folder, exist_ok=True)

    # --- 1. Visualize NMF Components (Analogous to Decoder Weights) ---
    components = nmf_model.components_
    num_components = components.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_components)))
    img_shape = (28, 28)  # MNIST specific

    fig_comp, axes_comp = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axes_comp = axes_comp.flatten()
    fig_comp.suptitle('NMF Components (Features)', fontsize=16, fontweight='bold')

    for i in range(num_components):
        comp_img = components[i].reshape(img_shape)
        axes_comp[i].imshow(comp_img, cmap='gray', interpolation='nearest')
        axes_comp[i].axis('off')
        axes_comp[i].set_title(f'F{i}', fontsize=8)

    for i in range(num_components, len(axes_comp)):
        axes_comp[i].axis('off')

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_path_comp = os.path.join(folder, "nmf_components.png")
    plt.savefig(save_path_comp, dpi=300, bbox_inches='tight')
    plt.close(fig_comp)
    print(f"✓ NMF components saved to {save_path_comp}")

    # --- 2. Visualize NMF Reconstructions ---
    inputs, _ = next(iter(data_loader))
    inputs = inputs[:n_samples]

    # Get NMF reconstructions
    W = nmf_model.transform(inputs.numpy())
    H = nmf_model.components_
    reconstructions = np.dot(W, H)

    fig_recons, axes_recons = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    fig_recons.suptitle('NMF Reconstructions', fontsize=14, fontweight='bold')

    for i in range(n_samples):
        # Original
        axes_recons[0, i].imshow(inputs[i].reshape(img_shape), cmap='gray', interpolation='nearest')
        axes_recons[0, i].axis('off')
        if i == 0:
            axes_recons[0, i].set_title('Original', fontsize=10)

        # Reconstructed
        axes_recons[1, i].imshow(reconstructions[i].reshape(img_shape), cmap='gray', interpolation='nearest')
        axes_recons[1, i].axis('off')
    if i == 0:
                axes_recons[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_path_recons = os.path.join(folder, "nmf_reconstructions.png")
    plt.savefig(save_path_recons, dpi=300, bbox_inches='tight')
    plt.close(fig_recons)
    print(f"✓ NMF reconstructions saved to {save_path_recons}")
