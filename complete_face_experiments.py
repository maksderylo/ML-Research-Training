import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from skimage.transform import resize
import os


def load_orl_faces(resize_to=(46, 56), test_size=100, train_size=300):
    print("Loading ORL Faces dataset...")
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    images = data.images  # Shape: (400, 64, 64)

    # Resize images to match paper dimensions
    print(f"Resizing images to {resize_to}...")
    resized_images = []
    for img in images:
        resized = resize(img, resize_to, anti_aliasing=True)
        resized_images.append(resized)

    resized_images = np.array(resized_images)

    # This prevents sigmoid saturation issues
    resized_images = (resized_images - resized_images.min()) / (resized_images.max() - resized_images.min())
    # Flatten for autoencoder
    flat_images = resized_images.reshape(len(images), -1)

    # Split into train/test as in paper
    X_train, X_test = train_test_split(
        flat_images,
        test_size=test_size,
        train_size=train_size,
        random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Input dimension: {X_train.shape[1]}")
    print(f"Data range: [{X_train.min():.4f}, {X_train.max():.4f}]")

    return X_train, X_test, resize_to


# SAE implementation
class SparseAutoencoderKL(nn.Module):
    def __init__(self, input_size=2576, hidden_size=100, sparsity_param=0.05):
        super(SparseAutoencoderKL, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity_param = sparsity_param

        self.encoder = nn.Linear(input_size, hidden_size, bias=True)
        self.decoder = nn.Linear(hidden_size, input_size, bias=True)

        # FIXED INITIALIZATION
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

        # Initialize encoder bias to encourage sparsity
        # Negative bias = lower activation probability
        nn.init.constant_(self.encoder.bias, -2.0)  # Start with sparse activations
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        h = torch.sigmoid(self.encoder(x))
        x_hat = torch.sigmoid(self.decoder(h))
        return h, x_hat

    def compute_kl_loss(self, h):
        rho = self.sparsity_param
        rho_hat = torch.mean(h, dim=0)

        # FIXED: Larger epsilon and explicit clamping
        eps = 1e-7  # Increased from 1e-10
        rho_hat = torch.clamp(rho_hat, min=eps, max=1 - eps)

        # Add small constant to rho as well for numerical stability
        rho_safe = torch.clamp(torch.tensor(rho), min=eps, max=1 - eps)

        kl_div = rho_safe * torch.log(rho_safe / rho_hat) + \
                 (1 - rho_safe) * torch.log((1 - rho_safe) / (1 - rho_hat))

        kl_div = torch.clamp(kl_div, min=0.0)

        return torch.sum(kl_div)

    def compute_loss(self, x, x_hat, h, beta=3.0, lambda_reg=0.003):
        """
        J_SAE = J_E + β*J_KL + λ*weight_decay (Eq. 7)
        """
        # Reconstruction error
        mse_loss = F.mse_loss(x_hat, x, reduction='mean')

        # Only sum positive KL divergence values (KL should always be >= 0)
        kl_loss = self.compute_kl_loss(h)

        # Weight decay (L2 regularization)
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())

        total_loss = mse_loss + beta * kl_loss + (lambda_reg / 2) * l2_reg

        return total_loss, mse_loss, kl_loss, l2_reg


class NCAE(nn.Module):
    """
    Nonnegativity Constrained Autoencoder
    Implements Eq. (8-13) from the paper
    """
    def __init__(self, input_size=2576, hidden_size=100, sparsity_param=0.05):
        super(NCAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity_param = sparsity_param
        
        self.encoder = nn.Linear(input_size, hidden_size, bias=True)
        self.decoder = nn.Linear(hidden_size, input_size, bias=True)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def forward(self, x):
        h = torch.sigmoid(self.encoder(x))
        x_hat = torch.sigmoid(self.decoder(h))
        return h, x_hat
    
    def compute_loss(self, x, x_hat, h, beta=3.0, alpha=0.003):
        # Reconstruction error
        mse_loss = F.mse_loss(x_hat, x, reduction='mean')
        
        # KL divergence sparsity
        rho_hat = torch.mean(h, dim=0)
        rho = self.sparsity_param
        kl_div = rho * torch.log(rho / (rho_hat + 1e-10)) + \
                 (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-10))
        kl_loss = torch.sum(kl_div)
        
        # Nonnegativity penalty
        nonnegativity_penalty = 0
        for param in self.parameters():
            negative_weights = torch.clamp(param, max=0)
            nonnegativity_penalty += torch.sum(negative_weights ** 2)
        
        total_loss = mse_loss + beta * kl_loss + (alpha / 2) * nonnegativity_penalty
        
        return total_loss, mse_loss, kl_loss, nonnegativity_penalty


# NNSAE implementation
class NNSAE(nn.Module):
    """
    Uses hard constraints: weights are clamped to be >= 0
    """
    def __init__(self, input_size=2576, hidden_size=100, sparsity_param=0.05):
        super(NNSAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity_param = sparsity_param
        
        # Encoder with tied weights for decoder
        self.encoder = nn.Linear(input_size, hidden_size, bias=True)
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize to nonnegative values
        nn.init.uniform_(self.encoder.weight, 0, 0.1)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder_bias)
    
    def forward(self, x):
        # Hard constraint: clamp weights to be nonnegative
        with torch.no_grad():
            self.encoder.weight.clamp_(min=0)
        
        h = torch.sigmoid(self.encoder(x))
        # Decoder uses transposed encoder weights (weight tying)
        x_hat = F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        x_hat = torch.sigmoid(x_hat)
        
        return h, x_hat
    
    def compute_loss(self, x, x_hat, h, beta=3.0):
        mse_loss = F.mse_loss(x_hat, x, reduction='mean')
        
        # KL divergence sparsity
        rho_hat = torch.mean(h, dim=0)
        rho = self.sparsity_param
        kl_div = rho * torch.log(rho / (rho_hat + 1e-10)) + \
                 (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-10))
        kl_loss = torch.sum(kl_div)
        
        total_loss = mse_loss + beta * kl_loss
        return total_loss, mse_loss, kl_loss


# ============================================================================
# 5. NONNEGATIVE MATRIX FACTORIZATION (NMF) - Lee & Seung 1999
# ============================================================================

class NMF_Autoencoder(nn.Module):
    """
    Nonnegative Matrix Factorization (Lee & Seung, 1999)
    Uses multiplicative update rules
    """
    def __init__(self, input_size=2576, hidden_size=100):
        super(NMF_Autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # W matrix (basis vectors) - learned
        self.W = nn.Parameter(torch.rand(input_size, hidden_size) + 0.1)
        
        # H matrix (coefficients) - computed during forward
        self.register_buffer('H_buffer', torch.zeros(1, hidden_size))
        
    def forward(self, X):
        """
        Compute H given X and W, then reconstruct
        """
        W = torch.clamp(self.W, min=1e-10)
        batch_size = X.size(0)
        
        # Initialize H
        H = torch.rand(batch_size, self.hidden_size, device=X.device) + 0.1
        
        # Iterative updates for H (simplified - 10 iterations)
        for _ in range(10):
            numerator = X.t() @ H
            denominator = W @ (H.t() @ H) + 1e-10
            W_new = W * (numerator / denominator)
            W = torch.clamp(W_new, min=1e-10)
            
            numerator = X @ W
            denominator = H @ (W.t() @ W) + 1e-10
            H = H * (numerator / denominator)
            H = torch.clamp(H, min=1e-10)
        
        # Reconstruct
        X_hat = H @ W.t()
        
        return H, X_hat
    
    def compute_loss(self, X, X_hat):
        return F.mse_loss(X_hat, X, reduction='mean')


# ============================================================================
# 6. TRAINING FUNCTIONS
# ============================================================================

def train_autoencoder(model, train_loader, num_epochs=400, lr=1.0,
                      beta=3.0, lambda_reg=0.003, alpha=0.003,
                      device='cuda', model_type='SAE'):
    """
    Fixed training function with proper L-BFGS usage
    """
    model.to(device)
    model.train()

    # L-BFGS optimizer - FIXED CONFIGURATION
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=20,  # REDUCED from 400 to 20 per step
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    history = {'loss': [], 'mse': [], 'sparsity': []}

    print(f"\nTraining {model_type} with L-BFGS...")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_mse = 0
        epoch_sparsity = 0
        num_batches = 0

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)

            # Define closure for L-BFGS - FIXED
            def closure():
                optimizer.zero_grad()
                h, x_hat = model(data)

                # Compute loss based on model type
                if model_type == 'SAE':
                    loss, _, _, _ = model.compute_loss(data, x_hat, h, beta, lambda_reg)
                elif model_type == 'NCAE':
                    loss, _, _, _ = model.compute_loss(data, x_hat, h, beta, alpha)
                else:  # NNSAE
                    loss, _, _ = model.compute_loss(data, x_hat, h, beta)

                loss.backward()
                return loss

            # Perform L-BFGS step
            optimizer.step(closure)

            # Evaluate metrics AFTER optimization step (no grad)
            with torch.no_grad():
                h, x_hat = model(data)

                if model_type == 'SAE':
                    total_loss, mse, kl, _ = model.compute_loss(data, x_hat, h, beta, lambda_reg)
                    sparsity_term = kl.item()
                elif model_type == 'NCAE':
                    total_loss, mse, kl, _ = model.compute_loss(data, x_hat, h, beta, alpha)
                    sparsity_term = kl.item()
                else:  # NNSAE
                    total_loss, mse, kl = model.compute_loss(data, x_hat, h, beta)
                    sparsity_term = kl.item()

            # Project weights for NNSAE
            if model_type == 'NNSAE':
                with torch.no_grad():
                    model.encoder.weight.clamp_(min=0)

            epoch_loss += total_loss.item()
            epoch_mse += mse.item()
            epoch_sparsity += sparsity_term
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_sparsity = epoch_sparsity / num_batches

        history['loss'].append(avg_loss)
        history['mse'].append(avg_mse)
        history['sparsity'].append(avg_sparsity)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}, '
                  f'MSE: {avg_mse:.6f}, Sparsity: {avg_sparsity:.6f}')

    return model, history


def train_autoencoder_batch(model, X_train, num_iterations=400,
                            beta=3.0, lambda_reg=0.003, alpha=0.003,
                            device='cuda', model_type='SAE'):
    """
    Single L-BFGS optimization with stopping
    """
    model.to(device)
    model.train()

    X_train_tensor = torch.FloatTensor(X_train).to(device)

    # FIXED: One optimizer call with proper max_iter
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,  # Paper uses lr=1.0
        max_iter=20,  # Internal iterations per step
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    history = {'loss': [], 'mse': [], 'sparsity': []}
    iteration_count = [0]  # Use list to modify in closure

    print(f"\nTraining {model_type} with L-BFGS (full batch)...")
    print(f"Training samples: {X_train.shape[0]}")

    def closure():
        optimizer.zero_grad()
        h, x_hat = model(X_train_tensor)

        if model_type == 'SAE':
            loss, mse, kl, reg = model.compute_loss(
                X_train_tensor, x_hat, h, beta, lambda_reg
            )
        elif model_type == 'NCAE':
            loss, mse, kl, reg = model.compute_loss(
                X_train_tensor, x_hat, h, beta, alpha
            )
        else:  # NNSAE
            loss, mse, kl = model.compute_loss(
                X_train_tensor, x_hat, h, beta
            )

        loss.backward()

        # Log metrics
        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            with torch.no_grad():
                avg_activation = torch.mean(h).item()
                print(f'Iteration [{iteration_count[0]}/{num_iterations}], '
                      f'Loss: {loss.item():.6f}, '
                      f'MSE: {mse.item():.6f}, '
                      f'KL: {kl.item():.6f}, '
                      f'Avg activation: {avg_activation:.4f}')

                history['loss'].append(loss.item())
                history['mse'].append(mse.item())
                history['sparsity'].append(kl.item())

        return loss

    print(f"\nTraining {model_type} with L-BFGS...")

    # FIXED: Proper iteration loop
    for step in range(num_iterations // 20):  # 20 iterations per step
        optimizer.step(closure)

        # Project weights for NNSAE
        if model_type == 'NNSAE':
            with torch.no_grad():
                model.encoder.weight.clamp_(min=0)

    return model, history



def train_nmf(X_train, hidden_size=100, num_epochs=400, device='cuda'):
    """
    Train NMF using multiplicative update rules
    """
    print("\nTraining NMF...")
    X = torch.FloatTensor(X_train).to(device)
    n_samples, n_features = X.shape
    
    # Initialize W and H
    W = torch.rand(n_features, hidden_size, device=device) + 0.1
    H = torch.rand(n_samples, hidden_size, device=device) + 0.1
    
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        # Multiplicative update for H
        numerator = X.t() @ H
        denominator = W @ (H.t() @ H) + 1e-10
        W = W * (numerator / denominator)
        W = torch.clamp(W, min=1e-10)
        
        # Multiplicative update for W
        numerator = X @ W
        denominator = H @ (W.t() @ W) + 1e-10
        H = H * (numerator / denominator)
        H = torch.clamp(H, min=1e-10)
        
        # Compute reconstruction error
        X_hat = H @ W.t()
        loss = F.mse_loss(X_hat, X)
        history['loss'].append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    
    # Create model to store W
    model = NMF_Autoencoder(n_features, hidden_size)
    model.W.data = W.cpu()
    
    return model, history


# ============================================================================
# 7. VISUALIZATION AND EVALUATION
# ============================================================================

def visualize_receptive_fields(model, img_shape, model_name, save_dir='results'):
    """
    Visualize learned receptive fields (encoder weights or basis vectors)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract weights
    if isinstance(model, NMF_Autoencoder):
        weights = model.W.detach().cpu().numpy().T  # Transpose for consistency
    else:
        weights = model.encoder.weight.detach().cpu().numpy()
    
    # Plot first 100 receptive fields (10x10 grid)
    n_display = min(100, weights.shape[0])
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    
    for i in range(n_display):
        ax = axes[i // 10, i % 10]
        receptive_field = weights[i].reshape(img_shape)
        
        # Normalize for display
        vmin, vmax = receptive_field.min(), receptive_field.max()
        normalized = (receptive_field - vmin) / (vmax - vmin + 1e-10)
        
        ax.imshow(normalized, cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'{model_name} - Receptive Fields', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_receptive_fields.png', dpi=150)
    plt.close()
    print(f"Saved receptive fields for {model_name}")


def evaluate_reconstruction(model, test_loader, img_shape, model_name,
                            device='cuda', save_dir='results', is_nmf=False):
    """
    Evaluate reconstruction quality
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    total_mse = 0
    total_reconstruction_error = 0  # NEW: Paper's metric
    num_samples = 0
    all_originals = []
    all_reconstructions = []

    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            data = data.to(device)

            if is_nmf:
                # For NMF, we need to compute H from X
                W = torch.clamp(model.W.to(device), min=1e-10)
                H = torch.rand(data.size(0), model.hidden_size, device=device) + 0.1

                for _ in range(10):
                    numerator = data @ W
                    denominator = H @ (W.t() @ W) + 1e-10
                    H = H * (numerator / denominator)
                    H = torch.clamp(H, min=1e-10)

                x_hat = H @ W.t()
            else:
                _, x_hat = model(data)

            mse = F.mse_loss(x_hat, data, reduction='mean')
            # Paper's reconstruction error: sum of squared differences per sample
            reconstruction_error = torch.sum((x_hat - data) ** 2, dim=1).mean()

            total_mse += mse.item()
            total_reconstruction_error += reconstruction_error.item()
            num_samples += 1

            all_originals.append(data.cpu().numpy())
            all_reconstructions.append(x_hat.cpu().numpy())

    avg_mse = total_mse / num_samples
    avg_recon_error = total_reconstruction_error / num_samples

    # Visualize sample reconstructions
    originals = np.concatenate(all_originals, axis=0)[:10]
    reconstructions = np.concatenate(all_reconstructions, axis=0)[:10]

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        # Original
        axes[0, i].imshow(originals[i].reshape(img_shape), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        # Reconstruction
        axes[1, i].imshow(reconstructions[i].reshape(img_shape), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12)

    plt.suptitle(f'{model_name} - MSE: {avg_mse:.6f}, Recon Error: {avg_recon_error:.4f}',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_reconstruction.png', dpi=150)
    plt.close()

    print(f"{model_name} - MSE: {avg_mse:.6f}, Reconstruction Error: {avg_recon_error:.4f}")
    return avg_mse


def compute_sparseness(weights):
    """
    Compute sparseness using Hoyer's measure (Eq. from paper references)
    Sparseness = (sqrt(n) - L1/L2) / (sqrt(n) - 1)
    where n is the number of features
    """
    n = weights.shape[1]
    l1_norm = np.abs(weights).sum(axis=1)
    l2_norm = np.sqrt((weights ** 2).sum(axis=1))
    
    sparseness = (np.sqrt(n) - l1_norm / (l2_norm + 1e-10)) / (np.sqrt(n) - 1)
    return sparseness.mean()


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters from paper (Table I)
    input_size = 2576  # 46x56 pixels
    hidden_size = 100
    sparsity_param = 0.05
    beta = 3.0
    lambda_reg = 0.003
    alpha = 0.003
    batch_size = 100
    num_epochs = 400
    lr = 1.0

    # Load data
    X_train, X_test, img_shape = load_orl_faces(resize_to=(46, 56))

    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train all models
    models = {}
    histories = {}

    # 1. Train SAE
    print("\n" + "="*70)
    print("Training Sparse Autoencoder (SAE)")
    print("="*70)
    sae_model = SparseAutoencoderKL(input_size, hidden_size, sparsity_param)
    # models['SAE'], histories['SAE'] = train_autoencoder(
    #     sae_model, train_loader, num_epochs, lr, beta, lambda_reg, alpha, device, 'SAE'
    # )
    models['SAE'], histories['SAE'] = train_autoencoder_batch(
        sae_model, X_train, num_iterations=400,
        beta=beta, lambda_reg=lambda_reg, alpha=alpha,
        device=device, model_type='SAE'
    )
    #
    # # 2. Train NCAE
    # print("\n" + "="*70)
    # print("Training Nonnegativity Constrained Autoencoder (NCAE)")
    # print("="*70)
    # ncae_model = NCAE(input_size, hidden_size, sparsity_param)
    # models['NCAE'], histories['NCAE'] = train_autoencoder(
    #     ncae_model, train_loader, num_epochs, lr, beta, lambda_reg, alpha, device, 'NCAE'
    # )
    #
    # # 3. Train NNSAE
    # print("\n" + "="*70)
    # print("Training Nonnegative Sparse Autoencoder (NNSAE)")
    # print("="*70)
    # nnsae_model = NNSAE(input_size, hidden_size, sparsity_param)
    # models['NNSAE'], histories['NNSAE'] = train_autoencoder(
    #     nnsae_model, train_loader, num_epochs, lr, beta, lambda_reg, alpha, device, 'NNSAE'
    # )

    # 4. Train NMF
    print("\n" + "="*70)
    print("Training Nonnegative Matrix Factorization (NMF)")
    print("="*70)
    models['NMF'], histories['NMF'] = train_nmf(X_train, hidden_size, num_epochs, device)

    # Evaluate and visualize
    print("\n" + "="*70)
    print("Evaluation and Visualization")
    print("="*70)

    results = {}
    for name, model in models.items():
        is_nmf = (name == 'NMF')
        results[name] = evaluate_reconstruction(
            model, test_loader, img_shape, name, device, is_nmf=is_nmf
        )
        visualize_receptive_fields(model, img_shape, name)

    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print("\nReconstruction MSE on Test Set:")
    for name, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:10s}: {mse:.6f}")

    print("\nReceptive fields and reconstruction visualizations saved to 'results/' directory")
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
