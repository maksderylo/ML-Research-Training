import torch.nn as nn
from typing import Callable, Dict, Any, Tuple
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AuxiliarySparseAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, k_top: int, aux_k: int,
                 dead_steps_threshold: int,
                 activation: Callable[[torch.Tensor], torch.Tensor]):
        super(AuxiliarySparseAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k_top = k_top
        self.auxk = aux_k  # Number of auxiliary features
        self.activation = activation
        self.dead_steps_threshold = dead_steps_threshold

        # Encoder maps input to hidden representation
        self.encoder = nn.Linear(input_size, hidden_size)
        # Decoder maps hidden representation back to input space
        self.decoder = nn.Linear(hidden_size, input_size)

        # Initialize encoder weights first with random directions
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        # Initialize the decoder to be the transpose of the encoder weights
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

        # Track dead features - counts steps since last activation
        self.register_buffer("stats_last_nonzero",
                             torch.zeros(hidden_size, dtype=torch.long))

    def _topk_mask(self, activations: torch.Tensor) -> torch.Tensor:
        """Select top-k activations"""
        k = max(0, min(self.k_top, activations.size(1)))
        _, idx = torch.topk(activations, k, dim=1)
        mask = torch.zeros_like(activations)
        mask.scatter_(1, idx, 1.0)
        return mask

    def _topk_auxk(self, activations: torch.Tensor, dead_mask: torch.Tensor) -> torch.Tensor:
        """Select top-k_aux from DEAD features only"""
        if self.auxk is None or self.auxk == 0:
            return torch.zeros_like(activations)

        # Set all non-dead features to very negative value
        masked_activations = activations.clone()
        masked_activations[:, ~dead_mask] = float('-inf')

        # TopK on remaining (dead) features
        k_aux_actual = min(self.auxk, dead_mask.sum().item())
        if k_aux_actual == 0:
            return torch.zeros_like(activations)

        _, idx = torch.topk(masked_activations, k_aux_actual, dim=1)
        mask = torch.zeros_like(activations)
        mask.scatter_(1, idx, 1.0)
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FIX: Always return 2 values for backward compatibility
        Auxiliary info is computed in compute_loss() instead
        """
        # Encode
        pre_activations = self.encoder(x)

        # Main TopK path
        pre_activations_activated = self.activation(pre_activations)
        mask = self._topk_mask(pre_activations_activated)
        h = pre_activations_activated * mask
        x_hat = self.decoder(h)

        # Update dead feature statistics (for next forward pass)
        active_mask = (h.abs() > 1e-5).any(dim=0)
        self.stats_last_nonzero *= (~active_mask).long()
        self.stats_last_nonzero += 1

        # FIX: Return pre_activations for use in loss calculation
        return h, x_hat, pre_activations


    def compute_loss(self, x: torch.Tensor, h: torch.Tensor, x_hat: torch.Tensor,
                     pre_activations: torch.Tensor,
                     auxk_coef: float = 1 / 32) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reconstruction loss + auxiliary loss for reviving dead neurons.
        The auxiliary loss trains dead neurons to reconstruct the residual error.
        """
        # Main reconstruction loss
        recon_loss = torch.sum((x - x_hat) ** 2) / x.size(0)

        # --- Auxiliary Loss for Dead Neurons ---
        auxk_loss = torch.tensor(0.0, device=x.device)
        num_dead = 0

        if self.training and self.auxk is not None and self.auxk > 0:
            # 1. Identify dead features
            dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
            num_dead = dead_mask.sum().item()

            if num_dead > 0:
                # 2. Calculate the residual error that the main model failed to reconstruct
                with torch.no_grad():
                    residual = x - x_hat

                # 3. Select top-k dead features to reconstruct the residual
                # We use the pre-activations from the original forward pass
                aux_mask = self._topk_auxk(pre_activations, dead_mask)
                h_aux = self.activation(pre_activations) * aux_mask

                # 4. Reconstruct the residual using only the auxiliary (dead) features
                residual_hat = self.decoder(h_aux)

                # 5. Calculate the auxiliary reconstruction loss
                auxk_loss = torch.sum((residual - residual_hat) ** 2) / x.size(0)

        total_loss = recon_loss + (auxk_loss * auxk_coef)

        return total_loss, {
            'recon_loss': recon_loss.item(),
            'auxk_loss': auxk_loss.item(),
            'num_dead': num_dead
        }
