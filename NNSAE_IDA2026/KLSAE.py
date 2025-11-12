# In NNSAE_IDA2026/KLSAE.py

import torch
from .BaseSAE import SparseAutoencoder
from typing import Callable
import torch.nn as nn


class KLSAE(SparseAutoencoder):
    def __init__(self, input_size: int, hidden_size: int, activation: Callable[[torch.tensor], torch.tensor],
                 rho: float = 0.05, beta: float = 3.0, aux_k: int = 10, dead_steps_threshold: int = 100):

        super().__init__(input_size=input_size, hidden_size=hidden_size, activation=activation)
        self.rho = rho
        self.beta = beta
        self.aux_k = aux_k
        self.dead_steps_threshold = dead_steps_threshold

        if isinstance(activation, nn.ReLU):
            self.penalty_type = 'l1'
        elif isinstance(activation, nn.Sigmoid):
            self.penalty_type = 'kl'
        else:
            print(f"Warning: Unsupported activation {type(activation)} for sparsity. Defaulting to L1 penalty.")
            self.penalty_type = 'l1'

        self.register_buffer("stats_last_nonzero", torch.zeros(hidden_size, dtype=torch.long))

    def _topk_auxk(self, activations: torch.Tensor, dead_mask: torch.Tensor) -> torch.Tensor:
        if self.aux_k == 0:
            return torch.zeros_like(activations)

        masked_activations = activations.clone()
        masked_activations[:, ~dead_mask] = float('-inf')

        k_aux_actual = min(self.aux_k, dead_mask.sum().item())
        if k_aux_actual == 0:
            return torch.zeros_like(activations)

        _, idx = torch.topk(masked_activations, k_aux_actual, dim=1)
        mask = torch.zeros_like(activations)
        mask.scatter_(1, idx, 1.0)
        return mask

    def forward(self, x):
        pre_activations = self.encoder(x)
        h = self.activation(pre_activations)
        x_hat = self.decoder(h)

        # --- Dead neuron revival logic ---
        h_aux = None
        if self.training and self.penalty_type == 'l1' and self.aux_k > 0:
            dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
            if dead_mask.any():
                aux_mask = self._topk_auxk(pre_activations, dead_mask)
                h_aux = self.activation(pre_activations) * aux_mask

            # --- Corrected dead feature statistics update ---
            # A neuron is active if it fired in the main pass OR the aux pass.
            main_active_mask = (h.abs() > 1e-6).any(dim=0)
            aux_active_mask = (h_aux.abs() > 1e-6).any(dim=0) if h_aux is not None else torch.zeros_like(
                main_active_mask)
            combined_active_mask = main_active_mask | aux_active_mask

            self.stats_last_nonzero[combined_active_mask] = 0
            self.stats_last_nonzero[~combined_active_mask] += 1

        # Pass h_aux to compute_loss. It will be None if not calculated.
        return h, x_hat, pre_activations, h_aux

    def sparsity_penalty(self, h):
        if self.penalty_type == 'kl':
            rho_hat = torch.mean(h, dim=0)
            rho = self.rho
            epsilon = 1e-8
            rho_hat = torch.clamp(rho_hat, min=epsilon, max=1.0 - epsilon)
            kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
            return self.beta * torch.sum(kl_div)
        elif self.penalty_type == 'l1':
            return self.beta * self.rho * torch.norm(h, 1, dim=-1).mean()
        else:
            return torch.tensor(0.0, device=h.device)

    def compute_loss(self, x: torch.Tensor, h: torch.Tensor, x_hat: torch.Tensor,
                     pre_activations: torch.Tensor, h_aux: torch.Tensor,
                     warmup: bool = False, auxk_coef: float = 1 / 32) -> tuple[torch.Tensor, dict]:
        recon_loss = torch.sum((x - x_hat) ** 2) / x.size(0)
        sparse_loss = torch.tensor(0.0, device=x.device)
        auxk_loss = torch.tensor(0.0, device=x.device)
        num_dead = 0

        if not warmup:
            sparse_loss = self.sparsity_penalty(h)

            if h_aux is not None:
                with torch.no_grad():
                    residual = x - x_hat
                residual_hat = self.decoder(h_aux)
                auxk_loss = torch.sum((residual - residual_hat) ** 2) / x.size(0)

            if self.training:
                num_dead = (self.stats_last_nonzero > self.dead_steps_threshold).sum().item()

        total_loss = recon_loss + sparse_loss + (auxk_loss * auxk_coef)

        return total_loss, {
            'recon_loss': recon_loss.item(),
            'sparse_loss': sparse_loss.item(),
            'auxk_loss': auxk_loss.item(),
            'num_dead': num_dead
        }
