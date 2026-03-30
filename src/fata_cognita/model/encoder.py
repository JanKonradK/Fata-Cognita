"""MLP encoder: static features → latent distribution parameters (mu, log_var)."""

from __future__ import annotations

import torch
import torch.nn as nn

from fata_cognita.config import ModelConfig


class Encoder(nn.Module):
    """MLP encoder that maps static features to latent distribution parameters.

    Architecture:
        Input (B, num_features) → hidden layers with LayerNorm + ReLU + Dropout
        → mu (B, latent_dim), log_var (B, latent_dim)

    Args:
        num_features: Dimensionality of the static feature vector.
        config: Model configuration with latent_dim, encoder_hidden_dims, dropout.
    """

    def __init__(self, num_features: int, config: ModelConfig) -> None:
        super().__init__()
        self.num_features = num_features
        self.latent_dim = config.latent_dim

        layers: list[nn.Module] = []
        in_dim = num_features
        for hidden_dim in config.encoder_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, config.latent_dim)
        self.logvar_head = nn.Linear(in_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode static features to latent distribution parameters.

        Args:
            x: Static feature tensor, shape (B, num_features).

        Returns:
            Tuple of (mu, log_var), each shape (B, latent_dim).
        """
        h = self.hidden(x)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        return mu, log_var
