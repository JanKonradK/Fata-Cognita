"""Full VAE model: encoder + reparameterization + decoder.

Combines all model components into a single module for training and inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from fata_cognita.model.decoder import TrajectoryDecoder
from fata_cognita.model.encoder import Encoder

if TYPE_CHECKING:
    from fata_cognita.config import Config, ModelConfig


class TrajectoryVAE(nn.Module):
    """Variational Autoencoder for life trajectory generation.

    Encodes static features into a latent distribution, samples via
    reparameterization, and decodes into year-by-year trajectories.

    Args:
        num_features: Number of static input features.
        config: Full project configuration.
    """

    def __init__(self, num_features: int, config: Config) -> None:
        super().__init__()
        self.config = config
        model_cfg: ModelConfig = config.model

        self.encoder = Encoder(num_features, model_cfg)
        self.decoder = TrajectoryDecoder(
            config=model_cfg,
            max_seq_len=config.data.max_seq_len,
            num_life_states=config.data.num_life_states,
        )

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample z from the latent distribution via reparameterization trick.

        Args:
            mu: Mean of the latent distribution, shape (B, latent_dim).
            log_var: Log-variance of the latent distribution, shape (B, latent_dim).
            deterministic: If True, return mu without sampling.

        Returns:
            Sampled latent vector z, shape (B, latent_dim).
        """
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        static_features: torch.Tensor,
        deterministic: bool = False,
        seq_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode → sample → decode.

        Args:
            static_features: Static feature tensor, shape (B, num_features).
            deterministic: If True, use mu instead of sampling z.
            seq_len: Optional sequence length override.

        Returns:
            Dictionary with keys: life_state_logits (B, S, C), income (B, S),
            satisfaction (B, S), mu (B, D), log_var (B, D), z (B, D).
        """
        mu, log_var = self.encoder(static_features)
        z = self.reparameterize(mu, log_var, deterministic=deterministic)
        outputs = self.decoder(z, seq_len=seq_len)

        return {
            "life_state_logits": outputs["life_state_logits"],
            "income": outputs["income"],
            "satisfaction": outputs["satisfaction"],
            "mu": mu,
            "log_var": log_var,
            "z": z,
        }

    def encode(self, static_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode static features to latent distribution parameters.

        Args:
            static_features: Shape (B, num_features).

        Returns:
            Tuple of (mu, log_var), each shape (B, latent_dim).
        """
        return self.encoder(static_features)

    def decode(self, z: torch.Tensor, seq_len: int | None = None) -> dict[str, torch.Tensor]:
        """Decode a latent vector into a trajectory.

        Args:
            z: Latent vector, shape (B, latent_dim).
            seq_len: Optional sequence length override.

        Returns:
            Decoder output dictionary.
        """
        return self.decoder(z, seq_len=seq_len)
