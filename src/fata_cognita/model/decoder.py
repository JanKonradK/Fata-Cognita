"""Transformer decoder: generates year-by-year trajectories conditioned on latent z.

Uses cross-attention to condition on the latent vector, with learnable positional
embeddings for age positions and three output heads for life state, income,
and satisfaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fata_cognita.config import ModelConfig


class TrajectoryDecoder(nn.Module):
    """Transformer decoder for trajectory generation.

    Architecture:
        z (B, latent_dim) → project to (B, 1, d_model) as cross-attention memory
        Positional embeddings (B, seq_len, d_model) as decoder target
        → N TransformerDecoderLayers with causal self-attention + z cross-attention
        → Three output heads: life states (classification), income (regression),
          satisfaction (bounded regression)

    Args:
        config: Model configuration.
        max_seq_len: Maximum sequence length (number of age positions).
        num_life_states: Number of life state categories.
    """

    def __init__(
        self,
        config: ModelConfig,
        max_seq_len: int = 62,
        num_life_states: int = 9,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.max_seq_len = max_seq_len

        # Project z to decoder memory space
        self.z_proj = nn.Linear(config.latent_dim, config.d_model)

        # Learnable positional embeddings (one per age slot)
        self.pos_embed = nn.Embedding(max_seq_len, config.d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,  # use (S, B, D) format
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
        )

        # Output heads
        self.life_state_head = nn.Linear(config.d_model, num_life_states)
        self.income_head = nn.Linear(config.d_model, 1)
        self.satisfaction_head = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )

        # Cache the causal mask
        self._causal_mask: torch.Tensor | None = None

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate or retrieve a cached upper-triangular causal mask.

        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Boolean mask of shape (seq_len, seq_len) where True = masked.
        """
        if self._causal_mask is None or self._causal_mask.size(0) != seq_len:
            self._causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
        return self._causal_mask.to(device)

    def forward(
        self, z: torch.Tensor, seq_len: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Decode latent vector z into a full trajectory.

        Args:
            z: Latent vector, shape (B, latent_dim).
            seq_len: Optional override for sequence length. Defaults to max_seq_len.

        Returns:
            Dictionary with:
                life_state_logits: (B, S, num_life_states)
                income: (B, S)
                satisfaction: (B, S)
        """
        if seq_len is None:
            seq_len = self.max_seq_len

        batch_size = z.size(0)
        device = z.device

        # Project z to memory: (B, d_model) → (1, B, d_model)
        z_memory = self.z_proj(z).unsqueeze(0)  # (1, B, D)

        # Positional embeddings as target: (S, B, D)
        positions = torch.arange(seq_len, device=device)
        tgt = self.pos_embed(positions)  # (S, D)
        tgt = tgt.unsqueeze(1).expand(-1, batch_size, -1)  # (S, B, D)

        # Causal mask for self-attention
        causal_mask = self._get_causal_mask(seq_len, device)

        # Transformer decoder: cross-attend to z_memory
        decoder_out = self.transformer(
            tgt=tgt,
            memory=z_memory,
            tgt_mask=causal_mask,
        )  # (S, B, D)

        # Transpose to (B, S, D)
        decoder_out = decoder_out.transpose(0, 1)

        # Output heads
        life_state_logits = self.life_state_head(decoder_out)  # (B, S, C)
        income = self.income_head(decoder_out).squeeze(-1)  # (B, S)
        satisfaction = self.satisfaction_head(decoder_out).squeeze(-1)  # (B, S)

        return {
            "life_state_logits": life_state_logits,
            "income": income,
            "satisfaction": satisfaction,
        }
