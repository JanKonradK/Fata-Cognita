"""Tests for the Transformer decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fata_cognita.model.decoder import TrajectoryDecoder

if TYPE_CHECKING:
    from fata_cognita.config import Config


def test_decoder_output_shapes(tiny_config: Config):
    """Decoder produces outputs with correct shapes."""
    dec = TrajectoryDecoder(
        config=tiny_config.model,
        max_seq_len=tiny_config.data.max_seq_len,
        num_life_states=tiny_config.data.num_life_states,
    )

    z = torch.randn(8, tiny_config.model.latent_dim)
    out = dec(z)

    seq_len = tiny_config.data.max_seq_len
    assert out["life_state_logits"].shape == (8, seq_len, 9)
    assert out["income"].shape == (8, seq_len)
    assert out["satisfaction"].shape == (8, seq_len)


def test_decoder_satisfaction_bounded(tiny_config: Config):
    """Satisfaction output is in [0, 1] due to sigmoid."""
    dec = TrajectoryDecoder(
        config=tiny_config.model,
        max_seq_len=tiny_config.data.max_seq_len,
        num_life_states=tiny_config.data.num_life_states,
    )

    z = torch.randn(8, tiny_config.model.latent_dim)
    out = dec(z)

    assert (out["satisfaction"] >= 0).all()
    assert (out["satisfaction"] <= 1).all()


def test_decoder_custom_seq_len(tiny_config: Config):
    """Decoder respects custom sequence length."""
    dec = TrajectoryDecoder(
        config=tiny_config.model,
        max_seq_len=tiny_config.data.max_seq_len,
        num_life_states=tiny_config.data.num_life_states,
    )

    z = torch.randn(4, tiny_config.model.latent_dim)
    out = dec(z, seq_len=5)

    assert out["life_state_logits"].shape == (4, 5, 9)
    assert out["income"].shape == (4, 5)


def test_decoder_causal_mask(tiny_config: Config):
    """Causal mask prevents future information leaking to earlier positions."""
    dec = TrajectoryDecoder(
        config=tiny_config.model,
        max_seq_len=tiny_config.data.max_seq_len,
        num_life_states=tiny_config.data.num_life_states,
    )
    dec.eval()

    z = torch.randn(1, tiny_config.model.latent_dim)

    # Get output with full sequence
    out_full = dec(z, seq_len=tiny_config.data.max_seq_len)

    # Position 0 output should depend only on z and position 0 embedding.
    # Run with seq_len=1 and compare.
    out_single = dec(z, seq_len=1)

    # The first position should produce the same logits regardless of sequence length
    # (due to causal masking, position 0 can't see positions 1+)
    assert torch.allclose(
        out_full["life_state_logits"][0, 0],
        out_single["life_state_logits"][0, 0],
        atol=1e-5,
    )
