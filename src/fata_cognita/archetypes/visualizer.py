"""Archetype visualization: t-SNE/UMAP, trajectory panels, radar charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend


def plot_latent_space(
    z: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    method: str = "tsne",
    perplexity: float = 30.0,
) -> None:
    """Create a 2D scatter plot of the latent space colored by archetype.

    Args:
        z: Latent representations, shape (N, latent_dim).
        labels: Archetype labels, shape (N,).
        output_path: Path to save the plot.
        method: Dimensionality reduction method ('tsne' or 'umap').
        perplexity: Perplexity for t-SNE (ignored for UMAP).
    """
    if method == "umap":
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42)
        z_2d = reducer.fit_transform(z)
    else:
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        z_2d = reducer.fit_transform(z)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=labels, cmap="tab10", alpha=0.6, s=10,
    )
    plt.colorbar(scatter, ax=ax, label="Archetype")
    ax.set_title(f"Latent Space ({method.upper()})")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_archetype_trajectories(
    profiles: list[dict],
    output_path: str | Path,
    ages: list[int] | None = None,
) -> None:
    """Plot canonical trajectories for all archetypes.

    Args:
        profiles: List of archetype profile dicts, each with 'canonical_trajectory'.
        output_path: Path to save the plot.
        ages: Age labels for x-axis. Defaults to 14-75.
    """
    if ages is None:
        n_steps = len(profiles[0]["canonical_trajectory"]["income"])
        ages = list(range(14, 14 + n_steps))

    n_archetypes = len(profiles)
    fig, axes = plt.subplots(n_archetypes, 2, figsize=(14, 3 * n_archetypes), squeeze=False)

    for i, profile in enumerate(profiles):
        traj = profile["canonical_trajectory"]
        aid = profile.get("archetype_id", i)

        # Income plot
        axes[i, 0].plot(ages, traj["income"], color="steelblue")
        axes[i, 0].set_ylabel("Log Income")
        axes[i, 0].set_title(f"Archetype {aid} — Income")

        # Satisfaction plot
        axes[i, 1].plot(ages, traj["satisfaction"], color="coral")
        axes[i, 1].set_ylabel("Satisfaction")
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_title(f"Archetype {aid} — Satisfaction")

    for ax in axes[-1]:
        ax.set_xlabel("Age")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
