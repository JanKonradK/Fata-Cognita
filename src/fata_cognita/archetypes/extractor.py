"""Archetype extraction: encode all data to latent space, fit GMM with BIC selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

if TYPE_CHECKING:
    from fata_cognita.model.vae import TrajectoryVAE

logger = logging.getLogger(__name__)


def encode_all(
    model: TrajectoryVAE,
    static_features: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode all individuals to latent mu vectors.

    Args:
        model: Trained VAE model.
        static_features: All static features, shape (N, F).
        device: Compute device.
        batch_size: Batch size for encoding.

    Returns:
        NumPy array of shape (N, latent_dim) with mu vectors.
    """
    model.eval()
    all_mu = []

    with torch.no_grad():
        for i in range(0, len(static_features), batch_size):
            batch = static_features[i : i + batch_size].to(device)
            mu, _ = model.encode(batch)
            all_mu.append(mu.cpu().numpy())

    return np.concatenate(all_mu, axis=0)


def fit_gmm_with_bic(
    z: np.ndarray,
    k_range: tuple[int, int] = (3, 20),
    n_init: int = 5,
) -> tuple[GaussianMixture, int, dict[int, float]]:
    """Fit Gaussian Mixture Models and select the best k via BIC.

    Args:
        z: Latent representations, shape (N, latent_dim).
        k_range: Range of k values to try (inclusive).
        n_init: Number of initializations per k.

    Returns:
        Tuple of (best GMM, optimal k, dict of k->BIC scores).
    """
    bic_scores: dict[int, float] = {}

    for k in range(k_range[0], k_range[1] + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=n_init,
            max_iter=300,
            random_state=42,
        )
        gmm.fit(z)
        bic_scores[k] = gmm.bic(z)
        logger.info("GMM k=%d, BIC=%.2f", k, bic_scores[k])

    # Select k with minimum BIC
    k_best = min(bic_scores, key=bic_scores.get)  # type: ignore[arg-type]
    logger.info("Selected k=%d (BIC=%.2f)", k_best, bic_scores[k_best])

    # Refit with more initializations for stability
    best_gmm = GaussianMixture(
        n_components=k_best,
        covariance_type="full",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    best_gmm.fit(z)

    return best_gmm, k_best, bic_scores


def assign_archetypes(gmm: GaussianMixture, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign archetype labels to encoded individuals.

    Args:
        gmm: Fitted GMM.
        z: Latent representations, shape (N, latent_dim).

    Returns:
        Tuple of (hard_labels (N,), soft_probs (N, k)).
    """
    hard = gmm.predict(z)
    soft = gmm.predict_proba(z)
    return hard, soft


def save_gmm(gmm: GaussianMixture, path: str | Path) -> None:
    """Save a fitted GMM to a safe NumPy .npz file.

    Args:
        gmm: Fitted GaussianMixture model.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weights=gmm.weights_,
        means=gmm.means_,
        covariances=gmm.covariances_,
        precisions_cholesky=gmm.precisions_cholesky_,
        n_components=np.array(gmm.n_components),
        converged=np.array(gmm.converged_),
        n_iter=np.array(gmm.n_iter_),
        lower_bound=np.array(gmm.lower_bound_),
    )


def load_gmm(path: str | Path) -> GaussianMixture:
    """Load a saved GMM from a safe NumPy .npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        Fitted GaussianMixture model.
    """
    data = np.load(path, allow_pickle=False)
    n_components = int(data["n_components"])
    gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.weights_ = data["weights"]
    gmm.means_ = data["means"]
    gmm.covariances_ = data["covariances"]
    gmm.precisions_cholesky_ = data["precisions_cholesky"]
    gmm.converged_ = bool(data["converged"])
    gmm.n_iter_ = int(data["n_iter"])
    gmm.lower_bound_ = float(data["lower_bound"])
    gmm.n_features_in_ = gmm.means_.shape[1]
    return gmm
