"""Tests for archetype extraction."""

from __future__ import annotations

import numpy as np

from fata_cognita.archetypes.extractor import assign_archetypes, fit_gmm_with_bic


def test_gmm_recovers_clusters():
    """GMM recovers known clusters from well-separated blobs."""
    rng = np.random.default_rng(42)
    # 3 well-separated clusters in 2D
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(100, 2))
    c2 = rng.normal(loc=[5, 5], scale=0.3, size=(100, 2))
    c3 = rng.normal(loc=[10, 0], scale=0.3, size=(100, 2))
    z = np.vstack([c1, c2, c3])

    gmm, k_best, bic_scores = fit_gmm_with_bic(z, k_range=(2, 6), n_init=3)

    # Should find 3 clusters
    assert k_best == 3
    assert len(bic_scores) == 5  # k=2,3,4,5,6


def test_assign_archetypes():
    """Assignment produces correct shapes."""
    rng = np.random.default_rng(42)
    z = rng.normal(size=(50, 4))

    gmm, _, _ = fit_gmm_with_bic(z, k_range=(2, 4), n_init=2)
    hard, soft = assign_archetypes(gmm, z)

    assert hard.shape == (50,)
    assert soft.shape[0] == 50
    assert soft.shape[1] == gmm.n_components
    # Soft probs should sum to ~1
    np.testing.assert_allclose(soft.sum(axis=1), 1.0, atol=1e-6)
