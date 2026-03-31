"""Archetype discovery via GMM clustering in the latent space."""

from fata_cognita.archetypes.extractor import (
    assign_archetypes,
    encode_all,
    fit_gmm_with_bic,
    load_gmm,
    save_gmm,
)
from fata_cognita.archetypes.profiler import ArchetypeProfile, profile_archetypes
from fata_cognita.archetypes.visualizer import plot_archetype_trajectories, plot_latent_space

__all__ = [
    "ArchetypeProfile",
    "assign_archetypes",
    "encode_all",
    "fit_gmm_with_bic",
    "load_gmm",
    "plot_archetype_trajectories",
    "plot_latent_space",
    "profile_archetypes",
    "save_gmm",
]
