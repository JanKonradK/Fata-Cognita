"""VAE-Transformer model architecture."""

from fata_cognita.model.beta_schedule import CyclicalBetaSchedule
from fata_cognita.model.decoder import TrajectoryDecoder
from fata_cognita.model.encoder import Encoder
from fata_cognita.model.loss import LossComponents, MultiTaskLoss
from fata_cognita.model.vae import TrajectoryVAE

__all__ = [
    "CyclicalBetaSchedule",
    "Encoder",
    "LossComponents",
    "MultiTaskLoss",
    "TrajectoryDecoder",
    "TrajectoryVAE",
]
