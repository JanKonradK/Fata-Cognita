"""Cyclical beta annealing schedule for VAE training.

Implements the cyclical annealing strategy from Fu et al. 2019 to prevent
posterior collapse / KL vanishing.
"""

from __future__ import annotations


class CyclicalBetaSchedule:
    """Cyclical beta annealing for KL divergence weight.

    In each cycle, beta ramps linearly from 0 to 1 over the first ``ratio``
    fraction of steps, then holds at 1 for the remainder.

    Args:
        total_steps: Total number of training steps (epochs * batches_per_epoch).
        n_cycles: Number of annealing cycles.
        ratio: Fraction of each cycle spent ramping up (0 < ratio <= 1).
    """

    def __init__(
        self, total_steps: int, n_cycles: int = 4, ratio: float = 0.5
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if n_cycles <= 0:
            raise ValueError("n_cycles must be positive")
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")

        self.total_steps = total_steps
        self.n_cycles = n_cycles
        self.ratio = ratio
        self._step = 0

    def step(self) -> None:
        """Advance the schedule by one step."""
        self._step += 1

    def get_beta(self) -> float:
        """Get the current beta value.

        Returns:
            Beta value in [0, 1].
        """
        cycle_length = self.total_steps / self.n_cycles
        if cycle_length == 0:
            return 1.0

        tau = (self._step % cycle_length) / cycle_length
        if tau <= self.ratio:
            return tau / self.ratio
        return 1.0

    @property
    def current_step(self) -> int:
        """Current step count."""
        return self._step

    def reset(self) -> None:
        """Reset the schedule to step 0."""
        self._step = 0
