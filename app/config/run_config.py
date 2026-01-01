from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationRunConfig:
    """Configuration that controls simulation determinism and behavior."""
    seed: int

    @staticmethod
    def from_env() -> "SimulationRunConfig":
        """Create a SimulationRunConfig by reading environment variables.

        Environment variables:
            HEXFOLD_SEED: Optional integer seed. If missing/invalid, a random seed is generated.

        Returns:
            A SimulationRunConfig instance.
        """
        raw = os.getenv("HEXFOLD_SEED")
        if raw is not None:
            try:
                seed = int(raw.strip())
                return SimulationRunConfig(seed=seed)
            except ValueError:
                pass

        seed = random.SystemRandom().randrange(0, 2**32)
        return SimulationRunConfig(seed=seed)
