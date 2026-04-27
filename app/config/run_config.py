from __future__ import annotations

import os
import random
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SimulationRunConfig:
    """Configuration that controls simulation determinism and behavior."""
    seed: int | str = field(default_factory=lambda: random.randint(0, 2**32))
    steps_before_fork: int = 5
    fork_agent_exclude_radius: int = 10
    max_nearby_agents: int = 0

    @staticmethod
    def from_env() -> "SimulationRunConfig":
        """Create a SimulationRunConfig by reading environment variables.

        Returns:
            A SimulationRunConfig instance.
        """
        defaults = SimulationRunConfig()
        return SimulationRunConfig(
            seed=os.environ.get("HEXFOLD_SEED", defaults.seed),
            steps_before_fork=_read_positive_int_env("HEXFOLD_STEPS_BEFORE_FORK", defaults.steps_before_fork),
            fork_agent_exclude_radius=_read_non_negative_int_env("HEXFOLD_AGENT_EXCLUDE_RADIUS", defaults.fork_agent_exclude_radius),
            max_nearby_agents=_read_non_negative_int_env("HEXFOLD_MAX_NEARBY_AGENTS", defaults.max_nearby_agents),
        )

def _read_non_negative_int_env(var_name: str, default_value: int) -> int:
    """Read a non-negative integer from an environment variable.

    Args:
        var_name: The environment variable name to read.
        default_value: Fallback value if missing/invalid.

    Returns:
        A non-negative integer value from the environment, or the provided default.
    """
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default_value

    try:
        parsed_value = int(raw_value.strip())
    except ValueError:
        return default_value

    return parsed_value if parsed_value >= 0 else default_value

def _read_positive_int_env(var_name: str, default_value: int) -> int:
    """Read a positive integer from an environment variable.

    Args:
        var_name: The environment variable name to read.
        default_value: The value to use when the variable is missing or invalid.

    Returns:
        A positive integer value from the environment, or the provided default.
    """
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default_value

    try:
        parsed_value = int(raw_value.strip())
    except ValueError:
        return default_value

    return parsed_value if parsed_value > 0 else default_value