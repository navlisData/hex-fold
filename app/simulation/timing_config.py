from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationTimingConfig:
    """Timing configuration for time-based animation and stepping."""
    edge_traverse_ms: int = 180
    travel_vertex_dwell_ms: int = 120

    @staticmethod
    def from_env() -> "SimulationTimingConfig":
        """Create a SimulationTimingConfig by reading environment variables.

        Environment variables:
            HEXFOLD_EDGE_TRAVERSE_MS: Positive int. Edge traversal duration in ms.
            HEXFOLD_TRAVEL_DWELL_MS: Non-negative int. Dwell duration at travel vertices in ms.

        Returns:
            A SimulationTimingConfig instance with values taken from the environment when valid,
            otherwise falling back to defaults.
        """
        defaults = SimulationTimingConfig()
        return SimulationTimingConfig(
            edge_traverse_ms=_read_positive_int_env("HEXFOLD_EDGE_TRAVERSE_MS", defaults.edge_traverse_ms),
            travel_vertex_dwell_ms=_read_non_negative_int_env("HEXFOLD_TRAVEL_DWELL_MS", defaults.travel_vertex_dwell_ms),
        )


def _read_positive_int_env(var_name: str, default_value: int) -> int:
    """Read a positive integer from an environment variable.

    Args:
        var_name: The environment variable name to read.
        default_value: Fallback value if missing/invalid.

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
