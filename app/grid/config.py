from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HexGridConfig:
    """Immutable configuration for a pointy-top hex grid."""

    base_padding_px: float = 16.0
    target_cols: int = 18
    target_rows: int = 7

    @staticmethod
    def from_env() -> "HexGridConfig":
        """Create a HexGridConfig by reading optional environment variables.

        Environment variables:
            HEXFOLD_COLS: Desired number of columns (int). Defaults to 18.
            HEXFOLD_ROWS: Desired number of rows (int). Defaults to 7.

        Returns:
            A HexGridConfig instance with values taken from the environment when valid,
            otherwise falling back to defaults.
        """
        defaults = HexGridConfig()

        return HexGridConfig(
            base_padding_px=defaults.base_padding_px,
            target_cols=_read_positive_int_env("HEXFOLD_COLS", defaults.target_cols),
            target_rows=_read_positive_int_env("HEXFOLD_ROWS", defaults.target_rows),
        )


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
