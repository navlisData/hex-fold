from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt, pi
from typing import Iterable

from .config import HexGridConfig

type Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class HexGridLayout:
    """Computed layout for rendering a hex grid."""

    radius_px: float
    vertex_offsets: tuple[Point, ...]
    centers: tuple[Point, ...]


def compute_vertex_offsets(radius_px: float) -> tuple[Point, ...]:
    """Compute the six vertex offsets for a pointy-top hexagon.

    Args:
        radius_px: Distance from center to each vertex in pixels.

    Returns:
        Tuple of (dx, dy) offsets for the 6 vertices.
    """
    offsets: list[Point] = []
    for i in range(6):
        angle_rad = -pi / 2.0 + i * (pi / 3.0)
        offsets.append((radius_px * cos(angle_rad), radius_px * sin(angle_rad)))
    return tuple(offsets)


def compute_hex_grid_layout(window_width_px: int, window_height_px: int, cfg: HexGridConfig) -> HexGridLayout:
    """Compute a centered hex grid layout for the given window size and configuration.

    Args:
        window_width_px: Window width in pixels.
        window_height_px: Window height in pixels.
        cfg: Grid configuration (padding and target rows/cols).

    Returns:
        A HexGridLayout containing radius, vertex offsets and all hex centers.
    """
    window_width_px = max(1, window_width_px)
    window_height_px = max(1, window_height_px)

    target_columns = max(1, cfg.target_cols)
    target_rows = max(1, cfg.target_rows)

    available_width_px = max(1.0, window_width_px - 2.0 * cfg.base_padding_px)
    available_height_px = max(1.0, window_height_px - 2.0 * cfg.base_padding_px)

    radius_from_available_width = (available_width_px / target_columns) / sqrt(3.0)
    radius_from_available_height = available_height_px / (1.5 * target_rows + 0.5)

    radius_px = max(2.0, min(radius_from_available_width, radius_from_available_height))

    hexagon_width_px = sqrt(3.0) * radius_px
    center_step_x_px = hexagon_width_px
    row_step_y_px = 1.5 * radius_px

    covered_width_px = target_columns * hexagon_width_px
    covered_height_px = (1.5 * target_rows + 0.5) * radius_px

    padding_x_px = cfg.base_padding_px + max(0.0, (available_width_px - covered_width_px) / 2.0)
    padding_y_px = cfg.base_padding_px + max(0.0, (available_height_px - covered_height_px) / 2.0)

    vertex_offsets = compute_vertex_offsets(radius_px)
    centers = tuple(
        _compute_centers(
            window_width_px=window_width_px,
            window_height_px=window_height_px,
            padding_x_px=padding_x_px,
            padding_y_px=padding_y_px,
            radius_px=radius_px,
            hexagon_width_px=hexagon_width_px,
            center_step_x_px=center_step_x_px,
            row_step_y_px=row_step_y_px,
        )
    )

    return HexGridLayout(radius_px=radius_px, vertex_offsets=vertex_offsets, centers=centers)


def _compute_centers(
    window_width_px: int,
    window_height_px: int,
    padding_x_px: float,
    padding_y_px: float,
    radius_px: float,
    hexagon_width_px: float,
    center_step_x_px: float,
    row_step_y_px: float,
) -> Iterable[Point]:
    """Generate centers for hexagons that fit inside the padded bounds.

    Args:
        window_width_px: Window width in pixels.
        window_height_px: Window height in pixels.
        padding_x_px: Effective left/right padding.
        padding_y_px: Effective top/bottom padding.
        radius_px: Hex radius in pixels.
        hexagon_width_px: Hex width (sqrt(3) * radius).
        center_step_x_px: Horizontal center-to-center step.
        row_step_y_px: Vertical row step.

    Yields:
        (center_x_px, center_y_px) centers for each hexagon.
    """
    min_center_x_px = padding_x_px + hexagon_width_px / 2.0
    max_center_x_px = window_width_px - padding_x_px - hexagon_width_px / 2.0
    min_center_y_px = padding_y_px + radius_px
    max_center_y_px = window_height_px - padding_y_px - radius_px

    row_index = 0
    current_center_y_px = min_center_y_px

    while current_center_y_px <= max_center_y_px + 1e-6:
        row_offset_x_px = (hexagon_width_px / 2.0) if (row_index % 2 == 1) else 0.0
        current_center_x_px = min_center_x_px + row_offset_x_px

        while current_center_x_px <= max_center_x_px + 1e-6:
            yield (current_center_x_px, current_center_y_px)
            current_center_x_px += center_step_x_px

        row_index += 1
        current_center_y_px = min_center_y_px + row_index * row_step_y_px
