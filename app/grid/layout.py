from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from types import MappingProxyType
from typing import Dict, Iterable, Mapping

from app.config import HexGridConfig

type Point = tuple[float, float]
type VertexKey = tuple[int, int]
type EdgeKey = tuple[VertexKey, VertexKey]


@dataclass(frozen=True, slots=True)
class HoneyVertex2D:
    """A honeycomb vertex identified by an integer lattice key, with cached screen position."""
    key: VertexKey
    px: Point

    @property
    def q(self) -> int:
        """Return the q component of the vertex key.

        Returns:
            The q coordinate.
        """
        return self.key[0]

    @property
    def r(self) -> int:
        """Return the r component of the vertex key.

        Returns:
            The r coordinate.
        """
        return self.key[1]


@dataclass(frozen=True, slots=True)
class HoneyEdge:
    """An undirected honeycomb edge between two vertex keys (canonicalized)."""
    start_key: VertexKey
    end_key: VertexKey

    def endpoints_px(self, vertices_by_key: Mapping[VertexKey, HoneyVertex2D]) -> tuple[Point, Point]:
        """Resolve this edge into its screen-space endpoints.

        Args:
            vertices_by_key: Mapping from vertex key to vertex object.

        Returns:
            A tuple of two points (start_px, end_px) representing the line segment endpoints.
        """
        start_px = vertices_by_key[self.start_key].px
        end_px = vertices_by_key[self.end_key].px
        return (start_px, end_px)


@dataclass(frozen=True, slots=True)
class HexGridLayout:
    """Computed layout: hex sizing + derived honeycomb vertices/edges.

    Notes:
        - Vertex keys are the source of truth (pure integer lattice).
        - Pixel positions are a projection for rendering.
    """
    radius_px: float
    vertex_offsets_px: tuple[Point, ...]
    hex_centers_px: tuple[Point, ...]
    lattice_origin_px: Point
    vertices: tuple[HoneyVertex2D, ...]
    edges: tuple[HoneyEdge, ...]
    vertices_by_key: Mapping[VertexKey, HoneyVertex2D]


# Corner vertex offsets in the same (q, r) lattice as vertex keys.
# Order matches pointy-top corners starting at angle -90° and advancing +60°:
#   top, top-right, bottom-right, bottom, bottom-left, top-left.
_CORNER_LATTICE_OFFSETS: tuple[VertexKey, ...] = (
    (-1, -1),
    (0, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 0),
)


def compute_vertex_offsets(radius_px: float) -> tuple[Point, ...]:
    """Compute the six corner offsets for a pointy-top hexagon in screen space.

    This avoids trig and stays consistent with the integer lattice used for vertex keys.

    Args:
        radius_px: Distance from center to each vertex in pixels (also the honeycomb edge length).

    Returns:
        Tuple of (dx, dy) offsets for the 6 corners, matching _CORNER_LATTICE_OFFSETS order.
    """
    return tuple(_lattice_offset_to_px(lattice_offset, radius_px) for lattice_offset in _CORNER_LATTICE_OFFSETS)


def compute_hex_grid_layout(window_width_px: int, window_height_px: int, cfg: HexGridConfig) -> HexGridLayout:
    """Compute a centered hex grid layout and derive the honeycomb graph (vertices/edges).

    Resize semantics:
        - cfg.target_cols / cfg.target_rows stay constant
        - resizing changes only the radius (scale), not the count of hexes

    Key strategy:
        - Hex centers are enumerated by (row_index, col_index).
        - Each center maps deterministically to an integer lattice key (q, r).
        - Corner vertex keys are computed by integer addition: center_key + corner_offset.
        - No screen-space inversion / rounding is required.

    Args:
        window_width_px: Window width in pixels.
        window_height_px: Window height in pixels.
        cfg: Grid configuration.

    Returns:
        A HexGridLayout containing radius, hex centers, honeycomb vertices and edges.
    """
    safe_window_width_px = max(1, window_width_px)
    safe_window_height_px = max(1, window_height_px)

    target_column_count = max(1, cfg.target_cols)
    target_row_count = max(1, cfg.target_rows)

    available_width_px = max(1.0, safe_window_width_px - 2 * cfg.base_padding_px)
    available_height_px = max(1.0, safe_window_height_px - 2 * cfg.base_padding_px)

    radius_from_width_px = (available_width_px / target_column_count) / sqrt(3.0)
    radius_from_height_px = available_height_px / (1.5 * target_row_count + 0.5)
    radius_px = max(2.0, min(radius_from_width_px, radius_from_height_px))

    hex_width_px = sqrt(3.0) * radius_px
    center_step_x_px = hex_width_px
    row_step_y_px = 1.5 * radius_px

    covered_width_px = target_column_count * hex_width_px
    covered_height_px = (1.5 * target_row_count + 0.5) * radius_px

    padding_x_px = cfg.base_padding_px + max(0.0, (available_width_px - covered_width_px) / 2.0)
    padding_y_px = cfg.base_padding_px + max(0.0, (available_height_px - covered_height_px) / 2.0)

    vertex_offsets_px = compute_vertex_offsets(radius_px)

    hex_centers_px: list[Point] = []
    vertices_by_key_mut: Dict[VertexKey, HoneyVertex2D] = {}
    undirected_edge_keys: set[EdgeKey] = set()

    for row_index, col_index, hex_center_px in _iter_hex_centers(
        padding_x_px=padding_x_px,
        padding_y_px=padding_y_px,
        radius_px=radius_px,
        hex_width_px=hex_width_px,
        center_step_x_px=center_step_x_px,
        row_step_y_px=row_step_y_px,
        target_column_count=target_column_count,
        target_row_count=target_row_count,
    ):
        center_x_px, center_y_px = hex_center_px
        hex_centers_px.append(hex_center_px)

        center_lattice_key = _center_key_from_row_col(row_index, col_index)
        corner_vertex_keys = _corner_keys_from_center_key(center_lattice_key)

        for vertex_key, (offset_x_px, offset_y_px) in zip(corner_vertex_keys, vertex_offsets_px, strict=True):
            if vertex_key in vertices_by_key_mut:
                continue
            vertex_px = (center_x_px + offset_x_px, center_y_px + offset_y_px)
            vertices_by_key_mut[vertex_key] = HoneyVertex2D(key=vertex_key, px=vertex_px)

        for corner_index in range(6):
            first_corner_key = corner_vertex_keys[corner_index]
            second_corner_key = corner_vertex_keys[(corner_index + 1) % 6]
            undirected_edge_keys.add(_make_edge_key(first_corner_key, second_corner_key))

    # row=0, col=0 maps to center_key (0,0), and hex_centers_px[0] is its pixel position.
    lattice_origin_px = hex_centers_px[0] if hex_centers_px else (0.0, 0.0)

    sorted_vertex_keys = sorted(vertices_by_key_mut.keys())
    vertices = tuple(vertices_by_key_mut[vertex_key] for vertex_key in sorted_vertex_keys)

    edges = tuple(
        HoneyEdge(start_key=edge_start_key, end_key=edge_end_key)
        for (edge_start_key, edge_end_key) in sorted(undirected_edge_keys)
    )

    vertices_by_key: Mapping[VertexKey, HoneyVertex2D] = MappingProxyType(dict(vertices_by_key_mut))

    return HexGridLayout(
        radius_px=radius_px,
        vertex_offsets_px=vertex_offsets_px,
        hex_centers_px=tuple(hex_centers_px),
        lattice_origin_px=lattice_origin_px,
        vertices=vertices,
        edges=edges,
        vertices_by_key=vertices_by_key,
    )


def _iter_hex_centers(
    padding_x_px: float,
    padding_y_px: float,
    radius_px: float,
    hex_width_px: float,
    center_step_x_px: float,
    row_step_y_px: float,
    target_column_count: int,
    target_row_count: int,
) -> Iterable[tuple[int, int, Point]]:
    """Generate exactly target_row_count * target_column_count hex center positions.

    Args:
        padding_x_px: Effective left/right padding.
        padding_y_px: Effective top/bottom padding.
        radius_px: Hex radius in pixels.
        hex_width_px: Hex width (sqrt(3) * radius).
        center_step_x_px: Horizontal center-to-center step.
        row_step_y_px: Vertical row step.
        target_column_count: Number of centers per row.
        target_row_count: Number of rows.

    Yields:
        (row_index, col_index, (center_x_px, center_y_px)) for each hex center.
    """
    first_center_x_px = padding_x_px + hex_width_px / 2.0
    first_center_y_px = padding_y_px + radius_px

    for row_index in range(target_row_count):
        center_y_px = first_center_y_px + row_index * row_step_y_px
        odd_row_x_offset_px = (hex_width_px / 2.0) if (row_index % 2 == 1) else 0.0

        for col_index in range(target_column_count):
            center_x_px = first_center_x_px + odd_row_x_offset_px + col_index * center_step_x_px
            yield (row_index, col_index, (center_x_px, center_y_px))


def _center_key_from_row_col(row_index: int, col_index: int) -> VertexKey:
    """Convert (row_index, col_index) in an odd-row offset layout to a lattice key (q, r).

    This mapping places hex centers onto the same (q, r) lattice used for corner vertex keys.
    Adjacent hexagons therefore share identical corner vertex keys, enabling stable deduplication.

    Args:
        row_index: Zero-based row index.
        col_index: Zero-based column index.

    Returns:
        Integer lattice key (q, r) for the hex center.
    """
    vertical_lattice_step = (3 * row_index) // 2
    center_q = col_index + vertical_lattice_step + (row_index & 1)
    center_r = -col_index + vertical_lattice_step
    return (center_q, center_r)


def _corner_keys_from_center_key(center_key: VertexKey) -> tuple[VertexKey, ...]:
    """Compute the six corner vertex keys for a given hex center key.

    Args:
        center_key: The integer lattice key (q, r) of the hex center.

    Returns:
        Six vertex keys (q, r) in the same order as _CORNER_LATTICE_OFFSETS.
    """
    center_q, center_r = center_key
    return tuple(
        (center_q + offset_q, center_r + offset_r)
        for (offset_q, offset_r) in _CORNER_LATTICE_OFFSETS
    )


def _lattice_offset_to_px(lattice_offset: VertexKey, step_px: float) -> Point:
    """Convert an integer lattice offset (dq, dr) to a pixel offset (dx, dy).

    Lattice basis (edge length = step_px):
        u = (sqrt(3)/2 * s, 1/2 * s)
        v = (-sqrt(3)/2 * s, 1/2 * s)

    Forward:
        dx = (dq - dr) * a
        dy = (dq + dr) * b
    where:
        a = sqrt(3)/2 * s
        b = 1/2 * s

    Args:
        lattice_offset: Lattice offset (dq, dr).
        step_px: Lattice edge length in pixels (hex side length == radius).

    Returns:
        Pixel offset (dx, dy).
    """
    delta_q, delta_r = lattice_offset
    basis_x_px = (sqrt(3.0) / 2.0) * step_px
    basis_y_px = 0.5 * step_px
    offset_x_px = (delta_q - delta_r) * basis_x_px
    offset_y_px = (delta_q + delta_r) * basis_y_px
    return (offset_x_px, offset_y_px)


def _make_edge_key(first_vertex_key: VertexKey, second_vertex_key: VertexKey) -> EdgeKey:
    """Create a canonical undirected edge key.

    Args:
        first_vertex_key: First vertex key.
        second_vertex_key: Second vertex key.

    Returns:
        Canonical undirected edge key with sorted endpoints.
    """
    return (
        (first_vertex_key, second_vertex_key)
        if first_vertex_key <= second_vertex_key
        else (second_vertex_key, first_vertex_key)
    )
