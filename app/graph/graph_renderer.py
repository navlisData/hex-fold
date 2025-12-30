from __future__ import annotations

from typing import Protocol

from app.graph.honey_graph import HoneyGraph
from app.grid.layout import HexGridLayout


class _StrokeSketch(Protocol):
    """Minimal sketch protocol for drawing operations used by the renderer."""
    def stroke(self, *args) -> None: ...
    def stroke_weight(self, weight: float) -> None: ...
    def line(self, x1: float, y1: float, x2: float, y2: float) -> None: ...


class GraphRenderer:
    """Render active edges from the graph using the layout's pixel projection."""

    def draw_active_edges(self, sketch: _StrokeSketch, layout: HexGridLayout, graph: HoneyGraph) -> None:
        """Draw all existing edges.

        Args:
            sketch: The py5 sketch (or compatible protocol).
            layout: Provides vertex key -> pixel coordinate mapping.
            graph: Provides which edges exist and their traffic values.

        Returns:
            None.
        """
        sketch.stroke(40,40)

        for (a, b) in graph.iter_active_edges():
            a_px = layout.vertices_by_key[a].px
            b_px = layout.vertices_by_key[b].px

            traffic = graph.edge_state(a, b).traffic
            sketch.stroke_weight(self._stroke_width_from_traffic(traffic))
            sketch.line(a_px[0], a_px[1], b_px[0], b_px[1])

    @staticmethod
    def _stroke_width_from_traffic(traffic: int) -> float:
        """Compute a saturated stroke width from traffic.

        Args:
            traffic: Traversal count.

        Returns:
            A stroke width in pixels.
        """
        clamped_traffic = max(0, traffic)
        return 1.0 + min(6.0, (clamped_traffic ** 0.5))
