from __future__ import annotations

from typing import Protocol, overload

from app.grid.layout import HexGridLayout


class _DebugSketch(Protocol):
    LEFT: int
    TOP: int

    @overload
    def fill(self, gray: int, alpha: int) -> None: ...
    @overload
    def fill(self, r: int, g: int, b: int, a: int) -> None: ...

    def stroke(self, gray: int, alpha: int) -> None: ...
    def stroke_weight(self, weight: float) -> None: ...
    def no_stroke(self) -> None: ...
    def no_fill(self) -> None: ...
    def circle(self, x: float, y: float, d: float) -> None: ...
    def line(self, x1: float, y1: float, x2: float, y2: float) -> None: ...
    def text_size(self, size: float) -> None: ...
    def text_align(self, horiz_align: int, vert_align: int) -> None: ...
    def text(self, s: str, x: float, y: float) -> None: ...


def draw_debug_overlays(sketch: _DebugSketch, layout: HexGridLayout) -> None:
    """Draw debug overlays: vertices and sparse axial labels, plus honeycomb edges.

    Args:
        sketch: The py5 sketch (or compatible protocol).
        layout: Precomputed layout.
    """
    # Vertices
    sketch.no_stroke()
    sketch.fill(255, 0, 0, 90)
    for v in layout.vertices:
        sketch.circle(v.px[0], v.px[1], 6)
    sketch.no_fill()

    # Sparse labels
    sketch.no_stroke()
    sketch.fill(0, 70)
    sketch.text_size(11)
    sketch.text_align(sketch.LEFT, sketch.TOP)
    for v in layout.vertices:
        if (v.q % 2 == 0) and (v.r % 2 == 0):
            sketch.text(f"({v.q},{v.r})", v.px[0] + 3, v.px[1] + 3)
    sketch.no_fill()

    # Draw honeycomb edges
    sketch.stroke(40, 40)
    sketch.stroke_weight(1)
    for e in layout.edges:
        a_px, b_px = e.endpoints_px(layout.vertices_by_key)
        sketch.line(a_px[0], a_px[1], b_px[0], b_px[1])

    sketch.no_fill()
