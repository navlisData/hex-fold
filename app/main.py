from __future__ import annotations

from py5 import Sketch

from app.grid import HexGridConfig, HexGridLayout, compute_hex_grid_layout

type Point = tuple[float, float]

class HexFold(Sketch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg: HexGridConfig | None = None
        self._layout: HexGridLayout | None = None
        self._last_size: tuple[int, int] = (-1, -1)

    def settings(self) -> None:
        self.size(600, 200, self.P2D)

    def setup(self) -> None:
        """Initialize sketch state and enable user window resizing."""
        self.window_resizable(True)
        self.frame_rate(60)
        self.no_fill()
        self.stroke_weight(1)

        self._cfg = HexGridConfig.from_env()
        self._layout = None
        self._last_size = (-1, -1)

    def draw(self) -> None:
        """Redraw the honeycomb each frame and rebuild layout on resize."""
        self._ensure_layout()

        self.background(245)
        self.stroke(40)

        assert self._layout is not None
        for cx, cy in self._layout.centers:
            self._draw_hexagon(cx, cy, self._layout.vertex_offsets)

    def _ensure_layout(self) -> None:
        """Update cached layout if the window size changed."""
        size = (self.width, self.height)
        if size == self._last_size and self._layout is not None:
            return

        assert self._cfg is not None
        self._layout = compute_hex_grid_layout(self.width, self.height, self._cfg)
        self._last_size = size

    def _draw_hexagon(self, cx: float, cy: float, vertex_offsets: tuple[Point, ...]) -> None:
        """Draw a hexagon at (cx, cy) using precomputed vertex offsets.

        Args:
            cx: Center x-coordinate in pixels.
            cy: Center y-coordinate in pixels.
            vertex_offsets: (dx, dy) offsets for the 6 vertices.

        Returns:
            None.
        """
        self.begin_shape()
        for dx, dy in vertex_offsets:
            self.vertex(cx + dx, cy + dy)
        self.end_shape(self.CLOSE)


app = HexFold()
app.run_sketch()
