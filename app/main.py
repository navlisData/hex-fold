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

        self._cfg = HexGridConfig.from_env()
        self._layout = None
        self._last_size = (-1, -1)

    def draw(self) -> None:
        """Redraw the honeycomb each frame and rebuild layout on resize."""
        self._ensure_layout()

        self.background(245)
        assert self._layout is not None
        assert self._cfg is not None

        if self._cfg.debug:
            self._draw_debug(self._layout)

    def _ensure_layout(self) -> None:
        """Update cached layout if the window size changed."""
        size = (self.width, self.height)
        if size == self._last_size and self._layout is not None:
            return

        assert self._cfg is not None
        self._layout = compute_hex_grid_layout(self.width, self.height, self._cfg)
        self._last_size = size

    def _draw_debug(self, layout: HexGridLayout) -> None:
        """Draw debug overlays: vertices and sparse axial labels.

        Args:
            layout: Precomputed layout.

        Returns:
            None.
        """
        # Vertices
        self.no_stroke()
        self.fill(255, 0, 0, 90)
        for v in layout.vertices:
            self.circle(v.px[0], v.px[1], 6)
        self.no_fill()

        # Sparse labels
        self.no_stroke()
        self.fill(0, 70)
        self.text_size(9)
        self.text_align(self.LEFT, self.TOP)
        for v in layout.vertices:
            if (v.q % 2 == 0) and (v.r % 2 == 0):
                self.text(f"({v.q},{v.r})", v.px[0] + 3, v.px[1] + 3)

        # Draw honeycomb edges
        self.stroke(40, 40)
        self.stroke_weight(1)
        for e in layout.edges:
            a_px, b_px = e.endpoints_px(layout.vertices_by_key)
            self.line(a_px[0], a_px[1], b_px[0], b_px[1])

        self.no_fill()


app = HexFold()
app.run_sketch()
