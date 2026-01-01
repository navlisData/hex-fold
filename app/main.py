from __future__ import annotations

import random
import app.rendering as rendering

from py5 import Sketch

from app.graph.growth_stepper import Agent, GrowthStepper
from app.graph.honey_graph import HoneyGraph
from app.grid import HexGridConfig, HexGridLayout, compute_hex_grid_layout
from app.simulation.agent_controller import AgentController
from app.simulation.timing_config import SimulationTimingConfig


class HexFold(Sketch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cfg: HexGridConfig | None = None
        self._timing: SimulationTimingConfig | None = None
        self._layout: HexGridLayout | None = None
        self._graph: HoneyGraph | None = None
        self._last_size: tuple[int, int] = (-1, -1)

        self._rng = random.Random()
        self._controller: AgentController | None = None

    def settings(self) -> None:
        self.size(600, 200, self.P2D)

    def setup(self) -> None:
        """Initialize sketch state and enable user window resizing."""
        self.window_resizable(True)
        self.frame_rate(60)
        self.no_fill()

        self._cfg = HexGridConfig.from_env()
        self._timing = SimulationTimingConfig.from_env()

        self._layout = None
        self._graph = None
        self._controller = None
        self._last_size = (-1, -1)

    def draw(self) -> None:
        """Redraw the honeycomb each frame and animate agents time-based."""
        self._ensure_layout_and_sim()

        assert self._layout is not None
        assert self._graph is not None
        assert self._controller is not None
        assert self._cfg is not None

        now_ms = int(self.millis())
        self._controller.update(now_ms, self._layout, self._graph)

        self.background(245)
        rendering.draw_active_edges(self, self._layout, self._graph)
        rendering.draw_agents(self, self._layout, self._controller.get_drawables())

        if self._cfg.debug:
            self._draw_debug(self._layout)

    def _ensure_layout_and_sim(self) -> None:
        """Rebuild layout+graph on resize and reset the simulation accordingly."""
        size = (self.width, self.height)
        if size == self._last_size and self._layout is not None and self._graph is not None and self._controller is not None:
            return

        assert self._cfg is not None
        assert self._timing is not None

        self._layout = compute_hex_grid_layout(self.width, self.height, self._cfg)
        self._graph = HoneyGraph(self._layout)
        self._last_size = size

        stepper = GrowthStepper(self._rng)
        self._controller = AgentController(stepper=stepper, timing=self._timing)
        self._controller.add_agent(Agent())

    def _draw_debug(self, layout: HexGridLayout) -> None:
        """Draw debug overlays: vertices and sparse axial labels.

        Args:
            layout: Precomputed layout.
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
        self.text_size(11)
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
