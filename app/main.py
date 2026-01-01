from __future__ import annotations

import random
import app.rendering as rendering

from py5 import Sketch

from app.config import SimulationTimingConfig, HexGridConfig, SimulationRunConfig
from app.graph.growth_stepper import Agent, GrowthStepper
from app.graph.honey_graph import HoneyGraph
from app.grid import HexGridLayout, compute_hex_grid_layout
from app.simulation.agent_controller import AgentController


class HexFold(Sketch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._grid_cfg: HexGridConfig | None = None
        self._timing_cfg: SimulationTimingConfig | None = None
        self._sim_cfg: SimulationRunConfig | None = None

        self._layout: HexGridLayout | None = None
        self._graph: HoneyGraph | None = None
        self._last_size: tuple[int, int] = (-1, -1)

        self._rng: random.Random | None = None
        self._controller: AgentController | None = None

    def settings(self) -> None:
        self.size(600, 200, self.P2D)

    def setup(self) -> None:
        """Initialize sketch state and enable user window resizing."""
        self.window_resizable(True)
        self.frame_rate(60)
        self.no_fill()

        self._grid_cfg = HexGridConfig.from_env()
        self._timing_cfg = SimulationTimingConfig.from_env()
        self._sim_cfg = SimulationRunConfig.from_env()

        self._rng = random.Random(self._sim_cfg.seed)

        self._layout = None
        self._graph = None
        self._controller = None
        self._last_size = (-1, -1)

        print(
            f"For reproducing this run use: "
            f"Columns={self._grid_cfg.target_cols} | "
            f"Rows={self._grid_cfg.target_rows} | "
            f"Seed={self._sim_cfg.seed}"
        )

    def draw(self) -> None:
        """Redraw the honeycomb each frame and animate agents time-based."""
        self._ensure_layout_and_sim()

        assert self._layout is not None
        assert self._graph is not None
        assert self._controller is not None
        assert self._grid_cfg is not None

        now_ms = int(self.millis())
        self._controller.update(now_ms, self._layout, self._graph)

        self.background(245)
        rendering.draw_active_edges(self, self._layout, self._graph)
        rendering.draw_agents(self, self._layout, self._controller.get_drawables())

        if self._grid_cfg.debug:
            rendering.draw_debug_overlays(self, self._layout)

    def _ensure_layout_and_sim(self) -> None:
        """Rebuild layout+graph on resize and reset the simulation accordingly."""
        size = (self.width, self.height)
        if size == self._last_size and self._layout is not None and self._graph is not None and self._controller is not None:
            return

        assert self._grid_cfg is not None
        assert self._timing_cfg is not None

        self._layout = compute_hex_grid_layout(self.width, self.height, self._grid_cfg)
        self._graph = HoneyGraph(self._layout)
        self._last_size = size

        stepper = GrowthStepper(self._rng)
        self._controller = AgentController(stepper=stepper, timing=self._timing_cfg)
        self._controller.add_agent(Agent())


app = HexFold()
app.run_sketch()
