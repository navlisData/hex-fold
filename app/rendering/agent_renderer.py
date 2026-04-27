from __future__ import annotations

from typing import Iterable, Protocol

from app.graph import AgentMode
from app.grid.layout import HexGridLayout
from app.simulation.agent_controller import AgentDrawable, AgentVisualState


class _AgentSketch(Protocol):
    def no_stroke(self) -> None: ...
    def fill(self, r: int, g: int, b: int, a: int | None = None) -> None: ...
    def circle(self, x: float, y: float, d: float) -> None: ...
    def no_fill(self) -> None: ...

_AGENT_VISUAL_STATE_COLORS: dict[AgentVisualState, tuple[int, int, int]] = {
    AgentVisualState.GROWING: (220, 40, 40),
    AgentVisualState.FORK_READY: (255, 170, 40),
    AgentVisualState.TRAVELING: (40, 120, 220),
}

def draw_agents(
    sketch: _AgentSketch,
    layout: HexGridLayout,
    agents: Iterable[AgentDrawable],
) -> None:
    """Draw all agents with visual-state-specific color and growth-dependent size.

    Args:
        sketch: The py5 sketch or compatible protocol.
        layout: Layout used to scale marker size.
        agents: Iterable of render-ready agent positions.
    """
    min_diameter = max(4.0, layout.radius_px * 0.15)
    max_diameter = max(min_diameter, layout.radius_px * 0.3)

    sketch.no_stroke()

    for agent in agents:
        r, g, b = _AGENT_VISUAL_STATE_COLORS[agent.visual_state]
        diameter = min_diameter + (max_diameter - min_diameter) * agent.growth_progress

        sketch.fill(r, g, b)
        sketch.circle(agent.px[0], agent.px[1], diameter)

    sketch.no_fill()