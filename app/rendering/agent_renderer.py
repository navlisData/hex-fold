from __future__ import annotations

from typing import Iterable, Protocol

from app.graph import AgentMode
from app.grid.layout import HexGridLayout
from app.simulation.agent_controller import AgentDrawable


class _AgentSketch(Protocol):
    def no_stroke(self) -> None: ...
    def fill(self, r: int, g: int, b: int, a: int | None = None) -> None: ...
    def circle(self, x: float, y: float, d: float) -> None: ...
    def no_fill(self) -> None: ...

_AGENT_MODE_COLORS: dict[AgentMode, tuple[int, int, int]] = {
    AgentMode.GROW: (220, 40, 40),
    AgentMode.TRAVEL: (40, 120, 220),
}

def draw_agents(
    sketch: _AgentSketch,
    layout: HexGridLayout,
    agents: Iterable[AgentDrawable],
) -> None:
    """Draw all agents with a mode-specific color.

    Args:
        sketch: The py5 sketch or compatible protocol.
        layout: Layout used to scale marker size.
        agents: Iterable of render-ready agent positions.
    """
    diameter = max(4.0, layout.radius_px * 0.15)

    sketch.no_stroke()

    for agent in agents:
        r, g, b = _AGENT_MODE_COLORS[agent.mode]
        sketch.fill(r, g, b)
        sketch.circle(agent.px[0], agent.px[1], diameter)

    sketch.no_fill()