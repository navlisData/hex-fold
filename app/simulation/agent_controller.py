from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from py5 import lerp

from app.graph.growth_stepper import Agent, AgentMode, GrowthStepper
from app.graph.honey_graph import HoneyGraph
from app.grid.layout import HexGridLayout, Point, VertexKey
from app.simulation.timing_config import SimulationTimingConfig


class _AnimPhase(Enum):
    """Internal animation phases for a single agent."""
    IDLE = auto()
    MOVING = auto()
    DWELLING = auto()
    STOPPED = auto()


@dataclass(frozen=True, slots=True)
class AgentDrawable:
    """Render-ready agent state."""
    px: Point
    mode: AgentMode


@dataclass(slots=True)
class _Move:
    """A single animated move from one vertex to another."""
    from_key: VertexKey
    to_key: VertexKey
    start_ms: int
    end_ms: int


class _AgentAnimator:
    """Time-based animator that advances the discrete simulation step-by-step."""

    def __init__(self, timing: SimulationTimingConfig) -> None:
        """Create an animator.

        Args:
            timing: Timing configuration for movement and dwelling.
        """
        self._timing = timing
        self._phase: _AnimPhase = _AnimPhase.IDLE
        self._pos_px: Point = (0.0, 0.0)
        self._move: _Move | None = None
        self._dwell_until_ms: int = 0
        self._last_mode: AgentMode = AgentMode.GROW

    def position_px(self) -> Point:
        """Return the current interpolated screen position.

        Returns:
            Current position in pixels.
        """
        return self._pos_px

    def update(self, now_ms: int, agent: Agent, layout: HexGridLayout, graph: HoneyGraph, stepper: GrowthStepper) -> None:
        """Advance animation and trigger discrete simulation steps when needed.

        Args:
            now_ms: Current sketch time in ms.
            agent: The simulation agent to mutate via the stepper.
            layout: Layout for vertex->pixel mapping.
            graph: Graph state.
            stepper: Discrete stepper that performs one logical move per call.
        """
        if self._phase == _AnimPhase.STOPPED:
            return

        if self._phase == _AnimPhase.IDLE:
            self._start_next_move(now_ms, agent, layout, graph, stepper)
            return

        if self._phase == _AnimPhase.MOVING:
            self._update_moving(now_ms, layout)
            return

        if self._phase == _AnimPhase.DWELLING:
            if now_ms < self._dwell_until_ms:
                return
            self._start_next_move(now_ms, agent, layout, graph, stepper)

    def _update_moving(self, now_ms: int, layout: HexGridLayout) -> None:
        """Update interpolation during a move and transition to dwelling at the end.

        Args:
            now_ms: Current time in ms.
            layout: Layout for pixel positions.
        """
        assert self._move is not None

        from_px = layout.vertices_by_key[self._move.from_key].px
        to_px = layout.vertices_by_key[self._move.to_key].px

        duration_ms = max(1, self._move.end_ms - self._move.start_ms)
        t = (now_ms - self._move.start_ms) / float(duration_ms)
        t_clamped = min(1.0, max(0.0, t))

        self._pos_px = (
            lerp(from_px[0], to_px[0], t_clamped),
            lerp(from_px[1], to_px[1], t_clamped),
        )

        if t_clamped < 1.0:
            return

        self._pos_px = to_px
        self._move = None
        self._phase = _AnimPhase.DWELLING
        self._dwell_until_ms = now_ms + self._dwell_ms_for_mode(self._last_mode)

    def _start_next_move(self, now_ms: int, agent: Agent, layout: HexGridLayout, graph: HoneyGraph, stepper: GrowthStepper) -> None:
        """Trigger exactly one discrete step and start animating it if it moved.

        Args:
            now_ms: Current time in ms.
            agent: Agent to step.
            layout: Layout for pixel mapping.
            graph: Graph state.
            stepper: Discrete stepper.
        """
        if graph.frontier_is_empty():
            self._phase = _AnimPhase.STOPPED
            return

        old_prev = agent.prev
        old_curr = agent.curr

        stepper.step(agent, layout, graph)
        assert agent.prev is not None and agent.curr is not None

        self._last_mode = agent.mode

        moved = (old_curr != agent.curr) or (old_prev != agent.prev)
        if not moved:
            self._pos_px = layout.vertices_by_key[agent.curr].px
            self._phase = _AnimPhase.DWELLING
            self._dwell_until_ms = now_ms + 1
            return

        from_key = agent.prev
        to_key = agent.curr

        self._move = _Move(
            from_key=from_key,
            to_key=to_key,
            start_ms=now_ms,
            end_ms=now_ms + self._timing.edge_traverse_ms,
        )
        self._phase = _AnimPhase.MOVING
        self._pos_px = layout.vertices_by_key[from_key].px

    def _dwell_ms_for_mode(self, mode: AgentMode) -> int:
        """Return dwell duration in ms for a given agent mode.

        Args:
            mode: Agent mode.

        Returns:
            Dwell duration in ms.
        """
        if mode == AgentMode.TRAVEL:
            return self._timing.travel_vertex_dwell_ms
        return 0


@dataclass(slots=True)
class AgentRuntime:
    """Runtime container binding simulation and presentation state.

    Attributes:
        agent: Pure simulation state.
        animator: Time-based presentation state.
    """
    agent: Agent
    animator: _AgentAnimator


class AgentController:
    """Orchestrates stepping and animation for one or more agents."""

    def __init__(self, stepper: GrowthStepper, timing: SimulationTimingConfig) -> None:
        """Create an agent controller.

        Args:
            stepper: Discrete simulation stepper.
            timing: Timing configuration for animation.
        """
        self._stepper = stepper
        self._timing = timing
        self._runtimes: list[AgentRuntime] = []

    def add_agent(self, agent: Agent) -> None:
        """Register a new agent for animation and stepping.

        Args:
            agent: Agent to register.
        """
        self._runtimes.append(AgentRuntime(agent=agent, animator=_AgentAnimator(self._timing)))

    def update(self, now_ms: int, layout: HexGridLayout, graph: HoneyGraph) -> None:
        """Update all agents (time-based) and advance discrete steps when phases finish.

        Args:
            now_ms: Current sketch time in ms.
            layout: Layout for pixel mapping.
            graph: Graph state.
        """
        for runtime in self._runtimes:
            runtime.animator.update(now_ms, runtime.agent, layout, graph, self._stepper)

    def get_drawables(self) -> tuple[AgentDrawable, ...]:
        """Return render-ready agent states.

        Returns:
            Tuple of AgentDrawable objects.
        """
        return tuple(
            AgentDrawable(px=runtime.animator.position_px(), mode=runtime.agent.mode)
            for runtime in self._runtimes
        )
