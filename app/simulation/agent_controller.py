from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import cast, Mapping

from py5 import lerp

from app.config import SimulationTimingConfig
from app.graph.growth_stepper import Agent, AgentMode, GrowthStepper, SpawnMove
from app.graph.honey_graph import HoneyGraph
from app.grid.layout import HexGridLayout, Point, VertexKey


class _AnimPhase(Enum):
    """Internal animation phases for a single agent."""
    IDLE = auto()
    MOVING = auto()
    DWELLING = auto()
    STOPPED = auto()

class AgentVisualState(Enum):
    GROWING = auto()
    FORK_READY = auto()
    TRAVELING = auto()

@dataclass(frozen=True, slots=True)
class AgentDrawable:
    """Render-ready agent state."""
    px: Point
    visual_state: AgentVisualState
    growth_progress: float


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

    def prime_move(
        self,
        now_ms: int,
        from_key: VertexKey,
        to_key: VertexKey,
        layout: HexGridLayout,
        mode: AgentMode,
    ) -> None:
        """Prime the animator with an already-applied simulation move.

        Args:
            now_ms: Current sketch time in ms.
            from_key: Move start vertex.
            to_key: Move end vertex.
            layout: Layout for pixel positions.
            mode: Mode used for dwell timing after the move finishes.
        """
        self._last_mode = mode
        self._move = _Move(
            from_key=from_key,
            to_key=to_key,
            start_ms=now_ms,
            end_ms=now_ms + self._timing.edge_traverse_ms,
        )
        self._phase = _AnimPhase.MOVING
        self._pos_px = layout.vertices_by_key[from_key].px

    def update(
            self,
            now_ms: int,
            agent: Agent,
            layout: HexGridLayout,
            graph: HoneyGraph,
            stepper: GrowthStepper,
            agent_counts_by_vertex: Mapping[VertexKey, int],
    ) -> tuple[SpawnMove, ...]:
        """Advance animation and trigger discrete simulation steps when needed.

        Args:
            now_ms: Current sketch time in ms.
            agent: The simulation agent to mutate via the stepper.
            layout: Layout for vertex->pixel mapping.
            graph: Graph state.
            stepper: Discrete stepper that performs one logical move per call.
            agent_counts_by_vertex: Snapshot of current agent counts by vertex.

        Returns:
            SpawnMove instructions created by the discrete step (empty if none).
        """
        if self._phase == _AnimPhase.STOPPED:
            return ()

        if self._phase == _AnimPhase.IDLE:
            return self._start_next_move(now_ms, agent, layout, graph, stepper, agent_counts_by_vertex)

        if self._phase == _AnimPhase.MOVING:
            self._update_moving(now_ms, layout)
            return ()

        if self._phase == _AnimPhase.DWELLING:
            if now_ms < self._dwell_until_ms:
                return ()
            return self._start_next_move(now_ms, agent, layout, graph, stepper, agent_counts_by_vertex)

        return ()

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
            cast(float, lerp(from_px[0], to_px[0], t_clamped)),
            cast(float, lerp(from_px[1], to_px[1], t_clamped)),
        )

        if t_clamped < 1.0:
            return

        self._pos_px = to_px
        self._move = None
        self._phase = _AnimPhase.DWELLING
        self._dwell_until_ms = now_ms + self._dwell_ms_for_mode(self._last_mode)

    def _start_next_move(
            self,
            now_ms: int,
            agent: Agent,
            layout: HexGridLayout,
            graph: HoneyGraph,
            stepper: GrowthStepper,
            agent_counts_by_vertex: Mapping[VertexKey, int],
    ) -> tuple[SpawnMove, ...]:
        """Trigger exactly one discrete step and start animating it if it moved.

        Args:
            now_ms: Current sketch time in ms.
            agent: Agent to step.
            layout: Layout for pixel mapping.
            graph: Graph state.
            stepper: Discrete stepper.
            agent_counts_by_vertex: Snapshot of current agent counts by vertex.

        Returns:
            SpawnMove instructions created by this discrete step (empty if none).
        """
        if graph.frontier_is_empty():
            self._phase = _AnimPhase.STOPPED
            return ()

        old_prev = agent.prev
        old_curr = agent.curr

        spawns = stepper.step(agent, layout, graph, agent_counts_by_vertex)
        assert agent.prev is not None and agent.curr is not None

        self._last_mode = agent.mode

        moved = (old_curr != agent.curr) or (old_prev != agent.prev)
        if not moved:
            self._pos_px = layout.vertices_by_key[agent.curr].px
            self._phase = _AnimPhase.DWELLING
            self._dwell_until_ms = now_ms + 1
            return spawns

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
        return spawns

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
        pending_spawns: list[SpawnMove] = []
        agent_counts_by_vertex = self._snapshot_agent_counts_by_vertex()

        for runtime in self._runtimes:
            pending_spawns.extend(
                runtime.animator.update(
                    now_ms,
                    runtime.agent,
                    layout,
                    graph,
                    self._stepper,
                    agent_counts_by_vertex,
                )
            )

        for spawn in pending_spawns:
            self._add_spawned_agent(spawn, now_ms, layout)

    def _snapshot_agent_counts_by_vertex(self) -> dict[VertexKey, int]:
        """Build a stable snapshot of current agent counts by simulation vertex.

        Agents without an initialized current vertex are ignored. The snapshot is created
        once per controller update to avoid order-dependent density checks.

        Returns:
            A dictionary mapping vertex keys to the number of agents currently located there.
        """
        counts_by_vertex: dict[VertexKey, int] = {}

        for runtime in self._runtimes:
            current_vertex = runtime.agent.curr
            if current_vertex is None:
                continue

            counts_by_vertex[current_vertex] = counts_by_vertex.get(current_vertex, 0) + 1

        return counts_by_vertex

    def _add_spawned_agent(self, spawn: SpawnMove, now_ms: int, layout: HexGridLayout) -> None:
        """Register a spawned agent and prime its animator for the already-applied move.

        Args:
            spawn: Spawn move instruction from the stepper.
            now_ms: Current sketch time in ms.
            layout: Layout for pixel mapping.
        """
        animator = _AgentAnimator(self._timing)
        animator.prime_move(
            now_ms=now_ms,
            from_key=spawn.from_key,
            to_key=spawn.to_key,
            layout=layout,
            mode=spawn.agent.mode,
        )
        self._runtimes.append(AgentRuntime(agent=spawn.agent, animator=animator))

    def get_drawables(self, steps_before_fork: int) -> tuple[AgentDrawable, ...]:
        """Return render-ready agent states.

        Args:
            steps_before_fork: Step count that maps an agent to full visual growth progress.

        Returns:
            Tuple of AgentDrawable objects.
        """
        return tuple(
            self._to_drawable(
                runtime=runtime,
                steps_before_fork=steps_before_fork,
            )
            for runtime in self._runtimes
        )

    @staticmethod
    def _to_drawable(
            runtime: AgentRuntime,
            steps_before_fork: int,
    ) -> AgentDrawable:
        """Convert an agent runtime into a render-ready drawable.

        Args:
            runtime: Runtime container holding simulation and animation state.
            steps_before_fork: Step count that maps an agent to full visual growth progress.

        Returns:
            Render-ready agent drawable.
        """
        growth_progress = _normalize_growth_progress(
            growth_steps_since_fork=runtime.agent.growth_steps_since_fork,
            steps_before_fork=steps_before_fork,
        )

        return AgentDrawable(
            px=runtime.animator.position_px(),
            visual_state=_resolve_agent_visual_state(
                mode=runtime.agent.mode,
                growth_progress=growth_progress,
            ),
            growth_progress=growth_progress,
        )

def _resolve_agent_visual_state(
    mode: AgentMode,
    growth_progress: float,
) -> AgentVisualState:
    """Resolve the visual state for an agent.

    Args:
        mode: Current simulation mode of the agent.
        growth_progress: Normalized growth progress between 0.0 and 1.0.

    Returns:
        Visual state used by the renderer.
    """
    if mode is AgentMode.TRAVEL:
        return AgentVisualState.TRAVELING

    if growth_progress >= 1.0:
        return AgentVisualState.FORK_READY

    return AgentVisualState.GROWING

def _normalize_growth_progress(
    growth_steps_since_fork: int,
    steps_before_fork: int,
) -> float:
    """Normalize growth progress to the range from 0.0 to 1.0.

    Args:
        growth_steps_since_fork: Number of growth steps since the agent last forked.
        steps_before_fork: Step count that represents full progress.

    Returns:
        Clamped growth progress between 0.0 and 1.0.
    """
    safe_steps_before_fork = max(1, steps_before_fork)
    clamped_steps = max(0, min(growth_steps_since_fork, safe_steps_before_fork))

    return clamped_steps / safe_steps_before_fork