from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import random

from app.graph.honey_graph import HoneyGraph
from app.grid.layout import HexGridLayout, VertexKey


class AgentMode(Enum):
    """High-level agent mode for stepping."""
    GROW = auto()
    TRAVEL = auto()


@dataclass(slots=True)
class Agent:
    """Agent state for growth mode and frontier travel mode."""
    prev: VertexKey | None = None
    curr: VertexKey | None = None

    mode: AgentMode = AgentMode.GROW

    travel_target: VertexKey | None = None
    travel_target_version: int = -1
    travel_path: deque[VertexKey] = field(default_factory=deque)

    arrived_in_grow: bool = False
    fork_cooldown: int = 0


@dataclass(frozen=True, slots=True)
class SpawnMove:
    """Spawn instruction for a new agent including its first already-applied move.

    Attributes:
        agent: Spawned agent after the move was applied in the simulation.
        from_key: Start vertex of the spawned move (for animation).
        to_key: End vertex of the spawned move (for animation).
    """
    agent: Agent
    from_key: VertexKey
    to_key: VertexKey


class GrowthStepper:
    """Advance an agent by one decision using 'prefer new edge' and BFS-based travel."""

    def __init__(
        self,
        rng: random.Random,
        prefer_new_probability: float = 0.85,
        fork_growth_factor: float = 1.7,
        fork_jitter: int = 1,
        fork_cooldown_steps: int = 6,
        min_first_fork_at: int = 4,
    ) -> None:
        """Create a stepper.

        Args:
            rng: Random number generator.
            prefer_new_probability: Probability to choose the new edge when exactly one candidate is new.
            fork_growth_factor: Multiplier for per-vertex next_fork_at after each fork.
            fork_jitter: Random additive jitter applied to next_fork_at updates.
            fork_cooldown_steps: Per-agent cooldown after a fork to prevent rapid consecutive spawns.
            min_first_fork_at: Lower bound for the initial next_fork_at value per vertex.
        """
        self._rng = rng
        self._prefer_new_probability = max(0.0, min(1.0, prefer_new_probability))

        self._fork_growth_factor = max(1.01, fork_growth_factor)
        self._fork_jitter = max(0, fork_jitter)
        self._fork_cooldown_steps = max(0, fork_cooldown_steps)
        self._min_first_fork_at = max(2, min_first_fork_at)

    def step(
        self,
        agent: Agent,
        layout: HexGridLayout,
        graph: HoneyGraph
    ) -> tuple["SpawnMove", ...]:
        """Perform one simulation step and return any spawn moves.

        Args:
            agent: Agent state to mutate.
            layout: Layout for pixel-based left/right ordering.
            graph: Graph state for edge existence, traffic, and frontier membership.

        Returns:
            SpawnMove instructions for newly created agents, or an empty tuple if none were created.
        """
        if graph.frontier_is_empty():
            return ()

        if agent.fork_cooldown > 0:
            agent.fork_cooldown -= 1

        if agent.prev is None or agent.curr is None:
            self._initialize_agent(agent, graph)
            return ()

        if agent.mode == AgentMode.GROW:
            spawns = self._step_grow(agent, layout, graph)
            if spawns is not None:
                return spawns
            self._enter_travel_mode(agent)

        self._step_travel(agent, graph)
        return ()

    def _initialize_agent(self, agent: Agent, graph: HoneyGraph) -> None:
        """Initialize the agent on a random directed start edge.

        Args:
            agent: Agent state to mutate.
            graph: Graph state for choosing and activating the start edge.
        """
        a, b = graph.choose_random_start_edge(self._rng)
        self._apply_grow_traverse(agent, from_vertex=a, to_vertex=b, graph=graph)
        self._clear_travel_plan(agent)

    def _step_grow(
        self,
        agent: Agent,
        layout: HexGridLayout,
        graph: HoneyGraph
    ) -> tuple["SpawnMove", ...] | None:
        """Try to perform one growth step and optionally fork.

        Args:
            agent: Agent state to mutate.
            layout: Layout for pixel-based left/right ordering.
            graph: Graph state.

        Returns:
            A tuple of SpawnMove instructions if a growth move was performed, or None if blocked.
        """
        assert agent.prev is not None
        assert agent.curr is not None

        prev = agent.prev
        curr = agent.curr

        left, right = GrowthStepper._forward_options_left_right(prev, curr, layout, graph)
        candidates = [vertex for vertex in (left, right) if vertex is not None]
        if not candidates:
            return None

        if (
            len(candidates) == 2
            and left is not None
            and right is not None
            and self._should_fork_here(agent, curr, left, right, graph)
        ):
            primary = self._choose_primary_for_fork(curr, left, right, graph)
            secondary = right if primary == left else left

            self._apply_grow_traverse(agent, from_vertex=curr, to_vertex=primary, graph=graph)

            spawned_agent = Agent(mode=AgentMode.GROW)
            self._apply_grow_traverse(spawned_agent, from_vertex=curr, to_vertex=secondary, graph=graph)

            self._commit_vertex_fork(curr, graph)
            agent.fork_cooldown = self._fork_cooldown_steps
            spawned_agent.fork_cooldown = self._fork_cooldown_steps

            return (SpawnMove(agent=spawned_agent, from_key=curr, to_key=secondary),)

        next_vertex = self._choose_next(curr, candidates, graph)
        if next_vertex is None:
            return None

        self._apply_grow_traverse(agent, from_vertex=curr, to_vertex=next_vertex, graph=graph)
        return ()

    def _apply_grow_traverse(self, agent: Agent, from_vertex: VertexKey, to_vertex: VertexKey, graph: HoneyGraph) -> None:
        """Apply a traversal in growth mode and update all counters.

        Args:
            agent: Agent state to mutate.
            from_vertex: Start vertex of the traversal.
            to_vertex: End vertex of the traversal.
            graph: Graph state to update.
        """
        graph.ensure_edge_exists(from_vertex, to_vertex)
        graph.edge_state(from_vertex, to_vertex).traffic += 1

        vs = graph.vertex_state(to_vertex)
        vs.visit_count += 1
        vs.grow_visit_count += 1

        if vs.next_fork_at < self._min_first_fork_at:
            # Initialize lazily for old graphs that do not set next_fork_at at creation time.
            vs.next_fork_at = self._min_first_fork_at

        agent.prev, agent.curr = from_vertex, to_vertex
        agent.mode = AgentMode.GROW
        agent.arrived_in_grow = True

    def _step_travel(self, agent: Agent, graph: HoneyGraph) -> None:
        """Perform one travel step (plan route if needed, then traverse one edge).

        Args:
            agent: Agent state to mutate.
            graph: Graph state used for BFS planning and traversal.
        """
        assert agent.curr is not None

        if self._is_travel_target_invalid(agent, graph):
            self._clear_travel_plan(agent)

        if agent.travel_target is None or not agent.travel_path:
            planned = self._plan_travel_to_nearest_frontier(agent, graph)
            if not planned:
                return

        self._traverse_one_travel_edge(agent, graph)

        if agent.travel_target is not None and agent.curr == agent.travel_target and not agent.travel_path:
            agent.mode = AgentMode.GROW
            self._clear_travel_plan(agent)

    def _is_travel_target_invalid(self, agent: Agent, graph: HoneyGraph) -> bool:
        """Check whether the currently assigned travel target has been invalidated.

        Args:
            agent: Agent state to check.
            graph: Graph state providing the current version of the target vertex.

        Returns:
            True if the target is set and its version changed, otherwise False.
        """
        if agent.travel_target is None:
            return False
        return graph.vertex_state(agent.travel_target).version != agent.travel_target_version

    def _plan_travel_to_nearest_frontier(self, agent: Agent, graph: HoneyGraph) -> bool:
        """Plan a BFS route (on existing edges) to the nearest frontier vertex.

        Args:
            agent: Agent state to mutate.
            graph: Graph state providing frontier membership and existing-edge traversal.

        Returns:
            True if a route was found and stored, otherwise False.
        """
        assert agent.curr is not None

        target, path = self._bfs_to_nearest_frontier(agent.curr, graph)
        if target is None:
            return False

        agent.travel_target = target
        agent.travel_target_version = graph.vertex_state(target).version
        agent.travel_path = path
        return True

    def _bfs_to_nearest_frontier(self, start: VertexKey, graph: HoneyGraph) -> tuple[VertexKey | None, deque[VertexKey]]:
        """Run BFS on the existing-edge graph to find the nearest frontier vertex.

        The returned path excludes the start vertex and contains the successive vertices
        to step through to reach the target.

        Args:
            start: BFS start vertex.
            graph: Graph state.

        Returns:
            (target, path) where target is the chosen frontier vertex or None if unreachable,
            and path is a deque of next vertices to traverse.
        """
        if graph.is_frontier_vertex(start):
            return (start, deque())

        visited: set[VertexKey] = {start}
        parent: dict[VertexKey, VertexKey] = {}
        open_queue: deque[VertexKey] = deque([start])

        while open_queue:
            vertex = open_queue.popleft()
            for neighbor in graph.iter_existing_neighbors(vertex):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parent[neighbor] = vertex

                if graph.is_frontier_vertex(neighbor):
                    return (neighbor, self._reconstruct_path(parent, start, neighbor))

                open_queue.append(neighbor)

        return (None, deque())

    def _reconstruct_path(
        self,
        parent: dict[VertexKey, VertexKey],
        start: VertexKey,
        target: VertexKey,
    ) -> deque[VertexKey]:
        """Reconstruct a BFS path from start to target using the parent map.

        Args:
            parent: Mapping child -> parent produced by BFS.
            start: Start vertex.
            target: Target vertex.

        Returns:
            Deque of vertices to traverse next (excluding start, including target).
        """
        reversed_path_vertices: list[VertexKey] = []
        current_vertex = target
        while current_vertex != start:
            reversed_path_vertices.append(current_vertex)
            current_vertex = parent[current_vertex]

        reversed_path_vertices.reverse()
        return deque(reversed_path_vertices)

    def _apply_travel_traverse(self, agent: Agent, from_vertex: VertexKey, to_vertex: VertexKey, graph: HoneyGraph) -> None:
        """Apply a traversal in travel mode and update counters.

        Args:
            agent: Agent state to mutate.
            from_vertex: Start vertex of the traversal.
            to_vertex: End vertex of the traversal.
            graph: Graph state to update.
        """
        graph.edge_state(from_vertex, to_vertex).traffic += 1
        graph.vertex_state(to_vertex).visit_count += 1

        agent.prev, agent.curr = from_vertex, to_vertex
        agent.arrived_in_grow = False

    def _traverse_one_travel_edge(self, agent: Agent, graph: HoneyGraph) -> None:
        """Traverse exactly one edge along the preplanned travel path.

        Args:
            agent: Agent state to mutate.
            graph: Graph state used to update traffic and visit_count.
        """
        assert agent.curr is not None
        if not agent.travel_path:
            return

        next_vertex = agent.travel_path.popleft()
        self._apply_travel_traverse(agent, from_vertex=agent.curr, to_vertex=next_vertex, graph=graph)

    def _enter_travel_mode(self, agent: Agent) -> None:
        """Switch the agent into travel mode and reset any existing travel plan.

        Args:
            agent: Agent state to mutate.
        """
        agent.mode = AgentMode.TRAVEL
        agent.arrived_in_grow = False
        self._clear_travel_plan(agent)

    @staticmethod
    def _clear_travel_plan(agent: Agent) -> None:
        """Clear any stored travel target and path.

        Args:
            agent: Agent state to mutate.
        """
        agent.travel_target = None
        agent.travel_target_version = -1
        agent.travel_path.clear()

    def _should_fork_here(
        self,
        agent: Agent,
        vertex: VertexKey,
        left: VertexKey,
        right: VertexKey,
        graph: HoneyGraph
    ) -> bool:
        """Return whether a fork should be triggered at the given vertex.

        Rules:
          - Only on GROW arrival.
          - Requires growth potential, meaning at least one forward edge is new.
          - Uses per-vertex scheduling via next_fork_at driven by grow_visit_count.
          - Respects per-agent cooldown to avoid rapid consecutive forks by the same agent.

        Args:
            agent: Current agent.
            vertex: Vertex to evaluate.
            left: Left forward option.
            right: Right forward option.
            graph: Graph state.

        Returns:
            True if a fork should be triggered here, otherwise False.
        """
        if agent.mode != AgentMode.GROW or not agent.arrived_in_grow:
            return False
        if agent.fork_cooldown > 0:
            return False

        if not self._has_growth_potential(vertex, left, right, graph):
            return False

        vs = graph.vertex_state(vertex)
        if vs.grow_visit_count < vs.next_fork_at:
            return False

        return True

    def _has_growth_potential(self, curr: VertexKey, left: VertexKey, right: VertexKey, graph: HoneyGraph) -> bool:
        """Check whether forking can create at least one new edge.

        Args:
            curr: Current vertex.
            left: Left candidate.
            right: Right candidate.
            graph: Graph state.

        Returns:
            True if at least one of the two forward edges does not yet exist.
        """
        return (not graph.edge_state(curr, left).exists) or (not graph.edge_state(curr, right).exists)

    def _commit_vertex_fork(self, vertex: VertexKey, graph: HoneyGraph) -> None:
        """Update per-vertex scheduling state after a fork.

        Args:
            vertex: Vertex where the fork happened.
            graph: Graph state to update.
        """
        vs = graph.vertex_state(vertex)
        vs.fork_count += 1

        base_next = max(vs.next_fork_at, self._min_first_fork_at)
        grown = int(math.ceil(base_next * self._fork_growth_factor))
        jitter = self._rng.randint(0, self._fork_jitter) if self._fork_jitter > 0 else 0
        vs.next_fork_at = grown + jitter

    # @staticmethod
    # def _is_power_of_two_ge_2(value: int) -> bool:
    #     """Return True if value is a power of two and >= 2.
    #
    #     Args:
    #         value: Value to check.
    #
    #     Returns:
    #         True if value is a power of two and >= 2, otherwise False.
    #     """
    #     return value >= 2 and (value & (value - 1)) == 0

    def _choose_primary_for_fork(
        self,
        curr: VertexKey,
        left: VertexKey,
        right: VertexKey,
        graph: HoneyGraph,
    ) -> VertexKey:
        """Choose which branch the current agent takes during a fork.

        This keeps 'prefer new edge' semantics for the primary branch while the spawned
        agent takes the other branch.

        Args:
            curr: Current vertex where the fork occurs.
            left: Left forward option.
            right: Right forward option.
            graph: Graph state.

        Returns:
            The chosen primary vertex (left or right).
        """
        e_left_new = not graph.edge_state(curr, left).exists
        e_right_new = not graph.edge_state(curr, right).exists

        if e_left_new and not e_right_new:
            return left if self._rng.random() < self._prefer_new_probability else right
        if e_right_new and not e_left_new:
            return right if self._rng.random() < self._prefer_new_probability else left
        return left if self._rng.random() < 0.5 else right

    def _choose_next(self, curr: VertexKey, candidates: list[VertexKey], graph: HoneyGraph) -> VertexKey | None:
        """Choose the next vertex by preferring non-existing edges.

        Args:
            curr: Current vertex.
            candidates: Forward candidate vertices.
            graph: Graph state.

        Returns:
            The chosen next vertex, or None if no decision is possible.
        """
        if len(candidates) == 1:
            return candidates[0]

        c0, c1 = candidates[0], candidates[1]
        e0_new = not graph.edge_state(curr, c0).exists
        e1_new = not graph.edge_state(curr, c1).exists

        if e0_new and not e1_new:
            return c0 if self._rng.random() < self._prefer_new_probability else c1
        if e1_new and not e0_new:
            return c1 if self._rng.random() < self._prefer_new_probability else c0
        if e0_new and e1_new:
            return c0 if self._rng.random() < 0.5 else c1

        return None

    @staticmethod
    def _forward_options_left_right(
        prev: VertexKey,
        curr: VertexKey,
        layout: HexGridLayout,
        graph: HoneyGraph,
    ) -> tuple[VertexKey | None, VertexKey | None]:
        """Compute forward options and order them as (left, right).

        Args:
            prev: Previous vertex.
            curr: Current vertex.
            layout: Layout for pixel coordinates.
            graph: Graph topology.

        Returns:
            (left, right) ordered options, where each may be None.
        """
        candidates = [ns for ns in graph.neighbors(curr) if ns != prev]
        if len(candidates) == 0:
            return (None, None)
        if len(candidates) == 1:
            return (candidates[0], None)

        vertex_prev = layout.vertices_by_key[prev].px
        vertex_curr = layout.vertices_by_key[curr].px
        ver_cand_0 = layout.vertices_by_key[candidates[0]].px
        ver_cand_1 = layout.vertices_by_key[candidates[1]].px

        # direction vector (y-up)
        fwd_dir_x = vertex_curr[0] - vertex_prev[0]
        fwd_dir_y = -(vertex_curr[1] - vertex_prev[1])

        # candidate vectors (y-up)
        c0_x = ver_cand_0[0] - vertex_curr[0]
        c0_y = -(ver_cand_0[1] - vertex_curr[1])

        c1_x = ver_cand_1[0] - vertex_curr[0]
        c1_y = -(ver_cand_1[1] - vertex_curr[1])

        cross0 = GrowthStepper._cross_z(fwd_dir_x, fwd_dir_y, c0_x, c0_y)
        cross1 = GrowthStepper._cross_z(fwd_dir_x, fwd_dir_y, c1_x, c1_y)

        if cross0 >= cross1:
            return (candidates[0], candidates[1])
        return (candidates[1], candidates[0])

    @staticmethod
    def _cross_z(ax: float, ay: float, bx: float, by: float) -> float:
        """Compute the z-component of the 2D cross product (a x b).

        Args:
            ax: X component of vector a.
            ay: Y component of vector a.
            bx: X component of vector b.
            by: Y component of vector b.

        Returns:
            The scalar z-component of the cross product.
        """
        return ax * by - ay * bx
