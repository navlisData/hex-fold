from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Mapping

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
    growth_steps_since_fork: int = 0


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
            fork_cooldown_steps: int = 2,
            min_growth_steps_before_fork: int = 5,
            fork_agent_exclusion_radius: int = 10,
            max_nearby_agents_for_fork: int | None = 0,
    ) -> None:
        """Create a stepper.

        Args:
            rng: Random number generator.
            prefer_new_probability: Probability to choose the new edge when exactly one candidate is new.
            fork_cooldown_steps: Per-agent cooldown after a fork to prevent rapid consecutive spawns.
            min_growth_steps_before_fork: Minimum number of growth-mode traversals an agent must perform before it may fork again.
            fork_agent_exclusion_radius: Static graph-distance radius used to detect nearby agents before forking.
            max_nearby_agents_for_fork: Maximum number of other agents allowed inside the exclusion radius.
                If None, the density check is disabled.
        """
        self._rng = rng
        self._prefer_new_probability = max(0.0, min(1.0, prefer_new_probability))

        self._fork_cooldown_steps = max(0, fork_cooldown_steps)
        self._min_growth_steps_before_fork = max(1, min_growth_steps_before_fork)

        self._fork_agent_exclusion_radius = max(0, fork_agent_exclusion_radius)
        self._max_nearby_agents_for_fork = (
            None
            if max_nearby_agents_for_fork is None
            else max(0, max_nearby_agents_for_fork)
        )

    def step(
        self,
        agent: Agent,
        layout: HexGridLayout,
        graph: HoneyGraph,
        agent_counts_by_vertex: Mapping[VertexKey, int] | None = None,
    ) -> tuple["SpawnMove", ...]:
        """Perform one simulation step and return any spawn moves.

        Args:
            agent: Agent state to mutate.
            layout: Layout for pixel-based left/right ordering.
            graph: Graph state for edge existence, traffic, and frontier membership.
            agent_counts_by_vertex: Snapshot of current agent counts by vertex.

        Returns:
            SpawnMove instructions for newly created agents, or an empty tuple if none were created.
        """
        if graph.frontier_is_empty():
            return ()

        fork_density_snapshot = agent_counts_by_vertex if agent_counts_by_vertex is not None else {}

        if agent.fork_cooldown > 0:
            agent.fork_cooldown -= 1

        if agent.prev is None or agent.curr is None:
            self._initialize_agent(agent, graph)
            return ()

        if agent.mode == AgentMode.GROW:
            spawns = self._step_grow(agent, layout, graph, fork_density_snapshot)
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
        graph: HoneyGraph,
        agent_counts_by_vertex: Mapping[VertexKey, int],
    ) -> tuple["SpawnMove", ...] | None:
        """Try to perform one growth step and optionally fork.

        Args:
            agent: Agent state to mutate.
            layout: Layout for pixel-based left/right ordering.
            graph: Graph state.
            agent_counts_by_vertex: Snapshot of current agent counts by vertex.

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
                and self._should_fork_here(agent, curr, left, right, graph, agent_counts_by_vertex)
        ):
            primary = self._choose_primary_for_fork(curr, left, right, graph)
            secondary = right if primary == left else left

            self._apply_grow_traverse(agent, from_vertex=curr, to_vertex=primary, graph=graph)

            spawned_agent = Agent(mode=AgentMode.GROW)
            self._apply_grow_traverse(spawned_agent, from_vertex=curr, to_vertex=secondary, graph=graph)

            self._commit_vertex_fork(curr, graph)
            self._reset_growth_age_after_fork(agent)
            self._reset_growth_age_after_fork(spawned_agent)

            agent.fork_cooldown = self._fork_cooldown_steps
            spawned_agent.fork_cooldown = self._fork_cooldown_steps

            return (SpawnMove(agent=spawned_agent, from_key=curr, to_key=secondary),)

        next_vertex = self._choose_next(curr, candidates, graph)
        if next_vertex is None:
            return None

        self._apply_grow_traverse(agent, from_vertex=curr, to_vertex=next_vertex, graph=graph)
        return ()

    @staticmethod
    def _reset_growth_age_after_fork(agent: Agent) -> None:
        agent.growth_steps_since_fork = 0

    def _apply_grow_traverse(self, agent: Agent, from_vertex: VertexKey, to_vertex: VertexKey,
                             graph: HoneyGraph) -> None:
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

        agent.growth_steps_since_fork += 1
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
            graph: HoneyGraph,
            agent_counts_by_vertex: Mapping[VertexKey, int],
    ) -> bool:
        """Return whether a fork should be triggered at the given vertex.

        Rules:
          - Only on GROW arrival.
          - Requires two new forward branches.
          - Requires enough growth-mode traversals by the agent since its last fork.
          - Respects per-agent cooldown to avoid rapid consecutive forks by the same agent.
          - Requires the static graph neighborhood to be sparse enough.

        Args:
            agent: Current agent.
            vertex: Vertex to evaluate.
            left: Left forward option.
            right: Right forward option.
            graph: Graph state.
            agent_counts_by_vertex: Snapshot of current agent counts by vertex.

        Returns:
            True if a fork should be triggered here, otherwise False.
        """
        if agent.mode != AgentMode.GROW or not agent.arrived_in_grow:
            return False
        if agent.fork_cooldown > 0:
            return False
        if agent.growth_steps_since_fork < self._min_growth_steps_before_fork:
            return False
        if not self._has_two_new_forward_edges(vertex, left, right, graph):
            return False

        return self._is_fork_neighborhood_sparse_enough(vertex, graph, agent_counts_by_vertex)

    def _is_fork_neighborhood_sparse_enough(
        self,
        center: VertexKey,
        graph: HoneyGraph,
        agent_counts_by_vertex: Mapping[VertexKey, int],
    ) -> bool:
        """Check whether the static neighborhood around a vertex is sparse enough for forking.

        The current agent at the center vertex is excluded from the count. Traversal uses
        the static honeycomb topology, not only already existing simulation edges.

        Args:
            center: Center vertex of the density check.
            graph: Graph state providing the static topology.
            agent_counts_by_vertex: Snapshot of current agent counts by vertex.

        Returns:
            True if the configured density threshold allows a fork, otherwise False.
        """
        if self._max_nearby_agents_for_fork is None:
            return True

        nearby_agent_count = 0
        visited: set[VertexKey] = {center}
        open_queue: deque[tuple[VertexKey, int]] = deque([(center, 0)])

        while open_queue:
            vertex, distance = open_queue.popleft()

            vertex_agent_count = agent_counts_by_vertex.get(vertex, 0)
            if vertex == center and vertex_agent_count > 0:
                vertex_agent_count -= 1

            nearby_agent_count += vertex_agent_count
            if nearby_agent_count > self._max_nearby_agents_for_fork:
                return False

            if distance >= self._fork_agent_exclusion_radius:
                continue

            for neighbor in graph.neighbors(vertex):
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                open_queue.append((neighbor, distance + 1))

        return True

    def _has_two_new_forward_edges(self, curr: VertexKey, left: VertexKey, right: VertexKey, graph: HoneyGraph) -> bool:
        return (
                not graph.edge_state(curr, left).exists
                and not graph.edge_state(curr, right).exists
        )

    def _commit_vertex_fork(self, vertex: VertexKey, graph: HoneyGraph) -> None:
        graph.vertex_state(vertex).fork_count += 1

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
