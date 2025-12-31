from __future__ import annotations

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


class GrowthStepper:
    """Advance an agent by one decision using 'prefer new edge' and BFS-based travel."""

    def __init__(self, rng: random.Random, prefer_new_probability: float = 0.85) -> None:
        """Create a stepper.

        Args:
            rng: Random number generator.
            prefer_new_probability: Probability to choose the new edge when exactly one candidate is new.
        """
        self._rng = rng
        self._prefer_new_probability = max(0.0, min(1.0, prefer_new_probability))

    def step(self, agent: Agent, layout: HexGridLayout, graph: HoneyGraph) -> None:
        """Perform one simulation step.

        Growth mode:
          - Computes forward options and chooses next by preferring a new edge.
          - If locally blocked, switches to travel mode.

        Travel mode:
          - Plans a BFS route to the nearest frontier vertex (on existing edges).
          - Traverses one edge per step, still increasing traffic and visit_count.
          - Soft-cancels the route if the target's version changes.

        Args:
            agent: Agent state to mutate.
            layout: Layout for pixel-based left/right ordering.
            graph: Graph state for edge existence, traffic and frontier membership.
        """
        if graph.frontier_is_empty():
            return

        if agent.prev is None or agent.curr is None:
            self._initialize_agent(agent, graph)
            return

        if agent.mode == AgentMode.GROW:
            progressed = self._step_grow(agent, layout, graph)
            if progressed:
                return
            self._enter_travel_mode(agent)

        self._step_travel(agent, graph)

    def _initialize_agent(self, agent: Agent, graph: HoneyGraph) -> None:
        """Initialize the agent on a random directed start edge.

        Args:
            agent: Agent state to mutate.
            graph: Graph state for choosing and activating the start edge.
        """
        a, b = graph.choose_random_start_edge(self._rng)
        graph.ensure_edge_exists(a, b)
        graph.edge_state(a, b).traffic += 1
        graph.vertex_state(b).visit_count += 1

        agent.prev, agent.curr = a, b
        agent.mode = AgentMode.GROW
        self._clear_travel_plan(agent)

    def _step_grow(self, agent: Agent, layout: HexGridLayout, graph: HoneyGraph) -> bool:
        """Try to perform one growth step.

        Args:
            agent: Agent state to mutate.
            layout: Layout for pixel-based left/right ordering.
            graph: Graph state.

        Returns:
            True if a growth move was performed, otherwise False.
        """
        assert agent.prev is not None
        assert agent.curr is not None

        prev = agent.prev
        curr = agent.curr

        left, right = GrowthStepper._forward_options_left_right(prev, curr, layout, graph)
        candidates = [vertex for vertex in (left, right) if vertex is not None]
        if not candidates:
            return False

        next_vertex = self._choose_next(curr, candidates, graph)
        if next_vertex is None:
            return False

        graph.ensure_edge_exists(curr, next_vertex)
        graph.edge_state(curr, next_vertex).traffic += 1
        graph.vertex_state(next_vertex).visit_count += 1

        agent.prev, agent.curr = curr, next_vertex
        return True

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

        graph.edge_state(agent.curr, next_vertex).traffic += 1
        graph.vertex_state(next_vertex).visit_count += 1

        agent.prev, agent.curr = agent.curr, next_vertex

    def _enter_travel_mode(self, agent: Agent) -> None:
        """Switch the agent into travel mode and reset any existing travel plan.

        Args:
            agent: Agent state to mutate.
        """
        agent.mode = AgentMode.TRAVEL
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
