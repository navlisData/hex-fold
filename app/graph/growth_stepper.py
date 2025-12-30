from __future__ import annotations

from dataclasses import dataclass
import random

from app.graph.honey_graph import HoneyGraph
from app.grid.layout import HexGridLayout, VertexKey


@dataclass(slots=True)
class Agent:
    """Minimal agent state for growth-mode stepping."""
    prev: VertexKey | None = None
    curr: VertexKey | None = None


class GrowthStepper:
    """Advance an agent by one growth decision using 'prefer new edge'."""

    def __init__(self, rng: random.Random, prefer_new_probability: float = 0.85) -> None:
        """Create a stepper.

        Args:
            rng: Random number generator.
            prefer_new_probability: Probability to choose the new edge when exactly one candidate is new.

        Returns:
            None.
        """
        self._rng = rng
        self._prefer_new_probability = max(0.0, min(1.0, prefer_new_probability))

    def step(self, agent: Agent, layout: HexGridLayout, graph: HoneyGraph) -> None:
        """Perform one growth step.

        This:
          - Initializes with a random directed start edge if the agent is uninitialized.
          - Computes the two forward options at the current vertex.
          - Chooses the next vertex by preferring a not-yet-existing edge.

        Args:
            agent: Agent state to mutate.
            layout: Layout for pixel-based left/right ordering.
            graph: Graph state for edge existence and traffic.

        Returns:
            None.
        """
        if agent.prev is None or agent.curr is None:
            a, b = graph.choose_random_start_edge(self._rng)
            graph.ensure_edge_exists(a, b)
            graph.edge_state(a, b).traffic += 1
            graph.vertex_state(b).visit_count += 1
            agent.prev, agent.curr = a, b
            return

        prev = agent.prev
        curr = agent.curr

        left, right = GrowthStepper._forward_options_left_right(prev, curr, layout, graph)
        candidates = [vertex for vertex in (left, right) if vertex is not None]
        if not candidates:
            return

        next_vertex = self._choose_next(curr, candidates, graph)
        if next_vertex is None:
            return

        graph.ensure_edge_exists(curr, next_vertex)
        graph.edge_state(curr, next_vertex).traffic += 1
        graph.vertex_state(next_vertex).visit_count += 1

        agent.prev, agent.curr = curr, next_vertex

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

        # Both exist => local growth blocked (travel-mode will handle later)
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

        # direction vector
        fwd_dir_x = vertex_curr[0] - vertex_prev[0]
        fwd_dir_y = -(vertex_curr[1] - vertex_prev[1])  # y-up

        # candidate vectors
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

        This value is positive if b is to the left of a (in a y-up coordinate system),
        negative if b is to the right, and zero if they are collinear.

        Args:
            ax: X component of vector a.
            ay: Y component of vector a.
            bx: X component of vector b.
            by: Y component of vector b.

        Returns:
            The scalar z-component of the cross product.
        """
        return ax * by - ay * bx
