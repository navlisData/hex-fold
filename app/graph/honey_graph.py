from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Set

from app.grid.layout import EdgeKey, HexGridLayout, VertexKey


@dataclass(slots=True)
class VertexState:
    """Mutable per-vertex state used by the simulation."""
    visit_count: int = 0
    version: int = 0
    open_incident_edges: int = 0


@dataclass(slots=True)
class EdgeState:
    """Mutable per-edge state used by the simulation."""
    exists: bool = False
    traffic: int = 0


class HoneyGraph:
    """Graph wrapper that separates static topology from mutable simulation state."""

    def __init__(self, layout: HexGridLayout) -> None:
        """Create a graph from a computed layout.

        Args:
            layout: Layout containing vertices, edges, and vertex->pixel mapping.
        """
        self._vertices_by_key: Mapping[VertexKey, object] = layout.vertices_by_key
        self._adjacency: Dict[VertexKey, list[VertexKey]] = {k: [] for k in layout.vertices_by_key.keys()}
        self._edge_state: Dict[EdgeKey, EdgeState] = {}
        self._vertex_state: Dict[VertexKey, VertexState] = {k: VertexState() for k in layout.vertices_by_key.keys()}

        self._active_edges: Set[EdgeKey] = set()

        for edge in layout.edges:
            a = edge.start_key
            b = edge.end_key
            self._adjacency[a].append(b)
            self._adjacency[b].append(a)
            self._edge_state[self._edge_key(a, b)] = EdgeState(exists=False, traffic=0)

        self._adjacency_ro: Dict[VertexKey, tuple[VertexKey, ...]] = {
            v: tuple(ns) for v, ns in self._adjacency.items()
        }

        self._frontier_count = 0
        for vertex, ns in self._adjacency_ro.items():
            state: VertexState = self._vertex_state[vertex]
            state.open_incident_edges = len(ns)
            if state.open_incident_edges > 0:
                self._frontier_count += 1

    def neighbors(self, vertex_key: VertexKey) -> tuple[VertexKey, ...]:
        """Return the neighbors of a vertex.

        Args:
            vertex_key: Vertex key for neighbor search.

        Returns:
            Tuple of neighboring vertex keys.
        """
        return self._adjacency_ro[vertex_key]

    def iter_existing_neighbors(self, vertex_key: VertexKey) -> Iterable[VertexKey]:
        """Iterate over neighbors connected via existing edges only.

        This is the traversal graph for BFS in travel mode.

        Args:
            vertex_key: Vertex key to iterate from.

        Yields:
            Neighbor vertex keys reachable via edges with exists=True.
        """
        for nb in self._adjacency_ro[vertex_key]:
            if self._edge_state[self._edge_key(vertex_key, nb)].exists:
                yield nb

    def is_frontier_vertex(self, vertex_key: VertexKey) -> bool:
        """Return whether a vertex is currently part of the frontier.

        Args:
            vertex_key: Vertex key to query.

        Returns:
            True if the vertex has at least one missing incident edge, otherwise False.
        """
        return self._vertex_state[vertex_key].open_incident_edges > 0

    def frontier_is_empty(self) -> bool:
        """Return whether the frontier is empty.

        Returns:
            True if there is no frontier vertex left, otherwise False.
        """
        return self._frontier_count == 0

    def vertex_state(self, vertex_key: VertexKey) -> VertexState:
        """Return mutable vertex state.

        Args:
            vertex_key: Vertex key to get state from.

        Returns:
            The VertexState instance for the vertex.
        """
        return self._vertex_state[vertex_key]

    def edge_state(self, key_a: VertexKey, key_b: VertexKey) -> EdgeState:
        """Return mutable edge state for the undirected edge (a, b).

        Args:
            key_a: First endpoint key.
            key_b: Second endpoint key.

        Returns:
            The EdgeState instance for the edge.
        """
        return self._edge_state[self._edge_key(key_a, key_b)]

    def iter_active_edges(self) -> Iterable[EdgeKey]:
        """Iterate over currently active (exists=True) edges.

        Returns:
            Iterable of edge keys.
        """
        return self._active_edges

    def ensure_edge_exists(self, key_a: VertexKey, key_b: VertexKey) -> None:
        """Mark an edge as existing and track it as active.

        Also updates frontier counters and increments vertex versions when a vertex
        stops being part of the frontier.

        Args:
            key_a: First endpoint key.
            key_b: Second endpoint key.
        """
        key = self._edge_key(key_a, key_b)
        state = self._edge_state[key]
        if state.exists:
            return

        state.exists = True
        self._active_edges.add(key)

        self._decrement_open_incident_edges(key_a)
        self._decrement_open_incident_edges(key_b)

    def choose_random_start_edge(self, rng: random.Random) -> tuple[VertexKey, VertexKey]:
        """Pick a random directed start edge (A -> B).

        Args:
            rng: Random number generator.

        Returns:
            A tuple (key_a, key_b) representing key_a directed edge A -> B.
        """
        vertices = list(self._adjacency_ro.keys())
        while True:
            key_b = rng.choice(vertices)
            neighbors = self._adjacency_ro[key_b]
            if neighbors:
                key_a = rng.choice(neighbors)
                return (key_a, key_b)

    def _decrement_open_incident_edges(self, vertex_key: VertexKey) -> None:
        """Decrease the missing-edge counter for a vertex and close frontier if it hits zero.

        Args:
            vertex_key: Vertex key to update.
        """
        state = self._vertex_state[vertex_key]
        if state.open_incident_edges <= 0:
            return

        state.open_incident_edges -= 1

        if state.open_incident_edges == 0:
            state.version += 1
            self._frontier_count -= 1

    @staticmethod
    def _edge_key(key_a: VertexKey, key_b: VertexKey) -> EdgeKey:
        """Create a canonical undirected edge key.

        Args:
            key_a: First vertex key.
            key_b: Second vertex key.

        Returns:
            Canonical edge key (sorted endpoints).
        """
        return (key_a, key_b) if key_a <= key_b else (key_b, key_a)
