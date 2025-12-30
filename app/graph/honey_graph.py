from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Set, Tuple
import random

from app.grid.layout import EdgeKey, HexGridLayout, VertexKey


@dataclass(slots=True)
class VertexState:
    """Mutable per-vertex state used by the simulation."""
    visit_count: int = 0
    version: int = 0


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

        Returns:
            None.
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

        self._adjacency_ro: Dict[VertexKey, Tuple[VertexKey, ...]] = {
            v: tuple(ns) for v, ns in self._adjacency.items()
        }

    def neighbors(self, vertex_key: VertexKey) -> Tuple[VertexKey, ...]:
        """Return the neighbors of a vertex.

        Args:
            vertex_key: Vertex key for neighbor search.

        Returns:
            Tuple of neighboring vertex keys.
        """
        return self._adjacency_ro[vertex_key]

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

        Args:
            key_a: First endpoint key.
            key_b: Second endpoint key.

        Returns:
            None.
        """
        key: EdgeKey = self._edge_key(key_a, key_b)
        state: EdgeState = self._edge_state[key]
        if state.exists:
            return
        state.exists = True
        self._active_edges.add(key)

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
