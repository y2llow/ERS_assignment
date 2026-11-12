
"""A* route planning with Bayesian road status and HMM traffic.

Edge travel time:
    base_time * max(status_factor, traffic_factor)

This assumes that the two congestion estimates may overlap.
This is a modelling assumption, not an empirically verified result.

Roads above the current blocking threshold are excluded.
A fallback can search progressively riskier routes when needed.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Any


CONGESTED_FACTOR = 1.5
DEFAULT_BLOCK_THRESHOLD = 0.5
DEFAULT_FALLBACK_THRESHOLD = 0.9
FALLBACK_STEP = 0.1
HEURISTICS = ("euclidean", "zero")


class RoutePlannerError(ValueError):
    """Raised for invalid route planner input."""


@dataclass(frozen=True)
class RouteResult:
    path: list[str]
    travel_time: float
    arrival_time: float
    nodes_expanded: int
    threshold_used: float


@dataclass(frozen=True)
class _Edge:
    start: str
    end: str
    base_time: float
    p_blocked: float
    status_factor: float | None


class RoutePlanner:
    def __init__(
        self,
        city_graph: dict[str, Any],
        road_model,
        traffic_model,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
        heuristic: str = "euclidean",
    ):
        if heuristic not in HEURISTICS:
            raise RoutePlannerError(
                f"heuristic must be one of {HEURISTICS}."
            )

        for name, value in (
            ("block_threshold", block_threshold),
            ("fallback_threshold", fallback_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise RoutePlannerError(
                    f"{name} must be between 0 and 1."
                )

        self.block_threshold = block_threshold
        self.fallback_threshold = fallback_threshold
        self.heuristic_name = heuristic
        self._traffic = traffic_model

        self._coords: dict[str, tuple[float, float]] = {}
        self._adj: dict[str, list[_Edge]] = {}

        for node in city_graph["nodes"]:
            node_id = str(node["id"])
            self._adj[node_id] = []

            if "x" in node and "y" in node:
                self._coords[node_id] = (
                    float(node["x"]),
                    float(node["y"]),
                )

        self._edges: dict[tuple[str, str], _Edge] = {}
        max_speed = 0.0
        ratios: list[float] = []

        for road in city_graph["roads"]:
            u = str(road["from"])
            v = str(road["to"])

            distance = float(road["distance"])
            speed = float(road["speed"])

            max_speed = max(max_speed, speed)

            dist = road_model.get(u, v)

            usable = dist["Open"] + dist["Congested"]

            factor = (
                (
                    dist["Open"]
                    + CONGESTED_FACTOR * dist["Congested"]
                ) / usable
                if usable > 0
                else None
            )

            edge = _Edge(
                u,
                v,
                distance / speed,
                dist["Blocked"],
                factor,
            )

            self._adj[u].append(edge)
            self._edges[(u, v)] = edge

            if u in self._coords and v in self._coords:
                straight = math.dist(
                    self._coords[u],
                    self._coords[v],
                )

                if straight > 0:
                    ratios.append(distance / straight)

        self._max_speed = max_speed if max_speed > 0 else 1.0

        self.heuristic_scale = (
            min(1.0, min(ratios))
            if ratios
            else 1.0
        )

    def heuristic(self, node: str, goal: str) -> float:
        """Admissible lower bound on travel time."""

        if self.heuristic_name == "zero":
            return 0.0

        a = self._coords.get(node)
        b = self._coords.get(goal)

        if a is None or b is None:
            return 0.0

        return (
            self.heuristic_scale
            * math.dist(a, b)
            / self._max_speed
        )

    def route(
        self,
        start: str,
        goal: str,
        departure_time: float = 0.0,
    ) -> RouteResult | None:
        """Find a route with incremental blocking thresholds."""

        for node in (start, goal):
            if node not in self._adj:
                raise RoutePlannerError(
                    f"Unknown node {node!r}."
                )

        if departure_time < 0:
            raise RoutePlannerError(
                "departure_time must not be negative."
            )

        if start == goal:
            return RouteResult(
                [start],
                0.0,
                departure_time,
                0,
                self.block_threshold,
            )

        thresholds = [self.block_threshold]

        if self.fallback_threshold > self.block_threshold:
            current = self.block_threshold

            while True:
                next_threshold = round(
                    current + FALLBACK_STEP, 10
                )

                if next_threshold >= self.fallback_threshold:
                    thresholds.append(self.fallback_threshold)
                    break

                thresholds.append(next_threshold)
                current = next_threshold

        expanded_total = 0

        for threshold in thresholds:
            found, expanded = self._search(
                start,
                goal,
                departure_time,
                threshold,
            )

            expanded_total += expanded

            if found is not None:
                path, travel_time = found

                return RouteResult(
                    path,
                    travel_time,
                    departure_time + travel_time,
                    expanded_total,
                    threshold,
                )

        return None

    def route_before_deadline(
        self,
        start,
        goal,
        departure_time,
        deadline,
    ):
        """Search for a timely alternative if the normal route is late."""

        original = self.route(
            start,
            goal,
            departure_time,
        )

        if (
            original is None
            or original.arrival_time <= deadline + 1e-9
        ):
            return original

        current = self.block_threshold

        while current < self.fallback_threshold - 1e-10:
            current = min(
                self.fallback_threshold,
                round(current + FALLBACK_STEP, 10),
            )

            found, expanded = self._search(
                start,
                goal,
                departure_time,
                current,
            )

            if found is None:
                continue

            path, travel_time = found
            arrival_time = departure_time + travel_time

            if arrival_time <= deadline + 1e-9:
                return RouteResult(
                    path,
                    travel_time,
                    arrival_time,
                    expanded,
                    current,
                )

        return original

    def path_travel_time(
        self,
        path: list[str],
        departure_time: float = 0.0,
    ) -> float:
        """Calculate travel time for an existing path."""

        elapsed = 0.0

        for u, v in zip(path, path[1:]):
            edge = self._edges.get((u, v))

            if edge is None:
                raise RoutePlannerError(
                    f"No road {u!r} -> {v!r}."
                )

            if edge.status_factor is None:
                raise RoutePlannerError(
                    f"Road {u!r} -> {v!r} is certainly blocked."
                )

            elapsed += self._edge_time(
                edge,
                departure_time + elapsed,
            )

        return elapsed

    def _edge_time(
        self,
        edge: _Edge,
        horizon: float,
    ) -> float:
        """Calculate the expected travel time of one edge."""

        traffic = self._traffic.expected_multiplier(
            edge.start,
            edge.end,
            horizon,
        )

        # Alternative assumption:
        # the two congestion estimates may overlap.
        return edge.base_time * max(
            edge.status_factor,
            traffic,
        )

    def _search(
        self,
        start,
        goal,
        departure_time,
        threshold,
    ):
        """A* search for a given blocking threshold."""

        counter = itertools.count()

        best_g = {start: 0.0}
        parent: dict[str, str] = {}

        heap = [
            (
                self.heuristic(start, goal),
                next(counter),
                start,
            )
        ]

        closed: set[str] = set()
        expanded = 0

        while heap:
            _, _, node = heapq.heappop(heap)

            if node in closed:
                continue

            closed.add(node)
            expanded += 1

            if node == goal:
                path = [node]

                while path[-1] in parent:
                    path.append(parent[path[-1]])

                path.reverse()

                return (
                    path,
                    best_g[goal],
                ), expanded

            for edge in self._adj[node]:

                if (
                    edge.status_factor is None
                    or edge.p_blocked > threshold
                ):
                    continue

                if edge.end in closed:
                    continue

                new_g = (
                    best_g[node]
                    + self._edge_time(
                        edge,
                        departure_time + best_g[node],
                    )
                )

                if new_g < best_g.get(
                    edge.end,
                    math.inf,
                ):
                    best_g[edge.end] = new_g
                    parent[edge.end] = node

                    heapq.heappush(
                        heap,
                        (
                            new_g
                            + self.heuristic(edge.end, goal),
                            next(counter),
                            edge.end,
                        ),
                    )

        return None, expanded
