"""Implements graph search to find suitable routes and calculate estimated travel times.

Edge cost (seconds), for a directed road u -> v entered at simulation time t:

    cost = distance / speed * status_factor(u, v) * traffic_factor(u, v, t)

status_factor (Bayesian network, RoadConditionModel)
    Blocked roads cannot be used. For usable roads we take the expected factor
    conditioned on "not blocked":
        (P(Open) * 1 + P(Congested) * 1.5) / (P(Open) + P(Congested))
    A road is EXCLUDED when P(Blocked) > threshold.

traffic_factor (HMM, TrafficModel)
    TrafficModel.expected_multiplier(u, v, horizon), where horizon is the
    time since simulation time 0 (departure_time + time already travelled),
    because the HMM belief is anchored at time 0.

Search: A* with f = g + h, g in seconds.
    h(n) = scale * euclidean(n, goal) / max_speed
    `scale` = min(1, min over roads of distance / euclidean), so h never
    overestimates even if a road were shorter than the straight line.
    h = 0 (heuristic="zero") gives Dijkstra.

Blocked handling:
    Search with block_threshold first.
    If unreachable, increase the threshold by 0.1 and retry.
    Continue until fallback_threshold (default 0.9).
    Stop immediately when a route is found.
    Return None if no route is found.

Deadline-aware fallback (route_before_deadline):
    If the default route misses a deadline, also check progressively higher
    blocking thresholds and return the first route that arrives in time.
    It never changes the travel-time formula or the deadline itself.

Assumptions:
    - Status and traffic are independent, so their factors multiply.
    - Both factors are >= 1, so travel time >= distance / speed.
    - Time-dependent costs may slightly violate the FIFO property.
    - Road status is treated as constant over time.
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
    """Raised for invalid input."""


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
            u, v = str(road["from"]), str(road["to"])
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

    # -------------------------------------------------- API

    def heuristic(self, node: str, goal: str) -> float:
        """Lower bound on travel time in seconds."""

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
        """Find the cheapest route using incremental fallback thresholds."""

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

        # Incremental fallback:
        # 0.5 -> 0.6 -> 0.7 -> 0.8 -> 0.9
        #
        # A* is executed again for every threshold.
        # Stop immediately when a route is found.

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


    def route_before_deadline(self, start, goal, departure_time, deadline):
        """Try progressively riskier paths ONLY when default route is too late."""
        original = self.route(start, goal, departure_time)
        if original is None or original.arrival_time <= deadline + 1e-9:
            return original
        current = self.block_threshold
        while current < self.fallback_threshold - 1e-10:
            current = min(self.fallback_threshold, round(current + FALLBACK_STEP, 10))
            found, expanded = self._search(start, goal, departure_time, current)
            if found is None:
                continue
            path, travel_time = found
            if departure_time + travel_time <= deadline + 1e-9:
                return RouteResult(path, travel_time, departure_time + travel_time,
                                   expanded, current)
        return original

    def path_travel_time(
        self,
        path: list[str],
        departure_time: float = 0.0,
    ) -> float:
        """Calculate expected travel time for a given path."""

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

    # --------------------------------------------- Internals

    def _edge_time(
        self,
        edge: _Edge,
        horizon: float,
    ) -> float:
        traffic = self._traffic.expected_multiplier(
            edge.start,
            edge.end,
            horizon,
        )

        return (
            edge.base_time
            * edge.status_factor
            * traffic
        )

    def _search(
        self,
        start,
        goal,
        departure_time,
        threshold,
    ):
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
