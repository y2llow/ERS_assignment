
"""Constraint-based allocation of responders to incidents.

The allocator runs at every simulation decision moment.

Constraints:
    - Correct responder capability.
    - Responder must be available.
    - A route must exist.
    - Arrival must be before the incident deadline.
    - One responder per incident in each decision round.

Objective:
    1. Maximise the total priority weight served.
    2. Minimise priority-weighted response time.

Uses backtracking with branch-and-bound and a greedy baseline.

A deadline-aware route fallback is attempted when the original
route arrives no more than two seconds after the deadline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_PRIORITY_WEIGHTS = {
    "Critical": 10.0,
    "High": 10.0,
    "Medium": 2.0,
    "Low": 1.0,
}

DEFAULT_MAX_NODES = 200_000
_EPS = 1e-9

NO_CAPABLE_RESPONDER = "no_capable_responder"
ALL_BUSY = "all_busy"
UNREACHABLE = "unreachable"
TOO_LATE = "too_late"


@dataclass
class Assignment:
    responder_id: str
    incident_id: str
    path: list[str]

    start_time: float
    arrival_time: float
    response_time: float
    busy_until: float

    cost: float = 0.0
    threshold_used: float = 0.5

    def to_plan_entry(self) -> dict[str, Any]:
        return {
            "responder_id": self.responder_id,
            "incident_id": self.incident_id,
            "path": list(self.path),
            "time": self.start_time,
        }


@dataclass
class Unassigned:
    incident_id: str
    reason: str


@dataclass
class AllocationResult:
    assignments: list[Assignment] = field(default_factory=list)
    unassigned: list[Unassigned] = field(default_factory=list)
    optimal: bool = True
    nodes: int = 0


@dataclass
class _Option:
    responder_id: str
    route: Any
    response_time: float


def _weight(
    incident: dict,
    weights: dict[str, float],
) -> float:
    return float(weights[incident["priority"]])


def allocate(
    incidents: list[dict[str, Any]],
    responders: list[dict[str, Any]],
    planner,
    now: float,
    priority_weights: dict[str, float] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> AllocationResult:

    weights = dict(DEFAULT_PRIORITY_WEIGHTS)

    if priority_weights:
        weights.update(priority_weights)

    free = [
        r for r in responders
        if r["free_at"] <= now + _EPS
    ]

    unassigned: dict[str, str] = {}
    options: dict[str, list[_Option]] = {}

    # Generate feasible responder options for each incident.

    for incident in incidents:
        iid = incident["incident_id"]
        capability = incident["required_capability"]

        if not any(
            capability in r["capabilities"]
            for r in responders
        ):
            unassigned[iid] = NO_CAPABLE_RESPONDER
            continue

        capable_free = [
            r for r in free
            if capability in r["capabilities"]
        ]

        if not capable_free:
            unassigned[iid] = ALL_BUSY
            continue

        found: list[_Option] = []
        any_route = False

        for r in capable_free:
            route = planner.route(
                r["location"],
                incident["location"],
                now,
            )

            # Retry a slightly late route with a more
            # permissive blocking threshold.

            if (
                route is not None
                and route.arrival_time > incident["deadline"] + _EPS
                and route.arrival_time <= incident["deadline"] + 2.0
                and hasattr(planner, "route_before_deadline")
            ):
                route = planner.route_before_deadline(
                    r["location"],
                    incident["location"],
                    now,
                    incident["deadline"],
                )

            if route is None:
                continue

            any_route = True

            if route.arrival_time <= incident["deadline"] + _EPS:
                found.append(
                    _Option(
                        r["responder_id"],
                        route,
                        route.arrival_time - incident["timestamp"],
                    )
                )

        if found:
            found.sort(
                key=lambda o: (
                    o.response_time,
                    o.responder_id,
                )
            )
            options[iid] = found
        else:
            unassigned[iid] = (
                TOO_LATE if any_route else UNREACHABLE
            )

    # Order incidents by priority, number of options
    # and earliest deadline.

    by_id = {
        i["incident_id"]: i
        for i in incidents
    }

    order = sorted(
        options,
        key=lambda iid: (
            -_weight(by_id[iid], weights),
            len(options[iid]),
            by_id[iid]["deadline"],
            iid,
        ),
    )

    def cost_of(
        iid: str,
        option: _Option,
    ) -> float:
        return (
            _weight(by_id[iid], weights)
            * option.response_time
        )

    n = len(order)

    suffix_weight = [0.0] * (n + 1)
    suffix_cost = [0.0] * (n + 1)

    for k in range(n - 1, -1, -1):
        iid = order[k]

        suffix_weight[k] = (
            suffix_weight[k + 1]
            + _weight(by_id[iid], weights)
        )

        suffix_cost[k] = (
            suffix_cost[k + 1]
            + cost_of(iid, options[iid][0])
        )

    # Greedy initial solution.

    best_choice: dict[str, _Option] = {}
    used: set[str] = set()

    for iid in order:
        for option in options[iid]:
            if option.responder_id not in used:
                best_choice[iid] = option
                used.add(option.responder_id)
                break

    best_score = (
        sum(
            _weight(by_id[i], weights)
            for i in best_choice
        ),
        -sum(
            cost_of(i, o)
            for i, o in best_choice.items()
        ),
    )

    # Backtracking and branch-and-bound.

    nodes = 0
    optimal = True

    current: dict[str, _Option] = {}
    used = set()

    def search(
        k: int,
        served: float,
        cost: float,
    ) -> None:
        nonlocal best_choice, best_score, nodes, optimal

        if nodes >= max_nodes:
            optimal = False
            return

        nodes += 1

        if (
            served + suffix_weight[k],
            -(cost + suffix_cost[k]),
        ) <= best_score:
            return

        if k == n:
            best_score = (served, -cost)
            best_choice = dict(current)
            return

        iid = order[k]

        for option in options[iid]:
            if option.responder_id in used:
                continue

            used.add(option.responder_id)
            current[iid] = option

            search(
                k + 1,
                served + _weight(by_id[iid], weights),
                cost + cost_of(iid, option),
            )

            del current[iid]
            used.discard(option.responder_id)

        # Also consider leaving the incident unassigned.
        search(k + 1, served, cost)

    search(0, 0.0, 0.0)

    # Build final assignments.

    assignments: list[Assignment] = []

    for iid in order:
        option = best_choice.get(iid)

        if option is None:
            unassigned[iid] = ALL_BUSY
            continue

        incident = by_id[iid]

        assignments.append(
            Assignment(
                responder_id=option.responder_id,
                incident_id=iid,
                path=list(option.route.path),
                start_time=now,
                arrival_time=option.route.arrival_time,
                response_time=option.response_time,
                busy_until=(
                    option.route.arrival_time
                    + incident["time_required"]
                ),
                cost=cost_of(iid, option),
                threshold_used=option.route.threshold_used,
            )
        )

    return AllocationResult(
        assignments=assignments,
        unassigned=[
            Unassigned(i, r)
            for i, r in sorted(unassigned.items())
        ],
        optimal=optimal,
        nodes=nodes,
    )
