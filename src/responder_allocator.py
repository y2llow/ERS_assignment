"""Constraint-based allocation of responders to incidents.

Called by the simulation at every decision moment (see simulation.py):

    allocate(pending_incidents, responders, planner, now, priority_weights)

It only decides ONE round: which of the responders that are free at `now` go to which of
the incidents that are known at `now`. It never changes the responder dicts; the
simulation updates location / free_at from the returned Assignment objects.

Constraint satisfaction problem (CSP)
-------------------------------------
Variables : one per pending incident.
Domain    : the free responders that satisfy the HARD unary constraints, or "unassigned":
              - capability   : responder has the incident's required capability
              - availability : responder is free at `now` (free_at <= now)
              - connectivity : the A* planner finds a route from the responder's location
              - deadline     : arrival time (now + travel time) <= incident deadline
Binary constraint (all-different): a responder serves at most one incident per round, and
              an incident gets at most one responder.

Objective (SOFT constraints, compared lexicographically)
    1. maximise the summed priority weight of the incidents that get a responder
       (Critical > High > Medium > Low, weights are configurable);
    2. minimise the priority-weighted response time of those incidents.

Search: depth-first backtracking with branch and bound.
    - variable ordering: highest priority first, then fewest options (MRV), then earliest deadline;
    - value ordering: fastest responder first;
    - bound: the best conceivable completion (all remaining incidents served at their
      fastest option); a branch that cannot beat the best solution so far is cut;
    - the search starts from a greedy solution, so there is always a valid answer even
      when `max_nodes` is reached (then `AllocationResult.optimal` is False).

Assumptions (document in the report):
    - Only responders that are free at `now` are assigned (no pre-booking of a busy one).
    - The responder departs at `now` from its current location; there is no return trip.
    - Response time = arrival time - incident timestamp (so waiting time counts).
    - busy_until = arrival time + time_required of the incident.
    - The deadline is the latest ARRIVAL time (arrival <= deadline is allowed).
    - Being allocated now and waiting for a better responder later is not considered
      (the simulation is online: it does not know future incidents).
    - If a route is at most 2 seconds late, a deadline-aware route fallback may
      try roads with higher blockage probability, up to the planner's limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_PRIORITY_WEIGHTS = {"Critical": 10.0, "High": 5.0, "Medium": 2.0, "Low": 1.0}
DEFAULT_MAX_NODES = 200_000
_EPS = 1e-9

# Reasons why an incident gets no responder in a round (used in statistics.txt).
NO_CAPABLE_RESPONDER = "no_capable_responder"   # nobody in the scenario has the capability
ALL_BUSY = "all_busy"                           # capable responders are busy / used elsewhere
UNREACHABLE = "unreachable"                     # no usable route to the incident
TOO_LATE = "too_late"                           # reachable, but only after the deadline


@dataclass
class Assignment:
    """One responder -> incident assignment (one entry of plan.json)."""

    responder_id: str
    incident_id: str
    path: list[str]
    start_time: float        # simulation time at which the responder departs
    arrival_time: float      # start_time + expected travel time
    response_time: float     # arrival_time - incident timestamp
    busy_until: float        # arrival_time + time_required
    cost: float = 0.0        # priority-weighted response time (objective contribution)
    threshold_used: float = 0.5  # P(Blocked) threshold with which the route was found

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
    optimal: bool = True      # False if the node limit stopped the search early
    nodes: int = 0            # search nodes visited


@dataclass
class _Option:
    responder_id: str
    route: Any                # RouteResult
    response_time: float


def _weight(incident: dict, weights: dict[str, float]) -> float:
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

    free = [r for r in responders if r["free_at"] <= now + _EPS]
    unassigned: dict[str, str] = {}
    options: dict[str, list[_Option]] = {}

    # ---- hard unary constraints -> domains
    for incident in incidents:
        iid = incident["incident_id"]
        capability = incident["required_capability"]

        if not any(capability in r["capabilities"] for r in responders):
            unassigned[iid] = NO_CAPABLE_RESPONDER
            continue
        capable_free = [r for r in free if capability in r["capabilities"]]
        if not capable_free:
            unassigned[iid] = ALL_BUSY
            continue

        found: list[_Option] = []
        any_route = False
        for r in capable_free:
            route = planner.route(r["location"], incident["location"], now)
            if (route is not None and route.arrival_time > incident["deadline"] + _EPS
                    and route.arrival_time <= incident["deadline"] + 2.0
                    and hasattr(planner, "route_before_deadline")):
                route = planner.route_before_deadline(
                    r["location"], incident["location"], now, incident["deadline"])
            if route is None:
                continue
            any_route = True
            if route.arrival_time <= incident["deadline"] + _EPS:
                found.append(_Option(r["responder_id"], route,
                                     route.arrival_time - incident["timestamp"]))
        if found:
            found.sort(key=lambda o: (o.response_time, o.responder_id))
            options[iid] = found
        else:
            unassigned[iid] = TOO_LATE if any_route else UNREACHABLE

    # ---- search: variable ordering
    by_id = {i["incident_id"]: i for i in incidents}
    order = sorted(
        options,
        key=lambda iid: (-_weight(by_id[iid], weights), len(options[iid]),
                         by_id[iid]["deadline"], iid),
    )

    def cost_of(iid: str, option: _Option) -> float:
        return _weight(by_id[iid], weights) * option.response_time

    # lower bound on the cost of serving incident k..end (each at its fastest option)
    n = len(order)
    suffix_weight = [0.0] * (n + 1)
    suffix_cost = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        iid = order[k]
        suffix_weight[k] = suffix_weight[k + 1] + _weight(by_id[iid], weights)
        suffix_cost[k] = suffix_cost[k + 1] + cost_of(iid, options[iid][0])

    # ---- greedy baseline (also the fallback if the node limit is reached)
    best_choice: dict[str, _Option] = {}
    used: set[str] = set()
    for iid in order:
        for option in options[iid]:
            if option.responder_id not in used:
                best_choice[iid] = option
                used.add(option.responder_id)
                break
    best_score = (
        sum(_weight(by_id[i], weights) for i in best_choice),
        -sum(cost_of(i, o) for i, o in best_choice.items()),
    )

    # ---- backtracking with branch and bound
    nodes = 0
    optimal = True
    current: dict[str, _Option] = {}
    used = set()

    def search(k: int, served: float, cost: float) -> None:
        nonlocal best_choice, best_score, nodes, optimal
        if nodes >= max_nodes:
            optimal = False
            return
        nodes += 1

        if (served + suffix_weight[k], -(cost + suffix_cost[k])) <= best_score:
            return                                   # cannot beat the best solution
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
            search(k + 1, served + _weight(by_id[iid], weights),
                   cost + cost_of(iid, option))
            del current[iid]
            used.discard(option.responder_id)
        search(k + 1, served, cost)                  # leave this incident unassigned

    search(0, 0.0, 0.0)

    # ---- build the result
    assignments: list[Assignment] = []
    for iid in order:
        option = best_choice.get(iid)
        if option is None:
            unassigned[iid] = ALL_BUSY               # capable responders went to others
            continue
        incident = by_id[iid]
        assignments.append(Assignment(
            responder_id=option.responder_id,
            incident_id=iid,
            path=list(option.route.path),
            start_time=now,
            arrival_time=option.route.arrival_time,
            response_time=option.response_time,
            busy_until=option.route.arrival_time + incident["time_required"],
            cost=cost_of(iid, option),
            threshold_used=option.route.threshold_used,
        ))

    return AllocationResult(
        assignments=assignments,
        unassigned=[Unassigned(i, r) for i, r in sorted(unassigned.items())],
        optimal=optimal,
        nodes=nodes,
    )