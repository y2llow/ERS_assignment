"""Manages simulation time, incoming incidents, responder availability and assignments.

Discrete-event simulation. The system only knows the past: an incident becomes
known at its timestamp and is never looked at before.

Decision moments (events):
    - an incident appears (its timestamp);
    - a responder becomes free again (free_at).

At every decision moment:
    1. reveal the incidents whose timestamp has been reached;
    2. drop pending incidents whose deadline has passed (-> unhandled);
    3. call allocate() for all pending incidents (constraint-based allocation,
       routes from the A* planner, departure time = current time);
    4. update the responders that received an assignment:
           location = incident location, free_at = arrival + time_required;
    5. jump to the next decision moment.

The simulation stops when no event is left that could change the situation.
Pending incidents that never got a responder are reported as unhandled, with the
last reason given by the allocator.

Assumptions: see responder_allocator.py (no return trip, deadline = latest arrival,
only responders that are free at the decision moment are assigned).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.responder_allocator import Assignment, Unassigned, allocate

DEADLINE_PASSED = "deadline_passed"
_EPS = 1e-9


@dataclass
class SimulationResult:
    assignments: list[Assignment]       # in chronological order
    unhandled: list[Unassigned]         # incident id + reason
    rounds: int                         # number of allocation rounds
    log: list[str] = field(default_factory=list)


def run_simulation(data, planner, priority_weights=None) -> SimulationResult:
    """Simulate the whole incident stream of one scenario.

    `data` needs `.responders` and `.incidents` (see data_loader.ScenarioData).
    """
    responders = [
        dict(r, location=r["start_location"], free_at=float(r["time_until_available"]))
        for r in data.responders
    ]
    incidents = sorted(data.incidents, key=lambda i: (i["timestamp"], i["incident_id"]))
    by_id = {i["incident_id"]: i for i in incidents}

    pending: dict[str, dict[str, Any]] = {}
    last_reason: dict[str, str] = {}
    assignments: list[Assignment] = []
    unhandled: list[Unassigned] = []
    log: list[str] = []
    rounds = 0
    next_index = 0
    now = 0.0

    while True:
        # 1. reveal incidents that are known now
        while next_index < len(incidents) and incidents[next_index]["timestamp"] <= now + _EPS:
            incident = incidents[next_index]
            pending[incident["incident_id"]] = incident
            log.append(f"t={now:g}: {incident['incident_id']} appears at "
                       f"{incident['location']} ({incident['priority']}, needs "
                       f"{incident['required_capability']}, deadline {incident['deadline']:g})")
            next_index += 1

        # 2. incidents that can no longer be reached in time
        for incident_id in [k for k, v in pending.items() if v["deadline"] < now - _EPS]:
            reason = last_reason.get(incident_id, DEADLINE_PASSED)
            unhandled.append(Unassigned(incident_id, reason))
            log.append(f"t={now:g}: {incident_id} unhandled, deadline "
                       f"{pending[incident_id]['deadline']:g} passed ({reason})")
            del pending[incident_id]

        # 3 + 4. allocate and update responder state
        if pending:
            result = allocate(list(pending.values()), responders, planner, now, priority_weights)
            rounds += 1
            for a in result.assignments:
                responder = next(r for r in responders if r["responder_id"] == a.responder_id)
                responder["location"] = by_id[a.incident_id]["location"]
                responder["free_at"] = a.busy_until
                assignments.append(a)
                del pending[a.incident_id]
                log.append(f"t={now:g}: {a.responder_id} -> {a.incident_id} via "
                           f"{'-'.join(a.path)}, arrives {a.arrival_time:.1f} "
                           f"(response {a.response_time:.1f} s), free again at {a.busy_until:.1f}")
            for u in result.unassigned:
                last_reason[u.incident_id] = u.reason

        # 5. next decision moment
        upcoming: list[float] = []
        if next_index < len(incidents):
            upcoming.append(incidents[next_index]["timestamp"])
        if pending:
            upcoming += [r["free_at"] for r in responders if r["free_at"] > now + _EPS]
        if not upcoming:
            break
        now = min(upcoming)

    for incident_id in pending:
        reason = last_reason.get(incident_id, DEADLINE_PASSED)
        unhandled.append(Unassigned(incident_id, reason))
        log.append(f"end: {incident_id} stays unhandled ({reason})")

    return SimulationResult(assignments, unhandled, rounds, log)