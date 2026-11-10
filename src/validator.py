"""Validates generated assignments and routes against the assignment requirements.

The validator recomputes everything from the plan itself (plan.json entries) and the
scenario data. It does not trust the allocator's own bookkeeping, so it can catch bugs
in the simulation and allocation.

Errors (plan is invalid):
    structure      missing/invalid fields, non-numeric or negative time
    references     unknown responder or incident, incident assigned twice
    capability     responder lacks the incident's required capability
    route          unknown node, missing directed road, path does not start at the
                   responder's current location or does not end at the incident,
                   road with P(Blocked) above `max_blocked_probability`
    timing         assignment starts before the incident is known, or while the
                   responder is still busy (initial busy time or previous incident,
                   including time_required) -> overlapping assignments
    deadline       arrival (start + expected travel time) after the deadline
Warnings (plan is valid but questionable):
    repeated node in a path; a road with P(Blocked) above `warn_blocked_probability`.

Limitations:
    - Travel times come from the same planner model (Bayes + HMM expectations), so a
      shared modelling error is not detected.
    - The true road status is hidden; "valid" means valid with respect to our estimate.
    - Only a responder's own timeline is checked, not interaction between responders
      (none is required by the assignment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.route_planner import RoutePlannerError

REQUIRED_KEYS = ("responder_id", "incident_id", "path", "time")
_EPS = 1e-6


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    entries_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_plan(
    plan: list[dict[str, Any]],
    data,
    planner,
    road_model=None,
    max_blocked_probability: float = 0.9,
    warn_blocked_probability: float = 0.5,
) -> ValidationReport:
    report = ValidationReport(entries_checked=len(plan) if isinstance(plan, list) else 0)
    if not isinstance(plan, list):
        report.errors.append("Plan must be a list of assignments.")
        return report

    nodes = {str(n["id"]) for n in data.city_graph["nodes"]}
    roads = {(str(r["from"]), str(r["to"])) for r in data.city_graph["roads"]}
    responders = {r["responder_id"]: r for r in data.responders}
    incidents = {i["incident_id"]: i for i in data.incidents}

    by_responder: dict[str, list[dict[str, Any]]] = {}
    seen_incidents: set[str] = set()

    # ---- per-entry checks that do not depend on other entries
    for index, entry in enumerate(plan):
        label = f"entry {index}"
        if not isinstance(entry, dict) or any(k not in entry for k in REQUIRED_KEYS):
            report.errors.append(f"{label}: must contain {', '.join(REQUIRED_KEYS)}.")
            continue
        rid, iid, path, start = (entry[k] for k in REQUIRED_KEYS)
        label = f"entry {index} ({rid}->{iid})"

        ok = True
        if rid not in responders:
            report.errors.append(f"{label}: unknown responder."); ok = False
        if iid not in incidents:
            report.errors.append(f"{label}: unknown incident."); ok = False
        if (isinstance(start, bool) or not isinstance(start, (int, float)) or start < 0):
            report.errors.append(f"{label}: 'time' must be a non-negative number."); ok = False
        if (not isinstance(path, list) or not path
                or not all(isinstance(p, str) for p in path)):
            report.errors.append(f"{label}: 'path' must be a non-empty list of node ids.")
            ok = False
        if not ok:
            continue

        if iid in seen_incidents:
            report.errors.append(f"{label}: incident assigned more than once.")
        seen_incidents.add(iid)

        incident = incidents[iid]
        if incident["required_capability"] not in responders[rid]["capabilities"]:
            report.errors.append(
                f"{label}: responder lacks capability {incident['required_capability']!r}.")
        if start < incident["timestamp"] - _EPS:
            report.errors.append(
                f"{label}: starts at {start:g}, before the incident is known "
                f"({incident['timestamp']:g}).")
        if path[-1] != incident["location"]:
            report.errors.append(
                f"{label}: path ends at {path[-1]!r}, incident is at {incident['location']!r}.")
        if len(set(path)) != len(path):
            report.warnings.append(f"{label}: path visits a node more than once.")

        for node in path:
            if node not in nodes:
                report.errors.append(f"{label}: unknown node {node!r}.")
        for u, v in zip(path, path[1:]):
            if (u, v) not in roads:
                report.errors.append(f"{label}: no road {u!r} -> {v!r}.")
            elif road_model is not None:
                p_blocked = road_model.get(u, v)["Blocked"]
                if p_blocked > max_blocked_probability:
                    report.errors.append(
                        f"{label}: road {u}->{v} is almost certainly blocked "
                        f"(P(Blocked)={p_blocked:.2f}).")
                elif p_blocked > warn_blocked_probability:
                    report.warnings.append(
                        f"{label}: road {u}->{v} is probably blocked "
                        f"(P(Blocked)={p_blocked:.2f}).")

        by_responder.setdefault(rid, []).append(entry)

    # ---- per-responder timeline: location, availability, overlap, deadline
    for rid, entries in by_responder.items():
        responder = responders[rid]
        location = responder["start_location"]
        free_at = float(responder["time_until_available"])

        for entry in sorted(entries, key=lambda e: e["time"]):
            iid, path, start = entry["incident_id"], entry["path"], entry["time"]
            incident = incidents[iid]
            label = f"{rid}->{iid}"

            if start < free_at - _EPS:
                report.errors.append(
                    f"{label}: starts at {start:g} but {rid} is busy until {free_at:g} "
                    f"(overlapping assignment or still unavailable).")
            if path[0] != location:
                report.errors.append(
                    f"{label}: path starts at {path[0]!r} but {rid} is at {location!r}.")

            try:
                travel = planner.path_travel_time(path, start)
            except RoutePlannerError:
                travel = None  # missing road already reported above
            if travel is None:
                free_at = max(free_at, start)
            else:
                arrival = start + travel
                if arrival > incident["deadline"] + _EPS:
                    report.errors.append(
                        f"{label}: arrives at {arrival:.1f}, after the deadline "
                        f"{incident['deadline']:g}.")
                free_at = arrival + incident["time_required"]
            location = incident["location"]

    return report