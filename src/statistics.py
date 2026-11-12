"""Calculates system performance metrics such as response time and deadline success rate.

Definitions (all times in seconds of simulation time):
    incidents handled     : incidents that received a responder (the responder arrives
                            at the incident location; the deadline is a hard constraint
                            of the allocation, so an assigned incident is also on time
                            according to the planner's expected travel time).
    responders used       : distinct responders with at least one assignment.
    average response time : mean of (arrival time - incident timestamp) over the
                            handled incidents. Unhandled incidents have no arrival,
                            so they are NOT part of this average.
    deadlines met         : handled incidents with arrival <= deadline, as a percentage
                            of ALL incidents. Unhandled incidents count as missed.
                            The percentage among handled incidents is also reported.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Statistics:
    incidents_total: int
    incidents_handled: int
    responders_used: int
    average_response_time: float | None      # None if nothing was handled
    deadlines_met: int
    deadlines_met_pct: float | None          # of all incidents; None if no incidents
    deadlines_met_pct_of_handled: float | None
    unhandled: list[tuple[str, str]]         # (incident id, reason)


def compute_statistics(incidents: list[dict], simulation_result) -> Statistics:
    deadlines = {i["incident_id"]: i["deadline"] for i in incidents}
    assignments = simulation_result.assignments

    handled = len(assignments)
    met = sum(1 for a in assignments if a.arrival_time <= deadlines[a.incident_id] + 1e-9)
    average = sum(a.response_time for a in assignments) / handled if handled else None

    return Statistics(
        incidents_total=len(incidents),
        incidents_handled=handled,
        responders_used=len({a.responder_id for a in assignments}),
        average_response_time=average,
        deadlines_met=met,
        deadlines_met_pct=100.0 * met / len(incidents) if incidents else None,
        deadlines_met_pct_of_handled=100.0 * met / handled if handled else None,
        unhandled=[(u.incident_id, u.reason) for u in simulation_result.unhandled],
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def format_statistics(stats: Statistics) -> str:
    """Text for statistics.txt (format of the assignment, plus explanation)."""
    average = ("n/a" if stats.average_response_time is None
               else f"{stats.average_response_time:.1f} s")
    unhandled = (", ".join(f"{i} ({r})" for i, r in sorted(stats.unhandled))
                 if stats.unhandled else "none")
    lines = [
        "System statistics :",
        f"* {stats.incidents_handled} incidents handled",
        f"* {stats.responders_used} responders used",
        f"* Average response time : {average}",
        f"* Deadlines met : {_pct(stats.deadlines_met_pct)}",
        "",
        "Details",
        f"* Incidents in scenario : {stats.incidents_total}",
        f"* Deadlines met among handled incidents : {_pct(stats.deadlines_met_pct_of_handled)}",
        f"* Unhandled incidents : {unhandled}",
        "",
    ]
    return "\n".join(lines) + "\n"