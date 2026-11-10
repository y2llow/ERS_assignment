"""Tests for the constraint-based allocator. Run: python -m pytest tests/test_responder_allocator.py -v"""

from types import SimpleNamespace

import pytest

from src.responder_allocator import (
    ALL_BUSY, NO_CAPABLE_RESPONDER, TOO_LATE, UNREACHABLE,
    Assignment, Unassigned, allocate,
)


class StubPlanner:
    """times[(start, goal)] = travel seconds; missing key = unreachable."""

    def __init__(self, times):
        self.times = times
        self.calls = []

    def route(self, start, goal, departure_time=0.0):
        self.calls.append((start, goal, departure_time))
        t = 0.0 if start == goal else self.times.get((start, goal))
        if t is None:
            return None
        path = [start] if start == goal else [start, goal]
        return SimpleNamespace(path=path, travel_time=t, arrival_time=departure_time + t,
                               threshold_used=0.5)


def resp(rid, loc, caps=("Medical",), free_at=0.0):
    return {"responder_id": rid, "location": loc, "capabilities": list(caps), "free_at": free_at}


def inc(iid, loc, ts=0.0, deadline=100.0, req=10.0, prio="High", cap="Medical"):
    return {"incident_id": iid, "location": loc, "priority": prio, "deadline": deadline,
            "required_capability": cap, "timestamp": ts, "time_required": req}


def pairs(result):
    return {(a.responder_id, a.incident_id) for a in result.assignments}


def reasons(result):
    return {u.incident_id: u.reason for u in result.unassigned}


# ------------------------------------------------------------ data classes
def test_assignment_plan_entry_has_exactly_the_required_keys():
    a = Assignment("R1", "I1", ["A", "B"], 5.0, 15.0, 15.0, 25.0)
    assert a.to_plan_entry() == {"responder_id": "R1", "incident_id": "I1",
                                 "path": ["A", "B"], "time": 5.0}


def test_assignment_can_be_built_with_nine_positional_arguments():
    # same call as tests/test_statistics.py
    a = Assignment("R1", "I1", ["A", "B"], 0.0, 40, 40, 50, 0.0, 0.5)
    assert a.threshold_used == 0.5 and Unassigned("I1", "x").reason == "x"


# ------------------------------------------------------------ one assignment
def test_single_assignment_values():
    result = allocate([inc("I1", "B", ts=5, req=20)], [resp("R1", "A")],
                      StubPlanner({("A", "B"): 10}), now=8.0)
    a = result.assignments[0]
    assert (a.responder_id, a.incident_id, a.path) == ("R1", "I1", ["A", "B"])
    assert a.start_time == 8.0 and a.arrival_time == 18.0
    assert a.response_time == 13.0            # arrival - incident timestamp
    assert a.busy_until == 38.0               # arrival + time_required
    assert result.unassigned == []


def test_responder_already_at_incident_has_single_node_path():
    a = allocate([inc("I1", "A")], [resp("R1", "A")], StubPlanner({}), now=0.0).assignments[0]
    assert a.path == ["A"] and a.arrival_time == 0.0


def test_departure_time_passed_to_planner_is_now():
    planner = StubPlanner({("A", "B"): 10})
    allocate([inc("I1", "B")], [resp("R1", "A")], planner, now=42.0)
    assert planner.calls == [("A", "B", 42.0)]


# --------------------------------------------------------- hard constraints
def test_capability_is_required():
    result = allocate([inc("I1", "B", cap="Rescue")],
                      [resp("R1", "A", caps=("Medical",)), resp("R2", "A", caps=("Rescue",))],
                      StubPlanner({("A", "B"): 10}), now=0.0)
    assert pairs(result) == {("R2", "I1")}


def test_no_capable_responder_anywhere():
    result = allocate([inc("I1", "B", cap="Security")], [resp("R1", "A")],
                      StubPlanner({("A", "B"): 10}), now=0.0)
    assert result.assignments == [] and reasons(result) == {"I1": NO_CAPABLE_RESPONDER}


def test_busy_responder_is_not_assigned():
    result = allocate([inc("I1", "B")], [resp("R1", "A", free_at=30.0)],
                      StubPlanner({("A", "B"): 10}), now=10.0)
    assert result.assignments == [] and reasons(result) == {"I1": ALL_BUSY}


def test_responder_that_becomes_free_exactly_now_is_available():
    result = allocate([inc("I1", "B")], [resp("R1", "A", free_at=10.0)],
                      StubPlanner({("A", "B"): 10}), now=10.0)
    assert pairs(result) == {("R1", "I1")}


def test_deadline_is_latest_arrival_inclusive():
    planner = StubPlanner({("A", "B"): 10})
    assert pairs(allocate([inc("I1", "B", deadline=10)], [resp("R1", "A")], planner, 0.0)) == {("R1", "I1")}
    late = allocate([inc("I1", "B", deadline=9.9)], [resp("R1", "A")], planner, 0.0)
    assert late.assignments == [] and reasons(late) == {"I1": TOO_LATE}


def test_unreachable_incident():
    result = allocate([inc("I1", "Z")], [resp("R1", "A")], StubPlanner({}), now=0.0)
    assert reasons(result) == {"I1": UNREACHABLE}


def test_deadline_is_checked_against_departure_now_not_timestamp():
    # incident known since 0, but the only responder can leave at t=95: arrives 105 > 100
    result = allocate([inc("I1", "B", deadline=100)], [resp("R1", "A")],
                      StubPlanner({("A", "B"): 10}), now=95.0)
    assert reasons(result) == {"I1": TOO_LATE}


def test_one_responder_serves_at_most_one_incident_per_round():
    result = allocate([inc("I1", "B"), inc("I2", "C")], [resp("R1", "A")],
                      StubPlanner({("A", "B"): 10, ("A", "C"): 10}), now=0.0)
    assert len(result.assignments) == 1 and len(result.unassigned) == 1


def test_each_incident_gets_at_most_one_responder():
    result = allocate([inc("I1", "B")], [resp("R1", "A"), resp("R2", "A")],
                      StubPlanner({("A", "B"): 10}), now=0.0)
    assert len(result.assignments) == 1


# -------------------------------------------------------------- objectives
def test_fastest_responder_is_chosen():
    result = allocate([inc("I1", "B")], [resp("R1", "A"), resp("R2", "C")],
                      StubPlanner({("A", "B"): 30, ("C", "B"): 5}), now=0.0)
    assert pairs(result) == {("R2", "I1")}


def test_higher_priority_wins_the_only_responder():
    result = allocate([inc("I1", "B", prio="Low"), inc("I2", "C", prio="Critical")],
                      [resp("R1", "A")], StubPlanner({("A", "B"): 10, ("A", "C"): 10}), now=0.0)
    assert pairs(result) == {("R1", "I2")} and reasons(result) == {"I1": ALL_BUSY}


def test_custom_priority_weights_change_the_outcome():
    incidents = [inc("I1", "B", prio="Low"), inc("I2", "C", prio="Critical")]
    planner = StubPlanner({("A", "B"): 10, ("A", "C"): 10})
    result = allocate(incidents, [resp("R1", "A")], planner, 0.0,
                      priority_weights={"Low": 100.0})
    assert pairs(result) == {("R1", "I1")}


def test_search_beats_greedy_when_the_best_responder_is_needed_elsewhere():
    # I1 (Critical) prefers R1 (5 s) over R2 (6 s). I2 (High) can ONLY be reached by R1.
    # Greedy gives R1 to I1 and leaves I2 unserved; the CSP search serves both.
    planner = StubPlanner({("A", "X"): 5, ("B", "X"): 6, ("A", "Y"): 8})
    result = allocate([inc("I1", "X", prio="Critical"), inc("I2", "Y", prio="High")],
                      [resp("R1", "A"), resp("R2", "B")], planner, now=0.0)
    assert pairs(result) == {("R2", "I1"), ("R1", "I2")}
    assert result.unassigned == [] and result.optimal


def test_node_limit_still_returns_a_valid_solution():
    # Same situation as the "search beats greedy" test, but the search is stopped right away:
    # the greedy baseline (valid, not optimal) is returned and the result says so.
    planner = StubPlanner({("A", "X"): 5, ("B", "X"): 6, ("A", "Y"): 8})
    result = allocate([inc("I1", "X", prio="Critical"), inc("I2", "Y", prio="High")],
                      [resp("R1", "A"), resp("R2", "B")], planner, now=0.0, max_nodes=1)
    assert not result.optimal
    assert pairs(result) == {("R1", "I1")}
    assert reasons(result) == {"I2": ALL_BUSY}


# ------------------------------------------------------------- purity / misc
def test_inputs_are_not_modified():
    responders = [resp("R1", "A")]
    incidents = [inc("I1", "B")]
    before = (repr(responders), repr(incidents))
    allocate(incidents, responders, StubPlanner({("A", "B"): 10}), 0.0)
    assert (repr(responders), repr(incidents)) == before


def test_empty_inputs():
    assert allocate([], [resp("R1", "A")], StubPlanner({}), 0.0).assignments == []
    result = allocate([inc("I1", "B")], [], StubPlanner({}), 0.0)
    assert reasons(result) == {"I1": NO_CAPABLE_RESPONDER}


def test_deterministic_with_equal_options():
    args = lambda: ([inc("I1", "B")], [resp("R1", "A"), resp("R2", "A")], StubPlanner({("A", "B"): 10}), 0.0)
    assert pairs(allocate(*args())) == pairs(allocate(*args())) == {("R1", "I1")}