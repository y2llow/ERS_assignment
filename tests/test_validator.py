"""Tests for plan validation. Run: python -m pytest tests/test_validator.py -v"""

from types import SimpleNamespace

import pytest

from src.route_planner import RoutePlanner
from src.validator import validate_plan


class Roads:
    def __init__(self, blocked=None):
        self.blocked = blocked or {}

    def get(self, a, b):
        p = self.blocked.get((a, b), 0.0)
        return {"Open": 1 - p, "Congested": 0.0, "Blocked": p}


class Traffic:
    def expected_multiplier(self, a, b, horizon=0.0):
        return 1.0


def make(blocked=None):
    nodes = [{"id": n, "x": x, "y": 0} for n, x in (("A", 0), ("B", 100), ("C", 200))]
    roads = [{"from": u, "to": v, "distance": 100, "speed": 10}
             for u, v in (("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"))]
    graph = {"nodes": nodes, "roads": roads}
    data = SimpleNamespace(
        city_graph=graph,
        responders=[
            {"responder_id": "R1", "start_location": "A", "capabilities": ["Medical"],
             "time_until_available": 0.0},
            {"responder_id": "R2", "start_location": "C", "capabilities": ["Rescue"],
             "time_until_available": 30.0},
        ],
        incidents=[
            {"incident_id": "I1", "location": "B", "timestamp": 0.0, "deadline": 15.0,
             "required_capability": "Medical", "time_required": 20.0},
            {"incident_id": "I2", "location": "C", "timestamp": 5.0, "deadline": 100.0,
             "required_capability": "Medical", "time_required": 5.0},
            {"incident_id": "I3", "location": "B", "timestamp": 0.0, "deadline": 100.0,
             "required_capability": "Rescue", "time_required": 5.0},
        ])
    roads_model = Roads(blocked)
    return data, RoutePlanner(graph, roads_model, Traffic()), roads_model


def check(plan, blocked=None):
    data, planner, roads = make(blocked)
    return validate_plan(plan, data, planner, roads)


def entry(r, i, path, t):
    return {"responder_id": r, "incident_id": i, "path": path, "time": t}


GOOD = [entry("R1", "I1", ["A", "B"], 0),         # arrives 10, busy until 30, at B
        entry("R1", "I2", ["B", "C"], 30)]         # arrives 40


def has(report, text):
    return any(text in e for e in report.errors)


def test_valid_plan_passes():
    report = check(GOOD)
    assert report.ok and report.errors == [] and report.entries_checked == 2


def test_empty_plan_is_valid():
    assert check([]).ok


def test_unknown_responder_and_incident():
    assert has(check([entry("R9", "I1", ["A", "B"], 0)]), "unknown responder")
    assert has(check([entry("R1", "I9", ["A", "B"], 0)]), "unknown incident")


def test_missing_fields_and_bad_time():
    assert has(check([{"responder_id": "R1"}]), "must contain")
    assert has(check([entry("R1", "I1", ["A", "B"], -1)]), "non-negative")
    assert has(check([entry("R1", "I1", [], 0)]), "non-empty")


def test_capability_mismatch():
    assert has(check([entry("R1", "I3", ["A", "B"], 0)]), "lacks capability")


def test_incident_assigned_twice():
    plan = [entry("R1", "I1", ["A", "B"], 0), entry("R2", "I1", ["C", "B"], 30)]
    assert has(check(plan), "more than once")


def test_start_before_incident_is_known():
    assert has(check([entry("R1", "I2", ["A", "B", "C"], 0)]), "before the incident is known")


def test_missing_road():
    assert has(check([entry("R1", "I2", ["A", "C"], 5)]), "no road")


def test_unknown_node():
    assert has(check([entry("R1", "I1", ["A", "Z", "B"], 0)]), "unknown node")


def test_path_must_start_at_responder_location_and_end_at_incident():
    assert has(check([entry("R1", "I1", ["B"], 0)]), "path starts at")
    assert has(check([entry("R1", "I1", ["A"], 0)]), "path ends at")


def test_responder_location_follows_previous_assignment():
    plan = [entry("R1", "I1", ["A", "B"], 0), entry("R1", "I2", ["A", "B", "C"], 30)]
    assert has(check(plan), "path starts at 'A' but R1 is at 'B'")


def test_overlapping_assignments_for_one_responder():
    plan = [entry("R1", "I1", ["A", "B"], 0), entry("R1", "I2", ["B", "C"], 20)]
    assert has(check(plan), "busy until 30")


def test_initially_busy_responder():
    assert has(check([entry("R2", "I3", ["C", "B"], 10)]), "busy until 30")
    assert check([entry("R2", "I3", ["C", "B"], 30)]).ok


def test_deadline_missed():
    report = check([entry("R1", "I1", ["A", "B"], 10)])   # arrives 20 > 15
    assert has(report, "after the deadline")


def test_blocked_road_is_an_error_and_probably_blocked_a_warning():
    assert has(check(GOOD, {("A", "B"): 0.95}), "almost certainly blocked")
    report = check(GOOD, {("A", "B"): 0.7})
    assert report.ok and any("probably blocked" in w for w in report.warnings)


def test_repeated_node_is_only_a_warning():
    report = check([entry("R1", "I2", ["A", "B", "A", "B", "C"], 5)])
    assert any("more than once" in w for w in report.warnings)


def test_plan_must_be_a_list():
    data, planner, roads = make()
    assert not validate_plan({"not": "a list"}, data, planner, roads).ok