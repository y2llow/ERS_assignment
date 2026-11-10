"""Tests for the statistics. Run: python -m pytest tests/test_statistics.py -v"""

from types import SimpleNamespace

import pytest

from src.responder_allocator import Assignment, Unassigned
from src.statistics import compute_statistics, format_statistics


def assignment(rid, iid, arrival, response):
    return Assignment(rid, iid, ["A", "B"], 0.0, arrival, response, arrival + 10, 0.0, 0.5)


INCIDENTS = [
    {"incident_id": "I1", "deadline": 50.0},
    {"incident_id": "I2", "deadline": 60.0},
    {"incident_id": "I3", "deadline": 70.0},
    {"incident_id": "I4", "deadline": 80.0},
]


def sim(assignments, unhandled):
    return SimpleNamespace(assignments=assignments, unhandled=unhandled)


def test_hand_computed_values():
    result = sim([assignment("R1", "I1", 40, 40), assignment("R1", "I2", 70, 20),
                  assignment("R2", "I3", 30, 30)],
                 [Unassigned("I4", "all_busy")])
    s = compute_statistics(INCIDENTS, result)
    assert s.incidents_total == 4
    assert s.incidents_handled == 3
    assert s.responders_used == 2                       # R1 counted once
    assert s.average_response_time == pytest.approx(30.0)   # (40+20+30)/3
    # I2 arrives at 70 > 60 -> missed; I1 and I3 met -> 2 of 4 incidents
    assert s.deadlines_met == 2
    assert s.deadlines_met_pct == pytest.approx(50.0)
    assert s.deadlines_met_pct_of_handled == pytest.approx(200 / 3)


def test_unhandled_incidents_count_as_missed_but_not_in_average():
    s = compute_statistics(INCIDENTS, sim([assignment("R1", "I1", 40, 40)],
                                          [Unassigned(i, "x") for i in ("I2", "I3", "I4")]))
    assert s.average_response_time == pytest.approx(40.0)
    assert s.deadlines_met_pct == pytest.approx(25.0)
    assert s.deadlines_met_pct_of_handled == pytest.approx(100.0)


def test_nothing_handled():
    s = compute_statistics(INCIDENTS, sim([], [Unassigned(i["incident_id"], "x") for i in INCIDENTS]))
    assert s.incidents_handled == 0 and s.responders_used == 0
    assert s.average_response_time is None
    assert s.deadlines_met_pct == 0.0
    assert "n/a" in format_statistics(s)


def test_output_format():
    s = compute_statistics(INCIDENTS, sim([assignment("R1", "I1", 40, 40)],
                                          [Unassigned("I2", "all_busy")]))
    text = format_statistics(s)
    assert text.startswith("Artificial Intelligence Practicum Emergency Response Assignment\n"
                           "System statistics :\n")
    assert "* 1 incidents handled" in text
    assert "* 1 responders used" in text
    assert "* Average response time : 40.0 s" in text
    assert "* Deadlines met : 25.0%" in text
    assert "I2 (all_busy)" in text