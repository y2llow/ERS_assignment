"""Tests for the simulation loop. Run: python -m pytest tests/test_simulation.py -v"""

from types import SimpleNamespace

from src.simulation import DEADLINE_PASSED, run_simulation


class StubPlanner:
    def __init__(self, times):
        self.times = times

    def route(self, start, goal, departure_time=0.0):
        t = 0.0 if start == goal else self.times.get((start, goal))
        if t is None:
            return None
        path = [start] if start == goal else [start, goal]
        return SimpleNamespace(path=path, travel_time=t,
                               arrival_time=departure_time + t, threshold_used=0.5)


def resp(rid, loc, caps=("Medical",), until=0.0):
    return {"responder_id": rid, "type": "Ambulance", "start_location": loc,
            "capabilities": list(caps),
            "availability": "Busy" if until else "Available",
            "time_until_available": until}


def inc(iid, loc, ts=0.0, deadline=100.0, req=10.0, prio="High", cap="Medical"):
    return {"incident_id": iid, "location": loc, "priority": prio, "deadline": deadline,
            "required_capability": cap, "timestamp": ts, "time_required": req}


def data(responders, incidents):
    return SimpleNamespace(responders=responders, incidents=incidents)


def by_incident(result):
    return {a.incident_id: a for a in result.assignments}


def test_incident_is_not_used_before_its_timestamp():
    result = run_simulation(
        data([resp("R1", "A")], [inc("I1", "B", ts=50)]),
        StubPlanner({("A", "B"): 10}))
    a = by_incident(result)["I1"]
    assert a.start_time == 50 and a.arrival_time == 60


def test_responder_is_reused_after_it_is_free_and_starts_where_it_ended():
    # I1: arrive 10, busy until 10+20=30, now at B. I2 appears at 5 and waits until 30.
    planner = StubPlanner({("A", "B"): 10, ("B", "C"): 5})
    result = run_simulation(
        data([resp("R1", "A")], [inc("I1", "B", req=20), inc("I2", "C", ts=5)]), planner)
    a1, a2 = by_incident(result)["I1"], by_incident(result)["I2"]
    assert a1.busy_until == 30
    assert a2.start_time == 30 and a2.path == ["B", "C"] and a2.arrival_time == 35


def test_no_overlap_for_one_responder():
    planner = StubPlanner({("A", "B"): 10, ("B", "C"): 5})
    result = run_simulation(
        data([resp("R1", "A")], [inc("I1", "B", req=20), inc("I2", "C", ts=5)]), planner)
    first, second = result.assignments
    assert second.start_time >= first.busy_until


def test_initially_busy_responder_is_used_once_free():
    result = run_simulation(
        data([resp("R1", "A", until=30.0)], [inc("I1", "B")]),
        StubPlanner({("A", "B"): 10}))
    a = by_incident(result)["I1"]
    assert a.start_time == 30 and a.arrival_time == 40


def test_incident_waits_for_responder_and_expires_if_deadline_passes():
    planner = StubPlanner({("A", "B"): 10, ("A", "C"): 10})
    result = run_simulation(
        data([resp("R1", "A")],
             [inc("I1", "B", req=100, prio="Critical"), inc("I2", "C", ts=5, deadline=50)]),
        planner)
    assert [a.incident_id for a in result.assignments] == ["I1"]
    assert [(u.incident_id, u.reason) for u in result.unhandled] == [("I2", "all_busy")]


def test_unreachable_and_incapable_incidents_are_reported_unhandled():
    result = run_simulation(
        data([resp("R1", "A")], [inc("I1", "Z"), inc("I2", "B", cap="Rescue")]),
        StubPlanner({("A", "B"): 10}))
    reasons = {u.incident_id: u.reason for u in result.unhandled}
    assert reasons == {"I1": "unreachable", "I2": "no_capable_responder"}


def test_every_incident_is_either_handled_or_unhandled_exactly_once():
    incidents = [inc("I1", "B"), inc("I2", "C", ts=10), inc("I3", "Z", ts=20)]
    result = run_simulation(data([resp("R1", "A")], incidents),
                            StubPlanner({("A", "B"): 5, ("B", "C"): 5}))
    seen = [a.incident_id for a in result.assignments] + [u.incident_id for u in result.unhandled]
    assert sorted(seen) == ["I1", "I2", "I3"]


def test_priority_decides_when_two_incidents_compete_at_the_same_time():
    planner = StubPlanner({("A", "B"): 10, ("A", "C"): 10})
    result = run_simulation(
        data([resp("R1", "A")],
             [inc("I1", "B", prio="Low", req=500, deadline=50),
              inc("I2", "C", prio="Critical", req=500, deadline=50)]), planner)
    assert [a.incident_id for a in result.assignments] == ["I2"]


def test_deterministic():
    d = data([resp("R1", "A"), resp("R2", "A")], [inc("I1", "B"), inc("I2", "B")])
    planner = StubPlanner({("A", "B"): 10})
    first = run_simulation(d, planner)
    second = run_simulation(d, planner)
    assert [(a.responder_id, a.incident_id) for a in first.assignments] == \
           [(a.responder_id, a.incident_id) for a in second.assignments]


def test_deadline_passed_reason_constant_exists():
    assert DEADLINE_PASSED == "deadline_passed"