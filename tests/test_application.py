"""End-to-end test: run.py pipeline on a tiny scenario and on the provided scenarios.

Run: python -m pytest tests/test_application.py -v
"""

import csv
import json
from pathlib import Path

import pytest

from src.application import run_application
from src.data_loader import load_scenario
from src.road_reasoning import RoadConditionModel
from src.route_planner import RoutePlanner
from src.traffic_model import TrafficModel
from src.validator import validate_plan

CPT = {
    "weather_prior": {"Clear": 0.5, "Rain": 0.3, "Storm": 0.2},
    "road_status_given_weather": {
        "Clear": {"Open": 0.95, "Congested": 0.04, "Blocked": 0.01},
        "Rain": {"Open": 0.60, "Congested": 0.30, "Blocked": 0.10},
        "Storm": {"Open": 0.15, "Congested": 0.35, "Blocked": 0.50}},
    "citizen_report": {
        "Open": {"Open": 0.70, "Congested": 0.20, "Blocked": 0.10},
        "Congested": {"Open": 0.20, "Congested": 0.60, "Blocked": 0.20},
        "Blocked": {"Open": 0.10, "Congested": 0.20, "Blocked": 0.70}},
    "police_report": {
        "Open": {"Open": 0.92, "Congested": 0.06, "Blocked": 0.02},
        "Congested": {"Open": 0.08, "Congested": 0.84, "Blocked": 0.08},
        "Blocked": {"Open": 0.02, "Congested": 0.08, "Blocked": 0.90}},
    "sensor_report": {
        "Open": {"Open": 0.88, "Congested": 0.10, "Blocked": 0.02},
        "Congested": {"Open": 0.10, "Congested": 0.80, "Blocked": 0.10},
        "Blocked": {"Open": 0.03, "Congested": 0.07, "Blocked": 0.90}},
}
TRAFFIC = {
    "initial": {"Low": 0.6, "Medium": 0.3, "High": 0.1},
    "transition": {
        "Low": {"Low": 0.8, "Medium": 0.18, "High": 0.02},
        "Medium": {"Low": 0.15, "Medium": 0.7, "High": 0.15},
        "High": {"Low": 0.05, "Medium": 0.2, "High": 0.75}},
    "emission": {
        "Low": {"Low": 0.85, "Medium": 0.14, "High": 0.01},
        "Medium": {"Low": 0.1, "Medium": 0.8, "High": 0.1},
        "High": {"Low": 0.02, "Medium": 0.18, "High": 0.8}},
}


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def tiny_scenario(tmp_path):
    d = tmp_path / "Tiny"
    d.mkdir()
    nodes = [{"id": n, "x": x, "y": 0} for n, x in (("A", 0), ("B", 100), ("C", 200))]
    roads = [{"from": u, "to": v, "distance": 100, "speed": 10}
             for u, v in (("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"))]
    (d / "city_graph.json").write_text(json.dumps({"nodes": nodes, "roads": roads}))
    (d / "road_probabilities.json").write_text(json.dumps(CPT))
    (d / "traffic_probabilities.json").write_text(json.dumps(TRAFFIC))
    write_csv(d / "responders.csv",
              ["responder_id", "type", "start_location", "capabilities",
               "availability", "time_until_available"],
              [["R1", "Ambulance", "A", "Medical", "Available", "0"],
               ["R2", "FireTruck", "C", "Rescue", "Available", "0"]])
    write_csv(d / "incidents.csv",
              ["incident_id", "location", "priority", "deadline", "required_capability",
               "timestamp", "time_required"],
              [["I1", "B", "Critical", "50", "Medical", "0", "10"],
               ["I2", "B", "High", "60", "Rescue", "5", "10"],
               ["I3", "C", "Medium", "80", "Medical", "20", "10"],
               ["I4", "A", "Low", "80", "Security", "0", "10"]])   # nobody has Security
    write_csv(d / "road_reports.csv", ["from", "to", "weather", "source", "reported_status"],
              [["A", "B", "Clear", "Police", "Open"]])
    write_csv(d / "observed_traffic.csv",
              ["observation_nr", "from", "to", "traffic_observation"],
              [["1", "A", "B", "Low"], ["2", "A", "B", "Low"]])
    return d


def test_run_creates_valid_plan_and_statistics(tiny_scenario, tmp_path, capsys):
    out = tmp_path / "out"
    run_application(tiny_scenario, out)

    plan = json.loads((out / "plan.json").read_text())
    assert {e["incident_id"] for e in plan} == {"I1", "I2", "I3"}      # I4 impossible
    for e in plan:
        assert set(e) == {"responder_id", "incident_id", "path", "time"}
        assert isinstance(e["path"], list) and e["time"] >= 0

    text = (out / "statistics.txt").read_text()
    assert "* 3 incidents handled" in text
    assert "* 2 responders used" in text
    assert "I4 (no_capable_responder)" in text
    assert "* Deadlines met : 75.0%" in text                           # 3 of 4 incidents
    assert "Validation: OK" in capsys.readouterr().out


def test_independent_validation_of_the_generated_plan(tiny_scenario, tmp_path):
    out = tmp_path / "out"
    run_application(tiny_scenario, out)
    data = load_scenario(tiny_scenario)
    roads = RoadConditionModel.from_scenario(data)
    planner = RoutePlanner(data.city_graph, roads, TrafficModel.from_scenario(data))
    report = validate_plan(json.loads((out / "plan.json").read_text()), data, planner, roads)
    assert report.ok, report.errors


def test_missing_scenario_exits_with_error(tmp_path):
    with pytest.raises(SystemExit):
        run_application(tmp_path / "does_not_exist", tmp_path / "out")


@pytest.mark.parametrize("scenario", ["Scenario1", "Scenario2", "Scenario3"])
def test_provided_scenarios_give_valid_plans(scenario, tmp_path):
    path = Path("Scenarios") / scenario
    if not path.is_dir():
        pytest.skip(f"{path} not present")
    run_application(path, tmp_path / "out")       # exits with code 1 if the plan is invalid
    assert (tmp_path / "out" / "plan.json").is_file()
    assert (tmp_path / "out" / "statistics.txt").is_file()