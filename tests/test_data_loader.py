"""Tests for the scenario data loader.

Run from the project root with:
    python -m pytest tests/test_data_loader.py -v

The tests use temporary scenario directories, so they do not modify the
provided Scenarios/ folders.
"""

import csv
import json
from pathlib import Path

import pytest

try:
    # Preferred import when the project uses the proposed src/ package.
    from src.data_loader import (
        ScenarioData,
        ScenarioDataError,
        load_scenario,
        _load_city_graph,
        _load_incidents,
        _load_observed_traffic,
        _load_responders,
    )
except ImportError:
    # Also supports running with data_loader.py directly on the project root.
    from data_loader import (
        ScenarioData,
        ScenarioDataError,
        load_scenario,
        _load_city_graph,
        _load_incidents,
        _load_observed_traffic,
        _load_responders,
    )


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_valid_scenario(directory: Path) -> Path:
    """Create the smallest complete scenario accepted by the loader."""
    directory.mkdir(parents=True, exist_ok=True)

    write_json(
        directory / "city_graph.json",
        {
            "nodes": [{"id": "A"}, {"id": "B"}],
            "roads": [
                {"from": "A", "to": "B", "distance": 100, "speed": 10},
                {"from": "B", "to": "A", "distance": 100, "speed": 10},
            ],
        },
    )

    write_csv(
        directory / "responders.csv",
        [
            "responder_id", "type", "start_location", "capabilities",
            "availability", "time_until_available",
        ],
        [{
            "responder_id": "R1",
            "type": "Ambulance",
            "start_location": "A",
            "capabilities": "Medical; Rescue",
            "availability": "Available",
            "time_until_available": "0",
        }],
    )

    write_csv(
        directory / "incidents.csv",
        [
            "incident_id", "location", "priority", "deadline",
            "required_capability", "timestamp", "time_required",
        ],
        [
            {
                "incident_id": "I2", "location": "B", "priority": "High",
                "deadline": "50", "required_capability": "Medical",
                "timestamp": "10", "time_required": "5",
            },
            {
                "incident_id": "I1", "location": "A", "priority": "Critical",
                "deadline": "20", "required_capability": "Rescue",
                "timestamp": "0", "time_required": "3",
            },
        ],
    )

    write_csv(
        directory / "road_reports.csv",
        ["from", "to", "weather", "source", "reported_status"],
        [{
            "from": "A", "to": "B", "weather": "Rain",
            "source": "Police", "reported_status": "Congested",
        }],
    )

    write_json(directory / "road_probabilities.json", {"example": "road CPT"})
    write_csv(
        directory / "observed_traffic.csv",
        ["observation_nr", "from", "to", "traffic_observation"],
        [
            {"observation_nr": "2", "from": "A", "to": "B", "traffic_observation": "High"},
            {"observation_nr": "1", "from": "A", "to": "B", "traffic_observation": "Low"},
            {"observation_nr": "1", "from": "B", "to": "A", "traffic_observation": "Medium"},
        ],
    )
    write_json(directory / "traffic_probabilities.json", {"example": "traffic HMM"})
    return directory


@pytest.fixture
def valid_scenario(tmp_path):
    return make_valid_scenario(tmp_path / "ScenarioTest")


def test_load_scenario_returns_scenario_data(valid_scenario):
    result = load_scenario(valid_scenario)

    assert isinstance(result, ScenarioData)
    assert result.scenario_dir == valid_scenario.resolve()
    assert len(result.city_graph["nodes"]) == 2
    assert len(result.responders) == 1
    assert len(result.incidents) == 2


def test_load_scenario_normalizes_responder_capabilities(valid_scenario):
    result = load_scenario(valid_scenario)

    assert result.responders[0]["capabilities"] == ["Medical", "Rescue"]
    assert result.responders[0]["time_until_available"] == 0.0


def test_incidents_are_sorted_by_timestamp_then_id(valid_scenario):
    result = load_scenario(valid_scenario)

    assert [item["incident_id"] for item in result.incidents] == ["I1", "I2"]


def test_traffic_observations_are_grouped_and_sorted(valid_scenario):
    result = load_scenario(valid_scenario)

    assert result.observed_traffic[("A", "B")] == ["Low", "High"]
    assert result.observed_traffic[("B", "A")] == ["Medium"]


def test_directed_roads_are_not_treated_as_undirected(valid_scenario):
    result = load_scenario(valid_scenario)

    assert ("A", "B") in result.observed_traffic
    assert ("B", "A") in result.observed_traffic


def test_missing_directory_raises_clear_error(tmp_path):
    with pytest.raises(ScenarioDataError, match="does not exist"):
        load_scenario(tmp_path / "does_not_exist")


def test_file_path_instead_of_directory_raises_error(tmp_path):
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(ScenarioDataError, match="not a directory"):
        load_scenario(file_path)


def test_missing_required_file_is_reported(tmp_path):
    scenario = tmp_path / "Incomplete"
    scenario.mkdir()

    with pytest.raises(ScenarioDataError, match="missing required file"):
        load_scenario(scenario)


def test_invalid_json_raises_scenario_data_error(valid_scenario):
    (valid_scenario / "road_probabilities.json").write_text(
        "{not valid json", encoding="utf-8"
    )

    with pytest.raises(ScenarioDataError, match="Invalid JSON"):
        load_scenario(valid_scenario)


def test_csv_missing_required_column_is_rejected(valid_scenario):
    write_csv(
        valid_scenario / "responders.csv",
        ["responder_id", "type"],
        [{"responder_id": "R1", "type": "Ambulance"}],
    )

    with pytest.raises(ScenarioDataError, match="missing required column"):
        load_scenario(valid_scenario)


def test_duplicate_node_ids_are_rejected(tmp_path):
    path = tmp_path / "city_graph.json"
    write_json(
        path,
        {
            "nodes": [{"id": "A"}, {"id": "A"}],
            "roads": [],
        },
    )

    with pytest.raises(ScenarioDataError, match="Duplicate node id"):
        _load_city_graph(path)


def test_road_with_unknown_node_is_rejected(tmp_path):
    path = tmp_path / "city_graph.json"
    write_json(
        path,
        {
            "nodes": [{"id": "A"}],
            "roads": [{"from": "A", "to": "B", "distance": 10, "speed": 2}],
        },
    )

    with pytest.raises(ScenarioDataError, match="unknown node"):
        _load_city_graph(path)


@pytest.mark.parametrize("bad_value", [0, -1])
def test_road_distance_and_speed_must_be_positive(tmp_path, bad_value):
    path = tmp_path / "city_graph.json"
    write_json(
        path,
        {
            "nodes": [{"id": "A"}, {"id": "B"}],
            "roads": [{"from": "A", "to": "B", "distance": bad_value, "speed": 2}],
        },
    )

    with pytest.raises(ScenarioDataError, match="positive distance"):
        _load_city_graph(path)


def test_duplicate_directed_roads_are_rejected(tmp_path):
    path = tmp_path / "city_graph.json"
    road = {"from": "A", "to": "B", "distance": 10, "speed": 2}
    write_json(
        path,
        {"nodes": [{"id": "A"}, {"id": "B"}], "roads": [road, road]},
    )

    with pytest.raises(ScenarioDataError, match="Duplicate directed road"):
        _load_city_graph(path)


def test_invalid_responder_availability_is_rejected(tmp_path):
    path = tmp_path / "responders.csv"
    write_csv(
        path,
        [
            "responder_id", "type", "start_location", "capabilities",
            "availability", "time_until_available",
        ],
        [{
            "responder_id": "R1", "type": "Ambulance", "start_location": "A",
            "capabilities": "Medical", "availability": "Maybe",
            "time_until_available": "0",
        }],
    )

    with pytest.raises(ScenarioDataError, match="Unknown availability"):
        _load_responders(path)


def test_available_responder_cannot_have_nonzero_wait_time(tmp_path):
    path = tmp_path / "responders.csv"
    write_csv(
        path,
        [
            "responder_id", "type", "start_location", "capabilities",
            "availability", "time_until_available",
        ],
        [{
            "responder_id": "R1", "type": "Ambulance", "start_location": "A",
            "capabilities": "Medical", "availability": "Available",
            "time_until_available": "5",
        }],
    )

    with pytest.raises(ScenarioDataError, match="non-zero time_until_available"):
        _load_responders(path)


def test_duplicate_responder_ids_are_rejected(tmp_path):
    path = tmp_path / "responders.csv"
    row = {
        "responder_id": "R1", "type": "Ambulance", "start_location": "A",
        "capabilities": "Medical", "availability": "Available",
        "time_until_available": "0",
    }
    write_csv(
        path,
        [
            "responder_id", "type", "start_location", "capabilities",
            "availability", "time_until_available",
        ],
        [row, row],
    )

    with pytest.raises(ScenarioDataError, match="Duplicate responder_id"):
        _load_responders(path)


def test_incident_with_deadline_before_timestamp_is_rejected(tmp_path):
    path = tmp_path / "incidents.csv"
    write_csv(
        path,
        [
            "incident_id", "location", "priority", "deadline",
            "required_capability", "timestamp", "time_required",
        ],
        [{
            "incident_id": "I1", "location": "A", "priority": "High",
            "deadline": "5", "required_capability": "Medical",
            "timestamp": "10", "time_required": "2",
        }],
    )

    with pytest.raises(ScenarioDataError, match="deadline before its timestamp"):
        _load_incidents(path)


def test_invalid_incident_priority_is_rejected(tmp_path):
    path = tmp_path / "incidents.csv"
    write_csv(
        path,
        [
            "incident_id", "location", "priority", "deadline",
            "required_capability", "timestamp", "time_required",
        ],
        [{
            "incident_id": "I1", "location": "A", "priority": "Urgent",
            "deadline": "20", "required_capability": "Medical",
            "timestamp": "10", "time_required": "2",
        }],
    )

    with pytest.raises(ScenarioDataError, match="Unknown priority"):
        _load_incidents(path)


def test_responder_start_location_must_exist(valid_scenario):
    path = valid_scenario / "responders.csv"
    write_csv(
        path,
        [
            "responder_id", "type", "start_location", "capabilities",
            "availability", "time_until_available",
        ],
        [{
            "responder_id": "R1", "type": "Ambulance", "start_location": "Z",
            "capabilities": "Medical", "availability": "Available",
            "time_until_available": "0",
        }],
    )

    with pytest.raises(ScenarioDataError, match="unknown node"):
        load_scenario(valid_scenario)


def test_road_report_must_reference_existing_directed_edge(valid_scenario):
    write_csv(
        valid_scenario / "road_reports.csv",
        ["from", "to", "weather", "source", "reported_status"],
        [{
            "from": "B", "to": "B", "weather": "Rain",
            "source": "Police", "reported_status": "Open",
        }],
    )

    with pytest.raises(ScenarioDataError, match="unknown directed road"):
        load_scenario(valid_scenario)


def test_empty_csv_rows_are_ignored(valid_scenario):
    path = valid_scenario / "responders.csv"
    with path.open("a", encoding="utf-8") as file:
        file.write("\n")

    result = load_scenario(valid_scenario)
    assert len(result.responders) == 1


@pytest.mark.parametrize("scenario_name", ["Scenario1", "Scenario2", "Scenario3"])
def test_provided_scenarios_load_if_present(scenario_name):
    """Integration test against the supplied data, when run from project root."""
    scenario_path = Path("Scenarios") / scenario_name
    if not scenario_path.is_dir():
        pytest.skip(f"{scenario_path} is not present in this checkout")

    result = load_scenario(scenario_path)
    assert isinstance(result, ScenarioData)
    assert result.city_graph["nodes"]
    assert result.responders
    assert result.incidents
