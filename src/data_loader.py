"""
Load and validate the input data for one emergency-response scenario.

This module deliberately does not make AI decisions. It only:
- reads the scenario's JSON and CSV files;
- converts values into predictable Python types;
- groups historical traffic observations by directed road;
- performs basic structural and cross-reference validation.

Other modules should receive a ScenarioData instance from load_scenario().
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Keep filenames in one place so changes to the assignment's input format
# are easy to manage.
REQUIRED_FILES = {
    "city_graph": "city_graph.json",
    "responders": "responders.csv",
    "incidents": "incidents.csv",
    "road_reports": "road_reports.csv",
    "road_probabilities": "road_probabilities.json",
    "observed_traffic": "observed_traffic.csv",
    "traffic_probabilities": "traffic_probabilities.json",
}

RESPONDER_AVAILABILITIES = {"Available", "Busy"}
INCIDENT_PRIORITIES = {"Critical", "High", "Medium", "Low"}
ROAD_STATUSES = {"Open", "Congested", "Blocked"}
WEATHER_STATES = {"Clear", "Rain", "Storm"}
TRAFFIC_LEVELS = {"Low", "Medium", "High"}
ROAD_REPORT_SOURCES = {"Citizen", "Police", "Sensor"}


@dataclass
class ScenarioData:
    """All input data for a single scenario, in normalized Python structures."""

    scenario_dir: Path
    city_graph: dict[str, Any]
    responders: list[dict[str, Any]]
    incidents: list[dict[str, Any]]
    road_reports: list[dict[str, Any]]
    road_probabilities: dict[str, Any]
    # Key is a directed edge: (from_node, to_node).
    # Value is a list of observations ordered by observation_nr.
    observed_traffic: dict[tuple[str, str], list[str]]
    traffic_probabilities: dict[str, Any]


class ScenarioDataError(ValueError):
    """Raised when scenario files are missing, malformed, or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and report errors with the filename."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ScenarioDataError(
            f"Invalid JSON in '{path.name}' at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ScenarioDataError(f"Could not read '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ScenarioDataError(
            f"Expected '{path.name}' to contain a JSON object."
        )
    return data


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """Read a CSV file, ignoring empty lines and checking its header."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ScenarioDataError(f"'{path.name}' is empty or has no header.")

            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ScenarioDataError(
                    f"'{path.name}' is missing required column(s): {missing}."
                )

            # DictReader naturally skips completely empty lines.
            return [
                {key: (value.strip() if value is not None else "") for key, value in row.items()}
                for row in reader
                if row and any(value and value.strip() for value in row.values() if value is not None)
            ]
    except OSError as exc:
        raise ScenarioDataError(f"Could not read '{path}': {exc}") from exc


def _required_text(row: dict[str, str], field: str, filename: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ScenarioDataError(
            f"Missing value for '{field}' in '{filename}'."
        )
    return value


def _number(
    row: dict[str, str],
    field: str,
    filename: str,
    *,
    integer: bool = False,
) -> int | float:
    raw_value = _required_text(row, field, filename)
    try:
        return int(raw_value) if integer else float(raw_value)
    except ValueError as exc:
        expected = "an integer" if integer else "a number"
        raise ScenarioDataError(
            f"Expected {expected} in column '{field}' of '{filename}', "
            f"got {raw_value!r}."
        ) from exc


def _load_city_graph(path: Path) -> dict[str, Any]:
    graph = _read_json(path)

    if not isinstance(graph.get("nodes"), list):
        raise ScenarioDataError("'city_graph.json' must contain a 'nodes' list.")
    if not isinstance(graph.get("roads"), list):
        raise ScenarioDataError("'city_graph.json' must contain a 'roads' list.")

    node_ids: set[str] = set()
    for index, node in enumerate(graph["nodes"]):
        if not isinstance(node, dict) or not node.get("id"):
            raise ScenarioDataError(f"Invalid node at index {index} in 'city_graph.json'.")
        node_id = str(node["id"])
        if node_id in node_ids:
            raise ScenarioDataError(f"Duplicate node id {node_id!r} in 'city_graph.json'.")
        node_ids.add(node_id)

    directed_edges: set[tuple[str, str]] = set()
    for index, road in enumerate(graph["roads"]):
        if not isinstance(road, dict):
            raise ScenarioDataError(f"Invalid road at index {index} in 'city_graph.json'.")

        start = str(road.get("from", ""))
        end = str(road.get("to", ""))
        if not start or not end:
            raise ScenarioDataError(
                f"Road at index {index} must have 'from' and 'to' fields."
            )
        if start not in node_ids or end not in node_ids:
            raise ScenarioDataError(
                f"Road {start!r} -> {end!r} references an unknown node."
            )
        if (start, end) in directed_edges:
            raise ScenarioDataError(
                f"Duplicate directed road {start!r} -> {end!r}."
            )
        directed_edges.add((start, end))

        for field in ("distance", "speed"):
            try:
                value = float(road[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ScenarioDataError(
                    f"Road {start!r} -> {end!r} has an invalid '{field}'."
                ) from exc
            if value <= 0:
                raise ScenarioDataError(
                    f"Road {start!r} -> {end!r} must have positive {field}."
                )

    return graph


def _load_responders(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(
        path,
        {
            "responder_id",
            "type",
            "start_location",
            "capabilities",
            "availability",
            "time_until_available",
        },
    )

    responders: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in rows:
        responder_id = _required_text(row, "responder_id", path.name)
        if responder_id in seen_ids:
            raise ScenarioDataError(f"Duplicate responder_id {responder_id!r}.")
        seen_ids.add(responder_id)

        availability = _required_text(row, "availability", path.name)
        if availability not in RESPONDER_AVAILABILITIES:
            raise ScenarioDataError(
                f"Unknown availability {availability!r} for responder {responder_id!r}."
            )

        capabilities = [
            capability.strip()
            for capability in _required_text(row, "capabilities", path.name).split(";")
            if capability.strip()
        ]

        time_until_available = float(
            _number(row, "time_until_available", path.name)
        )
        if time_until_available < 0:
            raise ScenarioDataError(
                f"Responder {responder_id!r} has negative time_until_available."
            )

        if availability == "Available" and time_until_available != 0:
            raise ScenarioDataError(
                f"Responder {responder_id!r} is marked Available but has "
                "non-zero time_until_available."
            )

        responders.append(
            {
                "responder_id": responder_id,
                "type": _required_text(row, "type", path.name),
                "start_location": _required_text(row, "start_location", path.name),
                "capabilities": capabilities,
                "availability": availability,
                "time_until_available": time_until_available,
            }
        )

    return responders


def _load_incidents(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(
        path,
        {
            "incident_id",
            "location",
            "priority",
            "deadline",
            "required_capability",
            "timestamp",
            "time_required",
        },
    )

    incidents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in rows:
        incident_id = _required_text(row, "incident_id", path.name)
        if incident_id in seen_ids:
            raise ScenarioDataError(f"Duplicate incident_id {incident_id!r}.")
        seen_ids.add(incident_id)

        priority = _required_text(row, "priority", path.name)
        if priority not in INCIDENT_PRIORITIES:
            raise ScenarioDataError(
                f"Unknown priority {priority!r} for incident {incident_id!r}."
            )

        timestamp = float(_number(row, "timestamp", path.name))
        deadline = float(_number(row, "deadline", path.name))
        time_required = float(_number(row, "time_required", path.name))

        if timestamp < 0 or deadline < 0 or time_required < 0:
            raise ScenarioDataError(
                f"Incident {incident_id!r} contains a negative time value."
            )
        if deadline < timestamp:
            raise ScenarioDataError(
                f"Incident {incident_id!r} has a deadline before its timestamp."
            )

        incidents.append(
            {
                "incident_id": incident_id,
                "location": _required_text(row, "location", path.name),
                "priority": priority,
                "deadline": deadline,
                "required_capability": _required_text(
                    row, "required_capability", path.name
                ),
                "timestamp": timestamp,
                "time_required": time_required,
            }
        )

    # Sorting here makes processing predictable. The simulation module must
    # still enforce that incidents are only revealed at their timestamp.
    incidents.sort(key=lambda incident: (incident["timestamp"], incident["incident_id"]))
    return incidents


def _load_road_reports(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(
        path,
        {"from", "to", "weather", "source", "reported_status"},
    )

    reports: list[dict[str, Any]] = []
    for row in rows:
        weather = _required_text(row, "weather", path.name)
        source = _required_text(row, "source", path.name)
        reported_status = _required_text(row, "reported_status", path.name)

        if weather not in WEATHER_STATES:
            raise ScenarioDataError(f"Unknown weather state {weather!r}.")
        if source not in ROAD_REPORT_SOURCES:
            raise ScenarioDataError(f"Unknown road report source {source!r}.")
        if reported_status not in ROAD_STATUSES:
            raise ScenarioDataError(f"Unknown reported road status {reported_status!r}.")

        reports.append(
            {
                "from": _required_text(row, "from", path.name),
                "to": _required_text(row, "to", path.name),
                "weather": weather,
                "source": source,
                "reported_status": reported_status,
            }
        )

    return reports


def _load_observed_traffic(path: Path) -> dict[tuple[str, str], list[str]]:
    rows = _read_csv(
        path,
        {"observation_nr", "from", "to", "traffic_observation"},
    )

    observations: dict[tuple[str, str], list[tuple[int, str]]] = {}

    for row in rows:
        try:
            observation_nr = int(_required_text(row, "observation_nr", path.name))
        except ValueError as exc:
            raise ScenarioDataError(
                f"Invalid observation_nr in '{path.name}': "
                f"{row.get('observation_nr')!r}."
            ) from exc

        start = _required_text(row, "from", path.name)
        end = _required_text(row, "to", path.name)
        traffic = _required_text(row, "traffic_observation", path.name)

        if traffic not in TRAFFIC_LEVELS:
            raise ScenarioDataError(f"Unknown traffic observation {traffic!r}.")

        observations.setdefault((start, end), []).append((observation_nr, traffic))

    # Keep each directed road's sequence in chronological observation order.
    return {
        edge: [traffic for _, traffic in sorted(sequence, key=lambda item: item[0])]
        for edge, sequence in observations.items()
    }


def _validate_references(data: ScenarioData) -> None:
    """Check that responders, incidents, and observations refer to real graph nodes/roads."""
    node_ids = {str(node["id"]) for node in data.city_graph["nodes"]}
    directed_edges = {
        (str(road["from"]), str(road["to"])) for road in data.city_graph["roads"]
    }

    for responder in data.responders:
        if responder["start_location"] not in node_ids:
            raise ScenarioDataError(
                f"Responder {responder['responder_id']!r} starts at unknown node "
                f"{responder['start_location']!r}."
            )

    for incident in data.incidents:
        if incident["location"] not in node_ids:
            raise ScenarioDataError(
                f"Incident {incident['incident_id']!r} is at unknown node "
                f"{incident['location']!r}."
            )

    for report in data.road_reports:
        edge = (report["from"], report["to"])
        if edge not in directed_edges:
            raise ScenarioDataError(
                f"Road report refers to unknown directed road {edge[0]!r} -> {edge[1]!r}."
            )

    for start, end in data.observed_traffic:
        if (start, end) not in directed_edges:
            raise ScenarioDataError(
                f"Traffic observations refer to unknown directed road {start!r} -> {end!r}."
            )


def load_scenario(scenario_dir: str | Path) -> ScenarioData:
    """
    Load one complete scenario directory.

    Example:
        scenario = load_scenario("Scenarios/Scenario1")

    Paths are interpreted relative to the current working directory unless
    an absolute path is provided.
    """
    directory = Path(scenario_dir).expanduser().resolve()

    if not directory.exists():
        raise ScenarioDataError(f"Scenario directory does not exist: {directory}")
    if not directory.is_dir():
        raise ScenarioDataError(f"Scenario path is not a directory: {directory}")

    missing_files = [
        filename
        for filename in REQUIRED_FILES.values()
        if not (directory / filename).is_file()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise ScenarioDataError(
            f"Scenario '{directory.name}' is missing required file(s): {missing}."
        )

    data = ScenarioData(
        scenario_dir=directory,
        city_graph=_load_city_graph(directory / REQUIRED_FILES["city_graph"]),
        responders=_load_responders(directory / REQUIRED_FILES["responders"]),
        incidents=_load_incidents(directory / REQUIRED_FILES["incidents"]),
        road_reports=_load_road_reports(directory / REQUIRED_FILES["road_reports"]),
        road_probabilities=_read_json(
            directory / REQUIRED_FILES["road_probabilities"]
        ),
        observed_traffic=_load_observed_traffic(
            directory / REQUIRED_FILES["observed_traffic"]
        ),
        traffic_probabilities=_read_json(
            directory / REQUIRED_FILES["traffic_probabilities"]
        ),
    )

    _validate_references(data)
    return data
