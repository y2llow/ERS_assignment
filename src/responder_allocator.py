"""Coordinates the AI Emergency Response System.

Order in which everything runs (run.py -> run_application):

    1. data_loader.load_scenario        read + check the scenario files
    2. road_reasoning (Bayesian net)    weather + noisy reports -> P(Open/Congested/Blocked) per road
    3. traffic_model (HMM)              traffic history -> predicted traffic level per road
    4. route_planner (A*)               cost = distance/speed * status factor * traffic factor
    5. simulation                       time loop; at every event ...
         responder_allocator (CSP)       ... assign free responders to the known incidents
    6. validator                        check the final plan against the constraints
    7. statistics                       performance numbers
    8. write output/plan.json and output/statistics.txt
"""

import json
from pathlib import Path

from src.data_loader import ScenarioDataError, load_scenario
from src.road_reasoning import RoadConditionModel
from src.route_planner import RoutePlanner
from src.simulation import run_simulation
from src.statistics import compute_statistics, format_statistics
from src.traffic_model import TrafficModel
from src.validator import validate_plan


def _tidy(value):
    """Write whole numbers without '.0' (0 instead of 0.0); keep fractional seconds."""
    return int(value) if float(value).is_integer() else value


def build_plan(assignments):
    """plan.json content: one entry per responder-incident assignment."""
    plan = []
    for a in assignments:
        entry = a.to_plan_entry()
        entry["time"] = _tidy(entry["time"])
        plan.append(entry)
    return plan


def write_outputs(plan, statistics_text, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (output_dir / "statistics.txt").write_text(statistics_text, encoding="utf-8")
    return output_dir


def run_application(scenario_path, output_dir="output"):
    """Execute the emergency response system for one scenario."""
    try:
        data = load_scenario(Path(scenario_path))
    except ScenarioDataError as error:
        print(f"\nScenario loading failed: {error}")
        raise SystemExit(1)

    print(f"Scenario {data.scenario_dir.name}: {len(data.city_graph['nodes'])} nodes, "
          f"{len(data.city_graph['roads'])} roads, {len(data.responders)} responders, "
          f"{len(data.incidents)} incidents")

    road_model = RoadConditionModel.from_scenario(data)        # Bayesian reasoning
    traffic_model = TrafficModel.from_scenario(data)           # HMM
    planner = RoutePlanner(data.city_graph, road_model, traffic_model)  # A*

    simulation = run_simulation(data, planner)                 # + constraint-based allocation
    plan = build_plan(simulation.assignments)
    report = validate_plan(plan, data, planner, road_model)
    stats = compute_statistics(data.incidents, simulation)
    out = write_outputs(plan, format_statistics(stats), output_dir)

    print("\nTimeline")
    for line in simulation.log:
        print("  " + line)

    print(f"\nValidation: {'OK' if report.ok else 'INVALID'} "
          f"({report.entries_checked} assignments checked)")
    for message in report.errors:
        print(f"  ERROR: {message}")
    for message in report.warnings:
        print(f"  warning: {message}")

    print(f"\n{format_statistics(stats)}")
    print(f"Written: {out / 'plan.json'} and {out / 'statistics.txt'}")

    if not report.ok:
        raise SystemExit(1)

