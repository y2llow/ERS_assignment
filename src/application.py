
"""Coordinates the AI Emergency Response System."""

from pathlib import Path

from src.data_loader import load_scenario, ScenarioDataError


def run_application(scenario_path):
    """Execute the emergency response system for one scenario."""

    try:
        # Step 1: Load scenario data
        print("Loading scenario...")
        data = load_scenario(Path(scenario_path))

        # Step 2: Display loaded data
        print("\nScenario successfully loaded.")
        print("-" * 40)
        print(f"Scenario: {data.scenario_dir.name}")
        print(f"Nodes: {len(data.city_graph['nodes'])}")
        print(f"Roads: {len(data.city_graph['roads'])}")
        print(f"Responders: {len(data.responders)}")
        print(f"Incidents: {len(data.incidents)}")

        # Future steps:
        # 3. Estimate road conditions using Bayesian reasoning
        # 4. Estimate traffic using the Hidden Markov Model
        # 5. Plan routes
        # 6. Allocate responders
        # 7. Simulate incident handling
        # 8. Validate the generated plan
        # 9. Generate statistics

        print("\nInput validation successful.")
        print("Planning components are not implemented yet.")

    except ScenarioDataError as error:
        print(f"\nScenario loading failed: {error}")
        raise SystemExit(1)
