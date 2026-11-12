# Emergency Response System (ERS)

Artificial Intelligence Practicum — University of Antwerp  
Student: Eray Ayyildiz (s0243673)

## Overview

This project simulates the allocation of emergency responders to incidents in a road network. It combines probabilistic road-condition estimation, traffic prediction, route search and constraint-based responder allocation. Incidents become known only at their timestamps; the system does not use future incident information when making decisions.

Processing pipeline:

`Scenario data -> Bayesian road model + HMM traffic model -> road travel costs -> A* routing -> CSP allocation -> simulation -> validation and statistics`

## Requirements

- Python 3 (use a recent version compatible with the project)
- `pytest` for running the tests

No separate dependency manifest is currently listed in the repository root. If an import fails, check the project's imports and install the required package in your environment.

## Quick start

From the project root:

```bash
git clone https://github.com/y2llow/ERS_assignment.git
cd ERS_assignment
python -m pip install pytest
python run.py
```

The default scenario is `Scenarios/Scenario1`. To run each supplied scenario separately:

```bash
python run.py Scenarios/Scenario1 --output output/scenario1
python run.py Scenarios/Scenario2 --output output/scenario2
python run.py Scenarios/Scenario3 --output output/scenario3
```

Each run creates two files in its chosen output directory:

- `plan.json`: responder-to-incident assignments and planned routes.
- `statistics.txt`: performance statistics for the simulated scenario.

To run the automated tests:

```bash
python -m pytest -q
```

Run these commands from the repository root. On systems where `python` refers to Python 2 or is unavailable, use `python3` instead.

## Repository structure

```text
ERS_assignment/
  Scenarios/               Supplied scenario inputs
  src/                     Implementation modules
    application.py          Application orchestration
    road_reasoning.py       Bayesian road-condition estimates
    traffic_model.py        Hidden Markov Model for traffic
    route_planner.py        A* routing and road costs
    responder_allocator.py  Constraint-based assignments
    simulation.py           Time-ordered simulation
    validator.py            Plan validation
  tests/                   Automated tests
  run.py                   Command-line entry point
  README.md
```

The structure above highlights the principal modules described in the project report.

## AI methods and design decisions

### Bayesian reasoning

For each directed road, the model estimates probabilities for `Open`, `Congested` and `Blocked` using weather and potentially noisy citizen, police and sensor reports. Posterior probabilities are normalized. Roads without reports use a prior estimate. The road-status estimate is treated as static during a run.

### Hidden Markov Model (HMM)

Traffic has three hidden levels: `Low`, `Medium` and `High`. Forward filtering processes historical observations, and the transition matrix predicts future traffic. The model uses an assumed 60-second transition step and traffic multipliers of 1.0, 1.25 and 1.5. These are modelling assumptions, not empirically calibrated parameters.

### A* route planning

Road travel cost is estimated in seconds from distance, speed, road-status factor and traffic factor. A* uses a distance-based heuristic scaled by maximum road speed. Roads whose blocked probability exceeds a configured threshold are excluded; a more permissive fallback may be used when a route is otherwise unreachable. The planner can be compared with Dijkstra by using a zero heuristic.

### Constraint-based allocation

Known, unassigned incidents are matched to suitable responders while respecting capability, availability, route feasibility, deadlines and non-overlap. The solver uses backtracking and branch-and-bound, with priority-aware ordering and a greedy initial solution. It prioritizes the total priority weight served, then weighted response time. A search-node limit means that some runs can return the best feasible solution found without proving optimality.

## Validation and testing

The test suite covers probabilistic inference, traffic filtering, routing, assignment constraints, simulation timing and invalid plans. Run `python -m pytest -q` to obtain the actual pass/fail count for the checked-out version.

The validator checks plan properties such as responder capabilities, route connectivity, timing, deadlines and assignment overlap. Validation establishes consistency with the implemented model; it does not independently prove that the predicted road conditions match reality.

## Reported scenario results

The following values are from the submitted project report. Re-run the scenarios to verify them against the exact repository revision being assessed.

| Scenario | Incidents handled | Responders used | Mean response time | Deadlines met (all incidents) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3 / 7 | 3 | 60.8 s | 42.9% |
| 2 | 5 / 12 | 4 | 31.0 s | 41.7% |
| 3 | 11 / 20 | 6 | 700.5 s | 55.0% |

The report counts unhandled incidents as missed deadlines and excludes them from mean response time. All assigned incidents in those reported runs met their modelled deadlines.

## Known limitations

- Decisions are made online and may be optimal for a decision round without being globally optimal across the simulation.
- The allocator does not reserve responders that will become available shortly.
- Responders remain at their last incident location; no return-to-base or repositioning is modelled.
- A more permissive blocked-road threshold is not necessarily tried merely because the current route arrives too late.
- Road and traffic parameters have not been established by sensitivity analysis.
- The validator shares assumptions and estimates with the planner.

Possible extensions include scheduling responders from their future availability times, testing a riskier fallback when a deadline is narrowly missed, including busy time in the allocation objective and comparing parameter settings across all scenarios.

## Reproducing the evaluation

1. Clone the repository and install `pytest`.
2. Run `python -m pytest -q` and note any failures or skipped tests.
3. Execute all three scenario commands above, using separate output directories.
4. Inspect each `plan.json` and `statistics.txt` and compare the results with the report.
5. Review the source modules and tests for implementation details and assumptions.

The project report contains the design rationale, LLM collaboration record, scenario analysis and critical reflection.
