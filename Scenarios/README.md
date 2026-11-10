# Scenario data

Each scenario folder is a complete input dataset.

## Units

- Coordinates and road `distance`: metres.
- Road `speed`: metres per second (30 on main roads, 15 on secondary roads).
- All simulation times and durations: seconds, including incident timestamps, arrival deadlines, handling durations, responder availability, and assignment start times. Fractional seconds are allowed.

## Interpretation

- Road reports are available at time 0. Weather is local to the reported road.
- Traffic observations are historical and available before time 0. `observation_nr` orders a separate sequence for each directed road; it is not a simulation timestamp.
- Traffic levels (Low/Medium/High) differ from road statuses (Open/Congested/Blocked). Define and justify how traffic affects your planning; no fixed traffic multipliers are prescribed.

Document and defend any additional assumptions.
# Emergency Response System (ERS) – AI Practicum

Decision-support system that assigns emergency responders to incidents after a storm and
plans their routes under uncertain road conditions.

## Install

Python 3.10 or newer. The system itself only uses the standard library.
Only the tests need an extra package:

    pip install pytest

## Run

    python run.py Scenarios/Scenario1
    python run.py Scenarios/Scenario2 --output output
    python run.py Scenarios/Scenario3

`run.py` writes two files to the output directory (default `output/`, overwritten on every run):

- `plan.json`      – list of `{responder_id, incident_id, path, time}` (time = departure, seconds)
- `statistics.txt` – incidents handled, responders used, average response time (s), deadlines met (%)

The generated plan is checked by `src/validator.py` before it is accepted; the run exits with
code 1 if the plan is invalid. All times are in seconds, distances in metres, speeds in m/s.

## Tests

    python -m pytest tests -v

## How it works (pipeline in `src/application.py`)

| Step | Module | AI technique |
|------|--------|--------------|
| 1 | `data_loader.py` | read + validate the scenario files |
| 2 | `road_reasoning.py` | Bayesian network: Weather -> Status -> Citizen/Police/Sensor reports |
| 3 | `traffic_model.py` | Hidden Markov Model: filtering over history + prediction |
| 4 | `route_planner.py` | A* search; cost = distance/speed x status factor x traffic factor |
| 5 | `simulation.py` | discrete-event simulation; incidents are only known from their timestamp |
| 5 | `responder_allocator.py` | constraint-based allocation (CSP, backtracking + branch and bound) |
| 6 | `validator.py` | independent check of the final plan |
| 7 | `statistics.py` | performance numbers |

## Layout

    run.py                 entry point
    src/                   source code
    tests/                 pytest tests
    Scenarios/             provided input data
    output/                generated plan.json and statistics.txt

Assumptions and limitations are documented in the docstring at the top of each module.