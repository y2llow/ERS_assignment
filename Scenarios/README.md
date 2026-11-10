# Emergency Response System (ERS) – AI Practicum

Decision-support system that assigns emergency responders to incidents after a storm and
plans their routes under uncertain road conditions.

## Install

Python 3.10 or newer. The system itself only uses the standard library; only the tests need `pytest`.

    git clone https://github.com/y2llow/ERS_assignment.git
    cd ERS_assignment

    # optional: virtual environment
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate

    pip install pytest

## Run

    python run.py                                   # default: Scenarios/Scenario1
    python run.py Scenarios/Scenario1
    python run.py Scenarios/Scenario2
    python run.py Scenarios/Scenario3
    python run.py Scenarios/Scenario3 --output out3  # write to another directory

(Use `python3` instead of `python` if that is how Python is installed on your system.)

`run.py` writes two files to the output directory (default `output/`, overwritten on every run):

- `plan.json`      - list of `{responder_id, incident_id, path, time}` (time = departure, seconds)
- `statistics.txt` - incidents handled, responders used, average response time (s), deadlines met (%)

The generated plan is checked by `src/validator.py` before it is accepted; the run exits with
code 1 if the plan is invalid. All times are in seconds, distances in metres, speeds in m/s.

## Tests

Run all commands from the project root.

    python -m pytest tests -v                                # everything
    python -m pytest tests/test_responder_allocator.py -v    # one file
    python -m pytest tests -q                                # short output

Tests that use the provided scenarios are skipped if the `Scenarios/` folder is missing.

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