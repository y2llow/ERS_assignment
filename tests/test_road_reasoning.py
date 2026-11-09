"""Tests for the Bayesian road-condition estimation.

Run from the project root: python -m pytest tests/test_road_reasoning.py -v
"""

import json
from pathlib import Path

import pytest

from src.road_reasoning import (
    RoadConditionModel,
    estimate_road_conditions,
    posterior_status,
    prior_status,
)

CPT = {
    "weather_prior": {"Clear": 0.5, "Rain": 0.3, "Storm": 0.2},
    "road_status_given_weather": {
        "Clear": {"Open": 0.95, "Congested": 0.04, "Blocked": 0.01},
        "Rain": {"Open": 0.60, "Congested": 0.30, "Blocked": 0.10},
        "Storm": {"Open": 0.15, "Congested": 0.35, "Blocked": 0.50},
    },
    "citizen_report": {
        "Open": {"Open": 0.70, "Congested": 0.20, "Blocked": 0.10},
        "Congested": {"Open": 0.20, "Congested": 0.60, "Blocked": 0.20},
        "Blocked": {"Open": 0.10, "Congested": 0.20, "Blocked": 0.70},
    },
    "police_report": {
        "Open": {"Open": 0.92, "Congested": 0.06, "Blocked": 0.02},
        "Congested": {"Open": 0.08, "Congested": 0.84, "Blocked": 0.08},
        "Blocked": {"Open": 0.02, "Congested": 0.08, "Blocked": 0.90},
    },
    "sensor_report": {
        "Open": {"Open": 0.88, "Congested": 0.10, "Blocked": 0.02},
        "Congested": {"Open": 0.10, "Congested": 0.80, "Blocked": 0.10},
        "Blocked": {"Open": 0.03, "Congested": 0.07, "Blocked": 0.90},
    },
}


def test_no_reports_equals_weather_cpt():
    result = posterior_status("Storm", [], CPT)
    assert result == pytest.approx({"Open": 0.15, "Congested": 0.35, "Blocked": 0.50})


def test_hand_computed_single_report():
    # Rain + Police says Blocked
    # Open: .60*.02=.012  Cong: .30*.08=.024  Blocked: .10*.90=.09  -> sum .126
    result = posterior_status("Rain", [("Police", "Blocked")], CPT)
    assert result["Open"] == pytest.approx(0.012 / 0.126)
    assert result["Congested"] == pytest.approx(0.024 / 0.126)
    assert result["Blocked"] == pytest.approx(0.09 / 0.126)


def test_result_is_normalised():
    result = posterior_status(
        "Rain", [("Citizen", "Open"), ("Police", "Congested"), ("Sensor", "Blocked")], CPT
    )
    assert sum(result.values()) == pytest.approx(1.0)
    assert set(result) == {"Open", "Congested", "Blocked"}


def test_three_blocked_reports_in_storm_is_almost_certainly_blocked():
    reports = [("Citizen", "Blocked"), ("Police", "Blocked"), ("Sensor", "Blocked")]
    assert posterior_status("Storm", reports, CPT)["Blocked"] > 0.99


def test_police_is_trusted_more_than_citizen():
    citizen = posterior_status("Rain", [("Citizen", "Blocked")], CPT)["Blocked"]
    police = posterior_status("Rain", [("Police", "Blocked")], CPT)["Blocked"]
    assert police > citizen


def test_order_of_reports_does_not_matter():
    a = posterior_status("Rain", [("Citizen", "Open"), ("Police", "Blocked")], CPT)
    b = posterior_status("Rain", [("Police", "Blocked"), ("Citizen", "Open")], CPT)
    assert a == pytest.approx(b)


def test_prior_without_weather_is_valid_distribution():
    prior = prior_status(CPT)
    assert sum(prior.values()) == pytest.approx(1.0)
    # Open: .5*.95 + .3*.6 + .2*.15 = 0.685
    assert prior["Open"] == pytest.approx(0.685)


def test_estimate_covers_every_directed_road_and_uses_directions_separately():
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "roads": [
            {"from": "A", "to": "B"}, {"from": "B", "to": "A"}, {"from": "B", "to": "C"},
        ],
    }
    reports = [
        {"from": "A", "to": "B", "weather": "Storm", "source": "Police", "reported_status": "Blocked"},
        {"from": "B", "to": "A", "weather": "Storm", "source": "Police", "reported_status": "Open"},
    ]
    est = estimate_road_conditions(graph, reports, CPT)

    assert set(est) == {("A", "B"), ("B", "A"), ("B", "C")}
    assert est[("A", "B")]["Blocked"] > 0.8
    assert est[("B", "A")]["Open"] > est[("A", "B")]["Open"]
    # B->C has no reports: falls back to prior
    assert est[("B", "C")] == pytest.approx(prior_status(CPT))


def test_mixed_weather_reports_are_still_a_distribution():
    graph = {"nodes": [{"id": "A"}, {"id": "B"}], "roads": [{"from": "A", "to": "B"}]}
    reports = [
        {"from": "A", "to": "B", "weather": "Clear", "source": "Police", "reported_status": "Open"},
        {"from": "A", "to": "B", "weather": "Storm", "source": "Citizen", "reported_status": "Blocked"},
    ]
    est = estimate_road_conditions(graph, reports, CPT)[("A", "B")]
    assert sum(est.values()) == pytest.approx(1.0)


def test_model_wrapper_lookup():
    model = RoadConditionModel({("A", "B"): {"Open": 0.1, "Congested": 0.25, "Blocked": 0.65}})
    assert model.get("A", "B")["Blocked"] == 0.65
    assert model.p_blocked("A", "B") == 0.65


@pytest.mark.parametrize("scenario", ["Scenario1", "Scenario2", "Scenario3"])
def test_provided_scenarios_produce_valid_distributions(scenario):
    path = Path("Scenarios") / scenario
    if not path.is_dir():
        pytest.skip(f"{path} not present")
    from src.data_loader import load_scenario

    data = load_scenario(path)
    est = RoadConditionModel.from_scenario(data).all()

    assert len(est) == len(data.city_graph["roads"])
    for dist in est.values():
        assert sum(dist.values()) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in dist.values())