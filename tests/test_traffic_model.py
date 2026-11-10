"""Tests for the HMM traffic model.

Run from the project root: python -m pytest tests/test_traffic_model.py -v
"""

from pathlib import Path

import pytest

from src.traffic_model import (
    TrafficModel,
    forward_filter,
    predict,
)

PROBS = {
    "initial": {"Low": 0.60, "Medium": 0.30, "High": 0.10},
    "transition": {
        "Low": {"Low": 0.80, "Medium": 0.18, "High": 0.02},
        "Medium": {"Low": 0.15, "Medium": 0.70, "High": 0.15},
        "High": {"Low": 0.05, "Medium": 0.20, "High": 0.75},
    },
    "emission": {
        "Low": {"Low": 0.85, "Medium": 0.14, "High": 0.01},
        "Medium": {"Low": 0.10, "Medium": 0.80, "High": 0.10},
        "High": {"Low": 0.02, "Medium": 0.18, "High": 0.80},
    },
}


def test_no_observations_returns_initial():
    assert forward_filter([], PROBS) == pytest.approx(PROBS["initial"])


def test_single_observation_hand_computed():
    # Low: .6*.85=.51  Medium: .3*.10=.03  High: .1*.02=.002  -> sum .542
    result = forward_filter(["Low"], PROBS)
    assert result["Low"] == pytest.approx(0.51 / 0.542)
    assert result["Medium"] == pytest.approx(0.03 / 0.542)
    assert result["High"] == pytest.approx(0.002 / 0.542)


def test_two_observations_hand_computed():
    first = {"Low": 0.51 / 0.542, "Medium": 0.03 / 0.542, "High": 0.002 / 0.542}
    pred_high = (
        first["Low"] * 0.02 + first["Medium"] * 0.15 + first["High"] * 0.75
    )
    pred_low = first["Low"] * 0.80 + first["Medium"] * 0.15 + first["High"] * 0.05
    pred_med = first["Low"] * 0.18 + first["Medium"] * 0.70 + first["High"] * 0.20
    scores = {
        "Low": pred_low * 0.01,    # P(obs=High | Low)
        "Medium": pred_med * 0.10,  # P(obs=High | Medium)
        "High": pred_high * 0.80,   # P(obs=High | High)
    }
    total = sum(scores.values())
    result = forward_filter(["Low", "High"], PROBS)
    for level in scores:
        assert result[level] == pytest.approx(scores[level] / total)


def test_result_is_a_distribution():
    result = forward_filter(["Low", "High", "Medium", "High", "High"], PROBS)
    assert sum(result.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in result.values())


def test_many_high_observations_make_high_most_likely():
    result = forward_filter(["High"] * 6, PROBS)
    assert max(result, key=result.get) == "High"


def test_long_sequence_does_not_underflow():
    result = forward_filter(["Low", "High"] * 2000, PROBS)
    assert sum(result.values()) == pytest.approx(1.0)


def test_predict_zero_steps_is_identity():
    belief = forward_filter(["High", "High"], PROBS)
    assert predict(belief, 0, PROBS) == pytest.approx(belief)


def test_predict_one_step_uses_transition_matrix():
    belief = {"Low": 1.0, "Medium": 0.0, "High": 0.0}
    assert predict(belief, 1, PROBS) == pytest.approx(PROBS["transition"]["Low"])


def test_prediction_forgets_history():
    # Far in the future both starting points converge to the same distribution.
    a = predict({"Low": 1.0, "Medium": 0.0, "High": 0.0}, 200, PROBS)
    b = predict({"Low": 0.0, "Medium": 0.0, "High": 1.0}, 200, PROBS)
    assert a == pytest.approx(b, abs=1e-6)


def test_model_lookup_and_directions_are_separate():
    model = TrafficModel(
        {("A", "B"): forward_filter(["High"] * 5, PROBS),
         ("B", "A"): forward_filter(["Low"] * 5, PROBS)},
        PROBS,
    )
    assert model.expected_multiplier("A", "B") > model.expected_multiplier("B", "A")


def test_road_without_history_uses_initial_distribution():
    model = TrafficModel({}, PROBS)
    assert model.get("X", "Y") == pytest.approx(PROBS["initial"])
    assert not model.has_history("X", "Y")


def test_horizon_is_converted_to_steps():
    model = TrafficModel({("A", "B"): {"Low": 1.0, "Medium": 0.0, "High": 0.0}},
                         PROBS, step_seconds=60.0)
    assert model.get("A", "B", horizon=60.0) == pytest.approx(PROBS["transition"]["Low"])
    assert model.get("A", "B", horizon=0.0) == pytest.approx(
        {"Low": 1.0, "Medium": 0.0, "High": 0.0})


def test_expected_multiplier_between_min_and_max():
    model = TrafficModel({}, PROBS)
    value = model.expected_multiplier("A", "B")
    assert 1.0 <= value <= 1.5


@pytest.mark.parametrize("scenario", ["Scenario1", "Scenario2", "Scenario3"])
def test_provided_scenarios_produce_valid_distributions(scenario):
    path = Path("Scenarios") / scenario
    if not path.is_dir():
        pytest.skip(f"{path} not present")
    from src.data_loader import load_scenario

    data = load_scenario(path)
    model = TrafficModel.from_scenario(data)
    for road in data.city_graph["roads"]:
        dist = model.get(road["from"], road["to"], horizon=300)
        assert sum(dist.values()) == pytest.approx(1.0)