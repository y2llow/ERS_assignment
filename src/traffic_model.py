"""Implements a Hidden Markov Model (HMM) to estimate traffic conditions from historical observations.

Model (per directed road):
    hidden state   : traffic level  in {Low, Medium, High}
    observation    : observed traffic level (noisy)
    initial        : P(X_1)
    transition     : P(X_t | X_{t-1})
    emission       : P(O_t | X_t)

Procedure:
    1. FILTERING (forward algorithm) over the historical observations gives
       P(X_now | o_1..o_n): the belief at simulation time 0.
    2. PREDICTION: for a later time, the belief is pushed through the
       transition matrix k times (no new observations are available):
           P(X_{n+k} | o_1..o_n) = P(X_{n+k-1} | o_1..o_n) . T
    3. The belief is turned into an expected travel-time multiplier that the
       route planner can use.

Assumptions (document in the report):
    - observation_nr only orders the history; it is NOT a simulation time.
    - One HMM transition step corresponds to STEP_SECONDS of simulation time.
    - Roads without history use the initial distribution as their belief.
    - TRAFFIC_MULTIPLIER values are our own choice (none prescribed).
    - Traffic is independent of road status (Bayes), so the two multipliers
      can be combined without double-counting evidence.
"""

from __future__ import annotations

from typing import Any

LEVELS = ("Low", "Medium", "High")

TRAFFIC_MULTIPLIER = {"Low": 1.0, "Medium": 1.25, "High": 1.5}
STEP_SECONDS = 60.0

RoadKey = tuple[str, str]
Distribution = dict[str, float]


def _normalise(scores: Distribution) -> Distribution:
    total = sum(scores.values())
    if total <= 0:
        return {level: 1.0 / len(LEVELS) for level in LEVELS}
    return {level: value / total for level, value in scores.items()}


def forward_filter(observations: list[str], probs: dict[str, Any]) -> Distribution:
    """Forward algorithm with normalisation at every step (avoids underflow).

    Returns P(X_n | o_1..o_n). With no observations returns the initial distribution.
    """
    initial = probs["initial"]
    transition = probs["transition"]
    emission = probs["emission"]

    belief: Distribution | None = None
    for observation in observations:
        if belief is None:
            predicted = {s: initial[s] for s in LEVELS}
        else:
            predicted = {
                s: sum(belief[p] * transition[p][s] for p in LEVELS) for s in LEVELS
            }
        belief = _normalise({s: predicted[s] * emission[s][observation] for s in LEVELS})

    return belief if belief is not None else {s: initial[s] for s in LEVELS}


def predict(belief: Distribution, steps: int, probs: dict[str, Any]) -> Distribution:
    """Push a belief `steps` transitions forward without new evidence."""
    transition = probs["transition"]
    current = dict(belief)
    for _ in range(max(0, steps)):
        current = {
            s: sum(current[p] * transition[p][s] for p in LEVELS) for s in LEVELS
        }
    return _normalise(current)


class TrafficModel:
    """Lets the route planner look up predicted traffic for a directed road."""

    def __init__(
        self,
        filtered: dict[RoadKey, Distribution],
        probabilities: dict[str, Any],
        step_seconds: float = STEP_SECONDS,
        multipliers: dict[str, float] | None = None,
    ):
        self._filtered = filtered
        self._probs = probabilities
        self._step_seconds = step_seconds
        self._multipliers = dict(multipliers or TRAFFIC_MULTIPLIER)

    @classmethod
    def from_scenario(cls, data) -> "TrafficModel":
        """Build from a ScenarioData object (see data_loader.load_scenario)."""
        filtered = {
            edge: forward_filter(obs, data.traffic_probabilities)
            for edge, obs in data.observed_traffic.items()
        }
        return cls(filtered, data.traffic_probabilities)

    def get(self, start: str, end: str, horizon: float = 0.0) -> Distribution:
        """P(traffic level) for the road, `horizon` seconds after time 0."""
        belief = self._filtered.get((start, end))
        if belief is None:
            belief = {s: self._probs["initial"][s] for s in LEVELS}
        steps = int(round(max(0.0, horizon) / self._step_seconds))
        return predict(belief, steps, self._probs)

    def expected_multiplier(self, start: str, end: str, horizon: float = 0.0) -> float:
        """E[travel-time multiplier] caused by traffic."""
        dist = self.get(start, end, horizon)
        return sum(dist[s] * self._multipliers[s] for s in LEVELS)

    def has_history(self, start: str, end: str) -> bool:
        return (start, end) in self._filtered