"""Uses Bayesian reasoning to estimate road conditions based on weather and road reports.

Bayesian network (per directed road):

    Weather -> Status -> CitizenReport
                Status -> PoliceReport
                Status -> SensorReport

The reports are conditionally independent given the road status, so:

    P(Status | weather, reports)  ∝  P(Status | weather) * Π_i P(report_i | Status)

The result is normalised so the three probabilities sum to 1.

Output format (one dict per directed road):
    {"Open": 0.10, "Congested": 0.25, "Blocked": 0.65}
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

STATUSES = ("Open", "Congested", "Blocked")

# Maps the 'source' column in road_reports.csv to the CPT name in road_probabilities.json.
SOURCE_TO_CPT = {
    "Citizen": "citizen_report",
    "Police": "police_report",
    "Sensor": "sensor_report",
}

RoadKey = tuple[str, str]
Distribution = dict[str, float]


def _normalise(scores: Distribution) -> Distribution:
    """Divide by the sum so the values form a probability distribution."""
    total = sum(scores.values())
    if total <= 0:
        # Contradictory evidence with zero probability everywhere: fall back to uniform.
        return {status: 1.0 / len(STATUSES) for status in STATUSES}
    return {status: value / total for status, value in scores.items()}


def prior_status(road_probabilities: dict[str, Any]) -> Distribution:
    """P(Status) = sum_w P(w) * P(Status | w). Used when no weather is known for a road."""
    weather_prior = road_probabilities["weather_prior"]
    cpt = road_probabilities["road_status_given_weather"]
    return {
        status: sum(weather_prior[w] * cpt[w][status] for w in weather_prior)
        for status in STATUSES
    }


def posterior_status(
    weather: str | None,
    reports: list[tuple[str, str]],
    road_probabilities: dict[str, Any],
) -> Distribution:
    """
    Posterior P(Status | weather, reports) for one road.

    weather:  observed weather for this road, or None if unknown (then the
              weather is marginalised out using P(Weather)).
    reports:  list of (source, reported_status), e.g. ("Police", "Blocked").
    """
    if weather is None:
        scores = dict(prior_status(road_probabilities))
    else:
        cpt = road_probabilities["road_status_given_weather"][weather]
        scores = {status: cpt[status] for status in STATUSES}

    for source, reported in reports:
        report_cpt = road_probabilities[SOURCE_TO_CPT[source]]
        for status in STATUSES:
            scores[status] *= report_cpt[status][reported]

    return _normalise(scores)


def estimate_road_conditions(
    city_graph: dict[str, Any],
    road_reports: list[dict[str, Any]],
    road_probabilities: dict[str, Any],
) -> dict[RoadKey, Distribution]:
    """
    Estimate P(Open/Congested/Blocked) for every directed road in the graph.

    - Roads with reports: Bayesian update with the reports and their weather.
      If one road has reports with different weather values, the posterior is
      computed per weather and averaged, weighted by how often each occurs.
    - Roads without reports: only the prior P(Status) is used.

    Returns {(from_node, to_node): {"Open": p, "Congested": p, "Blocked": p}}.
    """
    grouped: dict[RoadKey, list[dict[str, Any]]] = defaultdict(list)
    for report in road_reports:
        grouped[(report["from"], report["to"])].append(report)

    default = prior_status(road_probabilities)
    estimates: dict[RoadKey, Distribution] = {}

    for road in city_graph["roads"]:
        key = (str(road["from"]), str(road["to"]))
        rows = grouped.get(key)
        if not rows:
            estimates[key] = dict(default)
            continue

        reports = [(r["source"], r["reported_status"]) for r in rows]
        weather_counts = Counter(r["weather"] for r in rows)

        mixed = {status: 0.0 for status in STATUSES}
        for weather, count in weather_counts.items():
            posterior = posterior_status(weather, reports, road_probabilities)
            for status in STATUSES:
                mixed[status] += posterior[status] * count / len(rows)

        #TODO here u can change the rounding from bayes
        for status in STATUSES:
            mixed[status] = round(mixed[status],3)

        estimates[key] = mixed

    return estimates


class RoadConditionModel:
    """Small wrapper so the route planner can look up a road's distribution."""

    def __init__(self, estimates: dict[RoadKey, Distribution]):
        self._estimates = estimates

    @classmethod
    def from_scenario(cls, data) -> "RoadConditionModel":
        """Build from a ScenarioData object (see data_loader.load_scenario)."""
        return cls(
            estimate_road_conditions(
                data.city_graph, data.road_reports, data.road_probabilities
            )
        )

    def get(self, start: str, end: str) -> Distribution:
        """Return {"Open": p, "Congested": p, "Blocked": p} for the directed road."""
        return self._estimates[(start, end)]

    def p_blocked(self, start: str, end: str) -> float:
        return self._estimates[(start, end)]["Blocked"]

    def all(self) -> dict[RoadKey, Distribution]:
        return dict(self._estimates)