
"""Tests for the cost function and A* route planner.

Run from project root:
    python -m pytest tests/test_route_planner.py -v
"""

import math
import random
from pathlib import Path

import pytest

from src.route_planner import RoutePlanner, RoutePlannerError


OPEN = {
    "Open": 1.0,
    "Congested": 0.0,
    "Blocked": 0.0,
}


class StubRoads:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}

    def get(self, start, end):
        return self.overrides.get((start, end), OPEN)


class StubTraffic:
    def __init__(self, fn=None):
        self.fn = fn
        self.calls = []

    def expected_multiplier(self, start, end, horizon=0.0):
        self.calls.append((start, end, horizon))
        return self.fn(start, end, horizon) if self.fn else 1.0


def both_ways(u, v, distance, speed):
    return [
        {"from": u, "to": v, "distance": distance, "speed": speed},
        {"from": v, "to": u, "distance": distance, "speed": speed},
    ]


def diamond_graph():
    """A-B-C takes 20 s; A-D-C takes 30 s."""

    nodes = [
        {"id": "A", "x": 0, "y": 0},
        {"id": "B", "x": 100, "y": 0},
        {"id": "C", "x": 200, "y": 0},
        {"id": "D", "x": 100, "y": 100},
    ]

    roads = (
        both_ways("A", "B", 100, 10)
        + both_ways("B", "C", 100, 10)
        + both_ways("A", "D", 150, 10)
        + both_ways("D", "C", 150, 10)
    )

    return {"nodes": nodes, "roads": roads}


def planner(roads=None, traffic=None, **kwargs):
    return RoutePlanner(
        diamond_graph(),
        StubRoads(roads),
        traffic or StubTraffic(),
        **kwargs,
    )


def test_shortest_route_and_times():
    result = planner().route("A", "C", departure_time=5.0)

    assert result.path == ["A", "B", "C"]
    assert result.travel_time == pytest.approx(20.0)
    assert result.arrival_time == pytest.approx(25.0)


def test_congested_road_costs_1_5_times():
    roads = {
        ("A", "B"): {
            "Open": 0.0,
            "Congested": 1.0,
            "Blocked": 0.0,
        }
    }

    result = planner(roads).route("A", "C")

    assert result.path == ["A", "B", "C"]
    assert result.travel_time == pytest.approx(25.0)


def test_mixed_open_congested_uses_expected_factor():
    roads = {
        ("A", "B"): {
            "Open": 0.5,
            "Congested": 0.5,
            "Blocked": 0.0,
        }
    }

    assert planner(roads).route("A", "C").travel_time == pytest.approx(22.5)


def test_factor_is_conditioned_on_not_blocked():
    roads = {
        ("A", "B"): {
            "Open": 0.3,
            "Congested": 0.3,
            "Blocked": 0.4,
        }
    }

    assert planner(roads).route("A", "C").travel_time == pytest.approx(22.5)


def test_traffic_multiplier_is_applied():
    traffic = StubTraffic(
        lambda u, v, h: 2.0 if (u, v) == ("B", "C") else 1.0
    )

    assert planner(traffic=traffic).route("A", "C").travel_time == pytest.approx(30.0)


def test_correlated_road_and_traffic_slowdown_use_larger_factor():
    """Use max() rather than multiplying overlapping slowdowns."""

    roads = {
        ("A", "B"): {
            "Open": 0.2,
            "Congested": 0.8,
            "Blocked": 0.0,
        }
    }

    lower_traffic = StubTraffic(lambda u, v, h: 1.2)
    higher_traffic = StubTraffic(lambda u, v, h: 1.45)

    assert planner(
        roads,
        lower_traffic,
    ).route("A", "B").travel_time == pytest.approx(14.0)

    assert planner(
        roads,
        higher_traffic,
    ).route("A", "B").travel_time == pytest.approx(14.5)


def test_traffic_horizon_is_time_since_time_zero():
    traffic = StubTraffic()

    planner(traffic=traffic).route(
        "A",
        "B",
        departure_time=100.0,
    )

    assert ("A", "B", 100.0) in traffic.calls

    traffic2 = StubTraffic()

    planner(traffic=traffic2).route(
        "A",
        "C",
        departure_time=100.0,
    )

    assert ("B", "C", 110.0) in traffic2.calls


def test_blocked_road_is_avoided():
    roads = {
        ("A", "B"): {
            "Open": 0.1,
            "Congested": 0.1,
            "Blocked": 0.8,
        }
    }

    result = planner(roads).route("A", "C")

    assert result.path == ["A", "D", "C"]
    assert result.travel_time == pytest.approx(30.0)
    assert result.threshold_used == 0.5


def test_probability_equal_to_threshold_is_still_usable():
    roads = {
        ("A", "B"): {
            "Open": 0.25,
            "Congested": 0.25,
            "Blocked": 0.5,
        }
    }

    assert planner(roads).route("A", "C").path == ["A", "B", "C"]


def test_fallback_threshold_used_when_otherwise_unreachable():
    roads = {
        ("A", "D"): {
            "Open": 0.2,
            "Congested": 0.1,
            "Blocked": 0.7,
        },
        ("A", "B"): {
            "Open": 0.2,
            "Congested": 0.1,
            "Blocked": 0.7,
        },
    }

    result = planner(roads).route("A", "C")

    assert result is not None
    assert result.threshold_used == 0.7


def test_deadline_fallback_finds_faster_riskier_route():
    roads = {
        ("A", "B"): {
            "Open": 0.4,
            "Congested": 0.0,
            "Blocked": 0.6,
        }
    }

    p = planner(roads)

    default = p.route("A", "C")

    assert default.path == ["A", "D", "C"]
    assert default.arrival_time == pytest.approx(30.0)

    alternative = p.route_before_deadline(
        "A",
        "C",
        departure_time=0,
        deadline=25,
    )

    assert alternative.path == ["A", "B", "C"]
    assert alternative.arrival_time == pytest.approx(20.0)
    assert alternative.threshold_used == pytest.approx(0.6)


def test_deadline_fallback_preserves_normal_route_when_on_time():
    p = planner()
    regular = p.route("A", "C")

    assert p.route_before_deadline(
        "A",
        "C",
        0,
        deadline=25,
    ) == regular


def test_none_when_even_fallback_fails():
    roads = {
        ("A", "D"): {
            "Open": 0.0,
            "Congested": 0.05,
            "Blocked": 0.95,
        },
        ("A", "B"): {
            "Open": 0.0,
            "Congested": 0.05,
            "Blocked": 0.95,
        },
    }

    assert planner(roads).route("A", "C") is None


def test_certainly_blocked_road_is_never_used():
    roads = {
        ("A", "B"): {
            "Open": 0.0,
            "Congested": 0.0,
            "Blocked": 1.0,
        },
        ("A", "D"): {
            "Open": 0.0,
            "Congested": 0.0,
            "Blocked": 1.0,
        },
    }

    p = planner(
        roads,
        block_threshold=1.0,
        fallback_threshold=1.0,
    )

    assert p.route("A", "C") is None


def test_start_equals_goal():
    result = planner().route(
        "A",
        "A",
        departure_time=7.0,
    )

    assert result.path == ["A"]
    assert result.travel_time == 0.0
    assert result.arrival_time == 7.0


def test_unknown_node_raises():
    with pytest.raises(RoutePlannerError, match="Unknown node"):
        planner().route("A", "Z")

    with pytest.raises(RoutePlannerError, match="Unknown node"):
        planner().route("Z", "A")


def test_unreachable_goal_returns_none():
    graph = diamond_graph()

    graph["nodes"].append(
        {"id": "E", "x": 500, "y": 500}
    )

    p = RoutePlanner(
        graph,
        StubRoads(),
        StubTraffic(),
    )

    assert p.route("A", "E") is None


def test_invalid_parameters():
    with pytest.raises(RoutePlannerError):
        planner(heuristic="manhattan")

    with pytest.raises(RoutePlannerError):
        planner(block_threshold=1.5)

    with pytest.raises(RoutePlannerError):
        planner().route("A", "C", departure_time=-1)


def test_directions_are_independent():
    roads = {
        ("A", "B"): {
            "Open": 0.0,
            "Congested": 0.0,
            "Blocked": 1.0,
        }
    }

    p = planner(roads)

    assert p.route("A", "C").path == ["A", "D", "C"]
    assert p.route("C", "A").path == ["C", "B", "A"]


def test_path_travel_time_matches_route_and_rejects_bad_paths():
    p = planner()
    result = p.route("A", "C", 50.0)

    assert p.path_travel_time(
        result.path,
        50.0,
    ) == pytest.approx(result.travel_time)

    with pytest.raises(RoutePlannerError):
        p.path_travel_time(["A", "C"])


def test_heuristic_is_in_seconds_and_zero_for_dijkstra():
    p = planner()

    assert p.heuristic("A", "C") == pytest.approx(200 / 10)
    assert p.heuristic("C", "C") == 0.0
    assert planner(heuristic="zero").heuristic("A", "C") == 0.0


def test_heuristic_scale_shrinks_if_road_shorter_than_straight_line():
    graph = diamond_graph()
    graph["roads"][0]["distance"] = 50

    p = RoutePlanner(
        graph,
        StubRoads(),
        StubTraffic(),
    )

    assert p.heuristic_scale == pytest.approx(0.5)


def random_graph(seed, n=25):
    rng = random.Random(seed)

    nodes = [
        {
            "id": f"N{i}",
            "x": rng.uniform(0, 1000),
            "y": rng.uniform(0, 1000),
        }
        for i in range(n)
    ]

    pairs = set()

    for a in nodes:
        near = sorted(
            nodes,
            key=lambda b: math.dist(
                (a["x"], a["y"]),
                (b["x"], b["y"]),
            ),
        )[1:4]

        for b in near:
            pairs.add(
                tuple(sorted((a["id"], b["id"])))
            )

    by_id = {
        nd["id"]: nd
        for nd in nodes
    }

    roads = []
    overrides = {}
    mult = {}

    for u, v in sorted(pairs):
        straight = math.dist(
            (by_id[u]["x"], by_id[u]["y"]),
            (by_id[v]["x"], by_id[v]["y"]),
        )

        dist = math.ceil(
            straight * rng.uniform(1.0, 1.6)
        ) + 1

        speed = rng.choice([15, 30])

        roads += both_ways(u, v, dist, speed)

        for edge in ((u, v), (v, u)):
            p_block = rng.choice(
                [0.0, 0.0, 0.0, 0.8, 0.95]
            )

            p_open = (1 - p_block) * rng.random()

            overrides[edge] = {
                "Open": p_open,
                "Congested": 1 - p_block - p_open,
                "Blocked": p_block,
            }

            mult[edge] = rng.uniform(1.0, 1.5)

    return {
        "nodes": nodes,
        "roads": roads,
    }, overrides, mult


@pytest.mark.parametrize("seed", range(5))
def test_astar_matches_dijkstra_and_expands_fewer_nodes(seed):
    graph, overrides, mult = random_graph(seed)

    traffic = StubTraffic(
        lambda u, v, h: mult[(u, v)]
    )

    astar = RoutePlanner(
        graph,
        StubRoads(overrides),
        traffic,
    )

    dijkstra = RoutePlanner(
        graph,
        StubRoads(overrides),
        traffic,
        heuristic="zero",
    )

    ids = [n["id"] for n in graph["nodes"]]

    exp_a = 0
    exp_d = 0

    for s in ids[:6]:
        for g in ids:
            a = astar.route(s, g)
            d = dijkstra.route(s, g)

            assert (a is None) == (d is None)

            if a is None:
                continue

            assert a.travel_time == pytest.approx(
                d.travel_time,
                rel=1e-9,
            )

            assert a.threshold_used == d.threshold_used
            assert a.path[0] == s
            assert a.path[-1] == g

            exp_a += a.nodes_expanded
            exp_d += d.nodes_expanded

    assert exp_a <= exp_d


@pytest.mark.parametrize("seed", range(3))
def test_heuristic_never_overestimates(seed):
    graph, overrides, mult = random_graph(seed)

    traffic = StubTraffic(
        lambda u, v, h: mult[(u, v)]
    )

    p = RoutePlanner(
        graph,
        StubRoads(overrides),
        traffic,
        block_threshold=1.0,
    )

    ids = [n["id"] for n in graph["nodes"]]

    for s in ids:
        for g in ids:
            r = p.route(s, g)

            if r is not None:
                assert p.heuristic(s, g) <= r.travel_time + 1e-9


@pytest.mark.parametrize(
    "scenario",
    ["Scenario1", "Scenario2", "Scenario3"],
)
def test_provided_scenarios(scenario):
    path = Path("Scenarios") / scenario

    if not path.is_dir():
        pytest.skip(f"{path} not present")

    from src.data_loader import load_scenario
    from src.road_reasoning import RoadConditionModel
    from src.traffic_model import TrafficModel

    data = load_scenario(path)

    roads = RoadConditionModel.from_scenario(data)
    traffic = TrafficModel.from_scenario(data)

    astar = RoutePlanner(
        data.city_graph,
        roads,
        traffic,
    )

    dijkstra = RoutePlanner(
        data.city_graph,
        roads,
        traffic,
        heuristic="zero",
    )

    starts = {
        r["start_location"]
        for r in data.responders
    }

    goals = {
        i["location"]
        for i in data.incidents
    }

    reachable = 0

    for s in starts:
        for g in goals:
            for departure in (0.0, 600.0):
                a = astar.route(s, g, departure)
                d = dijkstra.route(s, g, departure)

                assert (a is None) == (d is None)

                if a is None:
                    continue

                reachable += 1

                assert a.path[0] == s
                assert a.path[-1] == g

                assert a.arrival_time == pytest.approx(
                    departure + a.travel_time
                )

                assert a.travel_time == pytest.approx(
                    astar.path_travel_time(
                        a.path,
                        departure,
                    )
                )

                assert a.travel_time == pytest.approx(
                    d.travel_time,
                    rel=1e-2,
                )

    assert reachable > 0
