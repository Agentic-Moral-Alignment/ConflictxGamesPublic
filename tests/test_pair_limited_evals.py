import pandas as pd

from conftest import make_spec
from conflictxgame.games.evaluate_games import build_play_setups, select_game_seats
from conflictxgame.solo.evaluate_models import load_scenarios_from_csv
from test_game_engine import CannedClient, first_option_responder


PAYOFFS = {"0,0": [1, 1], "0,1": [0, 2], "1,0": [2, 0], "1,1": [1, 1]}


def test_solo_pair_limit_selects_exact_equal_counts_deterministically(tmp_path):
    rows = []
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    for pair in pairs:
        for index in range(7):
            rows.append({
                "scenario_id": f"{pair[0]}-{pair[1]}-{index}",
                "value1": pair[0],
                "value2": pair[1],
                "keep_scenario": True,
            })
    path = tmp_path / "solo.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    left = load_scenarios_from_csv(path, True, pair_limit=5, seed=9)
    right = load_scenarios_from_csv(path, True, pair_limit=5, seed=9)
    assert len(left) == 15
    assert left["scenario_id"].tolist() == right["scenario_id"].tolist()
    counts = left.groupby(left.apply(
        lambda row: "|".join(sorted((row["value1"], row["value2"]))), axis=1
    )).size()
    assert set(counts) == {5}


def test_game_pair_limit_selects_exact_number_of_scenario_seats_per_pair():
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    specs = []
    for index in range(6):
        first = pairs[index % 3]
        second = pairs[(index + 1) % 3]
        spec = make_spec(
            "simultaneous", [2, 2], PAYOFFS, game_id=f"g{index}",
            tags=[[first[0], first[1]], [second[0], second[1]]],
        )
        specs.append((spec.game_id, spec, None))

    selected = select_game_seats(specs, pair_limit=3, seed=4)
    assert len(selected) == 9
    counts = {pair: 0 for pair in pairs}
    by_id = {sid: spec for sid, spec, _ in specs}
    for scenario_id, seat in selected:
        pair = tuple(sorted(by_id[scenario_id].players[seat].value_pair))
        counts[pair] += 1
    assert set(counts.values()) == {3}


def test_scripted_play_builds_only_selected_focal_seat():
    spec = make_spec(
        "simultaneous", [2, 2], PAYOFFS, game_id="g",
        tags=[["a", "b"], ["a", "c"]],
    )
    client = CannedClient(first_option_responder)
    setups = build_play_setups(
        "g", spec, client, "scripted", ["always_cooperate"], [], 2, None,
        selected_seats={("g", 1)},
    )
    assert len(setups) == 1
    assert setups[0].metadata["focal_seats"] == [1]
