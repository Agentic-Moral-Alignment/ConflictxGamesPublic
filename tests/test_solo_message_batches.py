"""Solo HHH Message Batch wiring: keys, pair gating, and batched stage one."""

import json
import re

import pytest

from conflictxgame.solo.generate_scenarios import (
    HHH_PAIRS,
    _generation_pairs,
    _hhh_pairs_or_exit,
    _next_scenario_index,
    _recover_stage_one_scenarios,
    generate_stage_one,
    parse_args,
)


def test_hhh_pairs_are_exactly_the_three_paper_conflicts():
    assert HHH_PAIRS == (
        ("harmlessness", "helpfulness"),
        ("harmlessness", "honesty"),
        ("helpfulness", "honesty"),
    )


def test_all_hhh_pairs_rejects_a_non_hhh_value_set():
    with pytest.raises(SystemExit):
        _hhh_pairs_or_exit({"helpfulness": "x", "honesty": "y", "privacy": "z"})


def test_all_hhh_pairs_cannot_be_combined_with_a_single_pair(tmp_path):
    args = parse_args([
        "--value-set", "HHH_new",
        "--output-dir", str(tmp_path),
        "--all-hhh-pairs",
        "-v1", "harmlessness",
        "-v2", "honesty",
    ])
    with pytest.raises(SystemExit):
        _generation_pairs(
            args,
            {
                "harmlessness": "a",
                "helpfulness": "b",
                "honesty": "c",
            },
        )


def test_generation_pairs_for_hhh_new():
    args = parse_args([
        "--value-set", "HHH_new",
        "--output-dir", "unused",
        "--all-hhh-pairs",
    ])
    pairs = _generation_pairs(
        args,
        {
            "harmlessness": "a",
            "helpfulness": "b",
            "honesty": "c",
        },
    )
    assert pairs == HHH_PAIRS


def test_stage_one_uses_batch_generate_when_a_batch_key_is_set():
    captured = {"keys": [], "n": 0}

    class Recorder:
        def generate(self, messages):
            raise AssertionError("stage one should not call generate() in batch mode")

        def batch_generate(self, messages_list, verbose=False, batch_key=None):
            captured["keys"].append(batch_key)
            captured["n"] = len(messages_list)
            captured["prompt"] = messages_list[0][-1]["content"]
            return ["{}"] * len(messages_list)

    generate_stage_one(
        "harmlessness", "helpfulness", Recorder(), 4, 0,
        {"harmlessness": "no harm", "helpfulness": "be helpful"},
        batch_key="solo-test-stage1",
        seed=0,
    )
    assert captured["keys"] == ["solo-test-stage1"]
    assert captured["n"] == 4
    assert "harmlessness" in captured["prompt"]


def test_seeded_remainder_assignment_is_stable():
    class Capture:
        def __init__(self):
            self.counts = None

        def batch_generate(self, messages_list, verbose=False, batch_key=None):
            self.counts = tuple(
                int(re.search(r"Generate (\d+) scenarios", m[-1]["content"]).group(1))
                for m in messages_list
            )
            return ["{}"] * len(messages_list)

    first, second = Capture(), Capture()
    values = {"harmlessness": "h", "honesty": "o"}
    generate_stage_one(
        "harmlessness", "honesty", first, 5, 0, values, batch_key="k", seed=0,
    )
    generate_stage_one(
        "harmlessness", "honesty", second, 5, 0, values, batch_key="k", seed=0,
    )
    assert first.counts == second.counts
    assert sum(first.counts) == 5


def test_resume_index_skips_existing_suffixes_when_the_corpus_has_a_gap():
    scenarios = {
        "helpfulness-honesty-cc_mild-0": {},
        "helpfulness-honesty-cc_mild-1": {},
        "helpfulness-honesty-cc-49": {},
    }

    assert _next_scenario_index(scenarios) == 50


def test_recovers_valid_stage_one_results_and_skips_flat_malformed_output(tmp_path):
    valid = {
        "helpfulness-honesty-cc-1": {
            "context": "c",
            "action_opportunity": "a",
            "necessary_strong_harm": "h",
            "strong_good": "g",
        }
    }
    flat = {
        "scenario_id": "helpfulness-honesty-cc-1",
        "context": "not nested",
        "action_opportunity": "not nested",
    }
    for index, payload in enumerate((valid, flat)):
        (tmp_path / f"{index}.json").write_text(json.dumps({
            "batch_key": f"solo-x-helpfulness-honesty-stage1-{index}",
            "status": "ended",
            "results": [{
                "type": "succeeded",
                "text": json.dumps(payload),
            }],
        }), encoding="utf-8")

    recovered = _recover_stage_one_scenarios(
        str(tmp_path), "helpfulness", "honesty", 50,
        {"helpfulness": "help", "honesty": "truth"}, {},
    )

    assert list(recovered) == ["helpfulness-honesty-cc-50"]
    assert recovered["helpfulness-honesty-cc-50"]["context"] == "c"
