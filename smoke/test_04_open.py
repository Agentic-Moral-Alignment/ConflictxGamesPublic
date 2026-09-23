"""Open-ended evaluation against the live gateway (paper §3.3, App. C.5/C.6).

The mode under test: a user model writes the opening message, the target
answers it with no scenario context, and a judge maps the completion onto the
seat's action menu.
"""

import json

import pandas as pd
import pytest

from conftest import (
    GEN_SHORT,
    JUDGE_MODEL,
    TARGET_MODEL,
    USER_MODEL,
    run_cli,
)

pytestmark = pytest.mark.live

#: Provenance columns record ``ModelWrapper.model_name``, which has the
#: ``openai:`` routing prefix stripped — the same id the gateway sees.
USER_MODEL_RECORDED = USER_MODEL.split(":", 1)[-1]
JUDGE_MODEL_RECORDED = JUDGE_MODEL.split(":", 1)[-1]


@pytest.fixture(scope="module")
def open_run(corpus, data_dir, results_dir):
    """Run --mode open once, with the opening-message cache enabled."""
    result = run_cli(
        "conflictxgame.games.evaluate_games",
        [
            "-m", TARGET_MODEL, "--mode", "open",
            "-d", str(data_dir), "-o", str(results_dir),
            "--user-model", USER_MODEL,
            "--judge-model", JUDGE_MODEL,
            "--open-prompts",
        ],
        stage_index=3,
        timeout=1800,
    )
    assert result.returncode == 0, "open-ended evaluation failed"
    csv_files = list(results_dir.glob("*_open.csv"))
    assert csv_files, f"no *_open.csv written to {results_dir}"
    return pd.read_csv(csv_files[0])


class TestOpenResults:
    def test_rows_produced(self, open_run):
        assert not open_run.empty
        assert set(open_run["mode"]) == {"open"}
        print(f"\nopen rows: {len(open_run)}")

    def test_user_model_wrote_a_real_opening_message(self, open_run):
        prompts = open_run["open_user_prompt"].dropna()
        assert not prompts.empty
        for prompt in prompts:
            assert len(str(prompt).strip()) > 40, f"suspiciously short: {prompt!r}"
        longest = max(prompts, key=len)
        print(f"\nsample opening message:\n{longest[:400]}")

    def test_target_produced_a_completion(self, open_run):
        completions = open_run["open_completion"].dropna()
        assert not completions.empty
        for completion in completions:
            assert len(str(completion).strip()) > 20
        print(f"\nsample completion:\n{max(completions, key=len)[:400]}")

    def test_judge_classified_completions_onto_the_menu(self, open_run):
        parsed = open_run["parsed_ok"].astype(str).str.lower().eq("true")
        rate = parsed.mean()
        print(f"\nopen judge parse rate: {rate:.0%}")
        assert rate > 0.5, "judge rarely returned a usable action letter"
        valid = open_run[open_run["choice_action"] != "INVALID"]
        assert not valid.empty
        # Choices index real actions of a 2-action menu.
        assert valid["choice_action"].astype(int).isin([0, 1]).all()
        assert valid["choice_value_tag"].notna().all()

    def test_binary_menus_get_the_appendix_c6_likert(self, open_run):
        likerts = open_run["open_judge_likert"].dropna()
        numeric = pd.to_numeric(likerts, errors="coerce").dropna()
        assert not numeric.empty, "no Likert alignment rating returned"
        assert numeric.between(1, 7).all()

    def test_provenance_columns_recorded(self, open_run):
        assert set(open_run["user_model"].dropna()) == {USER_MODEL_RECORDED}
        assert set(open_run["judge_model"].dropna()) == {JUDGE_MODEL_RECORDED}


class TestOpenPromptCache:
    def test_cache_file_written(self, open_run, data_dir):
        cache_path = data_dir / f"{GEN_SHORT}_open_prompts.json"
        assert cache_path.exists(), f"expected cache at {cache_path}"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cache, "cache is empty"
        for key, entry in cache.items():
            assert "|" in key, f"unexpected cache key {key!r}"
            assert entry["user_model"] == USER_MODEL_RECORDED
            assert entry["opening"].strip()
        print(f"\ncached openings: {len(cache)}")

    def test_second_run_reuses_cached_openings(self, open_run, data_dir, results_dir):
        """Re-running with --force-recompute must not re-ask the user model.

        Same openings in, same prompts out — that is what makes several target
        models comparable on identical inputs.
        """
        cache_path = data_dir / f"{GEN_SHORT}_open_prompts.json"
        before = json.loads(cache_path.read_text(encoding="utf-8"))

        result = run_cli(
            "conflictxgame.games.evaluate_games",
            [
                "-m", TARGET_MODEL, "--mode", "open",
                "-d", str(data_dir), "-o", str(results_dir),
                "--output-name", "cache_reuse_open.csv",
                "--user-model", USER_MODEL,
                "--judge-model", JUDGE_MODEL,
                "--open-prompts",
                # Without this, resume dedup would skip every scenario on a
                # re-run and the test would pass without exercising anything.
                "--force-recompute",
            ],
            stage_index=4,
            timeout=1800,
        )
        assert result.returncode == 0
        assert "0 new decision rows" not in result.stdout, (
            "the re-run evaluated nothing, so cache reuse was never exercised"
        )

        after = json.loads(cache_path.read_text(encoding="utf-8"))
        assert set(after) == set(before), "cache keys changed on reuse"
        for key in before:
            assert after[key]["opening"] == before[key]["opening"], (
                f"opening for {key} was regenerated instead of reused"
            )

        rerun = pd.read_csv(results_dir / "cache_reuse_open.csv")
        reused = set(rerun["open_user_prompt"].dropna())
        original = set(open_run["open_user_prompt"].dropna())
        assert reused <= original, "second run used prompts absent from the first"
