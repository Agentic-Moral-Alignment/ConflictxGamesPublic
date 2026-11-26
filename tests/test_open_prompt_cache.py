"""Tests for the open-mode opening-message cache."""

import json

from conftest import make_spec
from conflictxgame.games.evaluate_games import evaluate_open
from conflictxgame.games.open_ended import (
    cache_key,
    load_open_prompt_cache,
    save_open_prompt_cache,
)
from test_open_mode import RecordingClient, judge_picking, two_by_two


def run_open(spec, user, cache, scenario_id="sid"):
    target = RecordingClient("reply", name="target-model")
    judge = RecordingClient(judge_picking("A"), name="judge-model")
    return evaluate_open(user, target, judge, [(scenario_id, spec)], prompt_cache=cache)


class TestCacheRoundTrip:
    def test_write_then_read(self, tmp_path):
        path = str(tmp_path / "nested" / "open_prompts.json")
        cache = {cache_key("sid", 0): {"user_model": "u", "opening": "héllo — naïve"}}
        save_open_prompt_cache(path, cache)
        assert load_open_prompt_cache(path) == cache

    def test_missing_and_corrupt_files_yield_empty(self, tmp_path):
        assert load_open_prompt_cache(str(tmp_path / "absent.json")) == {}
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_open_prompt_cache(str(bad)) == {}


class TestCacheReuse:
    def test_second_run_issues_no_user_calls(self):
        spec = two_by_two()
        first_user = RecordingClient("canned opening", name="user-model")
        _, cache = run_open(spec, first_user, {})
        assert len(first_user.prompts) == 2
        assert set(cache) == {cache_key("sid", 0), cache_key("sid", 1)}

        second_user = RecordingClient("SHOULD NOT BE CALLED", name="user-model")
        rows, _ = run_open(spec, second_user, cache)
        assert second_user.prompts == []
        assert all(row["open_user_prompt"] == "canned opening" for row in rows)

    def test_different_user_model_regenerates(self):
        spec = two_by_two()
        cache = {
            cache_key("sid", seat): {"user_model": "old-model", "opening": "stale"}
            for seat in range(2)
        }
        user = RecordingClient("fresh opening", name="new-model")
        rows, updated = run_open(spec, user, cache)
        assert len(user.prompts) == 2
        assert all(row["open_user_prompt"] == "fresh opening" for row in rows)
        assert updated[cache_key("sid", 0)]["user_model"] == "new-model"

    def test_no_cache_means_fresh_every_run(self):
        """The paper's default: prompt_cache=None never reuses openings."""
        spec = two_by_two()
        user = RecordingClient("opening", name="user-model")
        rows, cache = evaluate_open(
            user,
            RecordingClient("reply", name="t"),
            RecordingClient(judge_picking("A"), name="j"),
            [("sid", spec)],
            prompt_cache=None,
        )
        assert len(user.prompts) == 2
        assert cache == {}


def test_cached_opening_survives_a_change_of_gateway():
    """The same user model reached through OpenRouter must reuse, not rewrite, the cache."""
    from conflictxgame.games.evaluate_games import _resolve_openings, _same_user_model
    from conftest import make_spec
    from conflictxgame.common.model_wrappers import ModelWrapper

    class Counting(ModelWrapper):
        def __init__(self, name):
            super().__init__(name)
            self.calls = 0

        def generate(self, messages):
            self.calls += 1
            return "fresh opening"

        def batch_generate(self, messages_list, verbose=False):
            return [self.generate(m) for m in messages_list]

        def batch_generate_with_probs(self, messages_list, outputs):
            raise NotImplementedError

    assert _same_user_model("gpt-5.6-luna", "openai/gpt-5.6-luna")
    assert not _same_user_model("gpt-5.6-luna", "openai/gpt-6-luna")
    spec = make_spec("simultaneous", [2, 2], {"0,0": [1, 1], "0,1": [0, 2], "1,0": [2, 0], "1,1": [1, 1]})
    cache = {"sid|0": {"user_model": "gpt-5.6-luna", "opening": "cached opening"}}
    client = Counting("openai/gpt-5.6-luna")
    openings = _resolve_openings(client, [("sid", spec, 0)], cache, True, False)
    assert openings == ["cached opening"] and client.calls == 0
    assert cache["sid|0"]["opening"] == "cached opening"
