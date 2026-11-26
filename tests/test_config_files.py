"""Tests for YAML/JSONC run configurations and shipped config files.

Two concerns. First, that the JSONC dialect is parsed correctly -- a comment
stripper that is not string-aware will corrupt a URL or a prompt fragment that
happens to contain ``//``, and it will do it silently. Second, that every file
under ``configs/`` still parses against the parser it was written for: a
renamed flag turns a config into an unknown-key error at launch, which is
worth catching offline rather than after paid credentials have loaded.
"""

import json
from pathlib import Path

import pytest

from conflictxgame.common.config import (
    add_config_argument,
    load_config,
    load_config_defaults,
    parse_with_config,
    strip_jsonc,
)
from conflictxgame.games import evaluate_games, generate_balanced
from conflictxgame.solo import (
    evaluate_models,
    filter_scenarios,
    generate_scenarios,
    topup_scenarios,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


# -- JSONC dialect ---------------------------------------------------------- #

class TestStripJsonc:
    """The comment stripper must not touch string contents or line numbers."""

    def test_line_and_block_comments_are_removed(self):
        text = """{
          // leading comment
          "a": 1, // trailing comment
          /* a block
             comment */
          "b": 2
        }"""
        assert json.loads(strip_jsonc(text)) == {"a": 1, "b": 2}

    def test_comment_markers_inside_strings_survive(self):
        text = '{"url": "https://x/y", "note": "keep // and /* this */"}'
        assert json.loads(strip_jsonc(text)) == {
            "url": "https://x/y",
            "note": "keep // and /* this */",
        }

    def test_escaped_quote_does_not_end_the_string(self):
        text = r'{"a": "he said \"//\" here", "b": 1}'
        assert json.loads(strip_jsonc(text)) == {"a": 'he said "//" here', "b": 1}

    @pytest.mark.parametrize("text, expected", [
        ('{"a": [1, 2, 3,],}', {"a": [1, 2, 3]}),
        ('{"a": 1,\n  // note\n}', {"a": 1}),
        ('{"a": {"b": 1,},}', {"a": {"b": 1}}),
    ])
    def test_trailing_commas_are_dropped(self, text, expected):
        assert json.loads(strip_jsonc(text)) == expected

    def test_trailing_comma_inside_a_string_is_kept(self):
        text = '{"a": "one, two,", "b": 1}'
        assert json.loads(strip_jsonc(text)) == {"a": "one, two,", "b": 1}

    def test_line_numbers_are_preserved_for_error_reporting(self):
        text = '{\n  // comment\n  /* two\n     lines */\n  "a": 1\n}'
        assert strip_jsonc(text).count("\n") == text.count("\n")

    def test_plain_json_is_unchanged(self):
        text = '{"a": 1, "b": [2, 3], "c": "d"}'
        assert strip_jsonc(text) == text

    def test_unterminated_block_comment_does_not_hang(self):
        assert json.loads(strip_jsonc('{"a": 1}\n/* trailing')) == {"a": 1}


# -- Strict validation ------------------------------------------------------ #

def _sample_parser():
    """A parser exercising each coercion branch."""
    import argparse

    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--mode', choices=('a', 'b'), default='a')
    parser.add_argument('--flag', action='store_true')
    return parser


def _write(tmp_path, payload, name="run.jsonc"):
    path = tmp_path / name
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload),
        encoding="utf-8",
    )
    return str(path)


class TestStrictValidation:
    """A bad config must abort before any client is constructed."""

    @pytest.mark.parametrize("payload", [
        ["not", "an", "object"],
        {"nonexistent_key": 1},
        {"count": "3"},          # string where an int belongs
        {"count": True},         # bool is not an int here
        {"ratio": "0.5"},
        {"flag": "true"},        # string where a bool belongs
        {"mode": "c"},           # outside choices
        {"name": None},          # required may not be null
    ])
    def test_rejected(self, tmp_path, payload):
        parser = _sample_parser()
        with pytest.raises(SystemExit):
            load_config_defaults(parser, _write(tmp_path, payload))

    def test_unreadable_file_is_rejected(self, tmp_path):
        parser = _sample_parser()
        with pytest.raises(SystemExit):
            load_config_defaults(parser, str(tmp_path / "missing.jsonc"))

    def test_malformed_jsonc_is_rejected(self, tmp_path):
        parser = _sample_parser()
        with pytest.raises(SystemExit):
            load_config_defaults(parser, _write(tmp_path, '{"count": }'))

    def test_int_is_widened_for_a_float_option(self, tmp_path):
        parser = _sample_parser()
        loaded = load_config_defaults(parser, _write(tmp_path, {"ratio": 1}))
        assert loaded["ratio"] == 1.0
        assert isinstance(loaded["ratio"], float)


class TestParseWithConfig:
    """Config supplies the run; the command line supplies the deviation."""

    def test_config_satisfies_a_required_option(self, tmp_path):
        path = _write(tmp_path, {"name": "from-config", "count": 9})
        args, defaults, explicit = parse_with_config(
            _sample_parser(), ["--config", path]
        )
        assert (args.name, args.count) == ("from-config", 9)
        assert defaults == {"name": "from-config", "count": 9}
        assert explicit == {"config"}

    def test_cli_overrides_the_config(self, tmp_path):
        path = _write(tmp_path, {"name": "from-config", "count": 9})
        args, _, explicit = parse_with_config(
            _sample_parser(), ["--config", path, "--count", "1"]
        )
        assert (args.name, args.count) == ("from-config", 1)
        assert "count" in explicit

    def test_missing_required_names_both_the_flag_and_the_key(self, capsys):
        with pytest.raises(SystemExit):
            parse_with_config(_sample_parser(), [])
        message = capsys.readouterr().err
        assert "--name" in message and '"name"' in message

    def test_comments_and_trailing_commas_are_accepted(self, tmp_path):
        path = _write(tmp_path, '{\n  // the run\n  "name": "x",\n}')
        args, _, _ = parse_with_config(_sample_parser(), ["--config", path])
        assert args.name == "x"

    def test_yaml_comments_are_accepted(self, tmp_path):
        path = _write(
            tmp_path,
            "# the run\nname: from-yaml\ncount: 9\nflag: true\n",
            name="run.yaml",
        )
        args, defaults, _ = parse_with_config(
            _sample_parser(), ["--config", path]
        )
        assert (args.name, args.count, args.flag) == ("from-yaml", 9, True)
        assert defaults["name"] == "from-yaml"

    def test_unsupported_suffix_is_rejected(self, tmp_path):
        path = _write(tmp_path, "name: x\n", name="run.toml")
        with pytest.raises(ValueError):
            load_config(path)


# -- The shipped configs ---------------------------------------------------- #

def _entry_point(path):
    """The parser a shipped config targets, and the arguments it leaves to the command line."""
    name = path.name
    if path.parent.name == "games":
        return (evaluate_games if "_eval-" in name else generate_balanced).parse_args, []
    if "_eval-" in name:
        return evaluate_models.parse_args, []
    if "topup-to-50" in name:
        return topup_scenarios.parse_args, []
    if "filter" in name:
        return filter_scenarios.parse_args, ["-i", "unused.json"]
    return generate_scenarios.parse_args, ["-v1", "harmlessness", "-v2", "honesty"]


SHIPPED_CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))


class TestShippedConfigs:
    """Guards the configs against flag renames and typos."""

    def test_configs_exist(self):
        assert SHIPPED_CONFIGS

    @pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda p: p.stem)
    def test_parses(self, path):
        parse, extra = _entry_point(path)
        parse(["--config", str(path)] + extra)

    def test_every_game_evaluation_has_its_own_released_output(self):
        parsed = [
            evaluate_games.parse_args(["--config", str(p)])
            for p in SHIPPED_CONFIGS
            if p.parent.name == "games" and "_eval-" in p.name
        ]
        outputs = [(a.output_dir, a.output_name) for a in parsed]
        assert len(set(outputs)) == len(outputs)
        released = {
            (d, p.name)
            for d in {directory for directory, _ in outputs}
            for p in (REPO_ROOT / d).iterdir()
        }
        assert set(outputs) <= released
