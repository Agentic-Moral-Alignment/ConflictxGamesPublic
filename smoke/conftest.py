"""Shared fixtures for the live smoke suite.

These tests call the **real** institutional gateway, unlike the offline suite
in ``tests/``. They are excluded from the default pytest run (``testpaths =
["tests"]`` in ``pyproject.toml``) and skip themselves entirely when no API
key is configured, so they can never fail a normal ``pytest`` invocation.

Credentials are read from a gitignored ``.env`` at the repository root, and
fall back to whatever is already exported in the environment (e.g. a key
stored with ``conda env config vars set``). See ``smoke/README.md``.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

#: Repository root (this file lives in <root>/smoke/).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where every artifact this suite produces is written. Gitignored, and kept
#: out of ``data/`` so generated smoke data never mixes with the real corpora.
ARTIFACTS = REPO_ROOT / "smoke" / "_artifacts"

#: Value set used throughout; small (3 values) and already shipped.
VALUE_SET = "HHH"

#: Gateway-served model ids. The ``openai:`` prefix routes them through the
#: OpenAI-compatible client instead of being matched by substring.
GEN_MODEL = "openai:anthropic.claude-sonnet-5"
TARGET_MODEL = "openai:anthropic.claude-sonnet-5"
JUDGE_MODEL = "openai:openai.gpt-5.6-luna"
USER_MODEL = "openai:openai.gpt-5.6-luna"

#: ``short_model_name`` of GEN_MODEL — the stem of every generated filename.
GEN_SHORT = "anthropic.claude-sonnet-5"


def load_dotenv(path: Path = None) -> dict:
    """Parse a ``.env`` file into a dict (no external dependency).

    Blank lines and ``#`` comments are skipped; ``export FOO=bar`` and quoted
    values are both accepted. Returns an empty dict when the file is absent.
    """
    path = path or (REPO_ROOT / ".env")
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            values[key.strip()] = value
    return values


def api_keys() -> list:
    """Return the institutional gateway keys to spread across pipeline stages.

    Resolution order: ``GATEWAY_API_KEY_1..9`` then ``GATEWAY_API_KEY``
    then ``OPENAI_API_KEY``, looking first in ``.env`` and then in the
    process environment (so an existing conda-stored key keeps working).
    Duplicates are collapsed, order preserved.
    """
    dotenv = load_dotenv()

    def lookup(name):
        return dotenv.get(name) or os.environ.get(name)

    candidates = [lookup(f"GATEWAY_API_KEY_{i}") for i in range(1, 10)]
    candidates += [lookup("GATEWAY_API_KEY"), lookup("OPENAI_API_KEY")]
    keys = []
    for key in candidates:
        if key and key not in keys and not key.startswith("sk-your-"):
            keys.append(key)
    return keys


def base_url() -> str:
    """Return the gateway base URL (``.env`` first, then the environment)."""
    dotenv = load_dotenv()
    return (
        dotenv.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://gateway.example.edu/api"
    )


def stage_env(stage_index: int) -> dict:
    """Build the environment for one pipeline stage, with a rotated API key.

    Each stage gets ``api_keys()[stage_index % n]`` so a rate limit on one key
    does not stall the whole run. The gateway base URL is always set.

    Args:
        stage_index: Monotonically increasing index of the pipeline stage.

    Returns:
        A copy of ``os.environ`` with ``OPENAI_API_KEY``/``OPENAI_BASE_URL``
        set for that stage.
    """
    keys = api_keys()
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = keys[stage_index % len(keys)]
    env["OPENAI_BASE_URL"] = base_url()
    # Keep child processes from importing a half-written cache of our own.
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_cli(module: str, args: list, stage_index: int, timeout: int = 900):
    """Run one pipeline CLI as a subprocess and return the completed process.

    Invoking the real ``python -m conflictxgame...`` entry points (rather than
    calling functions) is deliberate: it smoke-tests argument parsing, default
    output paths, and file writing the way a user would hit them.

    Args:
        module: Dotted module path, e.g. ``conflictxgame.games.evaluate_games``.
        args: CLI arguments.
        stage_index: Which API key to use (see :func:`stage_env`).
        timeout: Seconds before the stage is killed.

    Returns:
        The :class:`subprocess.CompletedProcess`, with stdout/stderr captured.
    """
    command = [sys.executable, "-m", module, *args]
    print(f"\n$ {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=stage_env(stage_index),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        print("--- stderr ---")
        print((result.stderr or "")[-4000:])
    return result


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #

def pytest_configure(config):
    config.addinivalue_line("markers", "live: hits the real API gateway")


@pytest.fixture(scope="session", autouse=True)
def require_api_key():
    """Skip the entire suite when no usable API key is configured."""
    if not api_keys():
        pytest.skip(
            "No gateway API key found. Copy .env.example to .env and fill in "
            "GATEWAY_API_KEY_1 (see smoke/README.md).",
            allow_module_level=True,
        )


@pytest.fixture(scope="session")
def artifacts() -> Path:
    """The artifacts directory, created on first use."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS


@pytest.fixture(scope="session")
def data_dir(artifacts) -> Path:
    """Directory holding generated scenarios (mirrors ``data/<value_set>/``)."""
    path = artifacts / "data" / VALUE_SET
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def results_dir(artifacts) -> Path:
    """Directory holding unsteered evaluation results."""
    path = artifacts / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def steered_results_dir(artifacts) -> Path:
    """Directory holding steered evaluation results (kept separate on purpose).

    Resume dedup keys on (scenario_id, mode), so a steered run must not share
    an output file with the unsteered one or it would be skipped entirely.
    """
    path = artifacts / "results_steered"
    path.mkdir(parents=True, exist_ok=True)
    return path


#: Number of games the smoke corpus generates. The LLM judge legitimately
#: rejects a good share of scenarios (ambiguity and value-guidedness), so a
#: 2-scenario corpus can easily end up empty; 3 keeps the run short while
#: making a survivor likely.
SMOKE_SCENARIOS = 3


@pytest.fixture(scope="session")
def corpus(data_dir) -> Path:
    """Generate and filter a tiny game corpus once, reused by later tests.

    Returns the path of the filtered ``*_games.csv``. If a previous run
    already produced it, generation is skipped entirely — so re-running the
    suite while iterating costs only the evaluation calls. Delete
    ``smoke/_artifacts/`` to force a fresh corpus.
    """
    filtered_csv = data_dir / f"{GEN_SHORT}_games.csv"
    if filtered_csv.exists():
        print(f"\nReusing existing smoke corpus: {filtered_csv}")
        return filtered_csv

    games_dir = data_dir / "games"
    generated = run_cli(
        "conflictxgame.games.generate_game_scenarios",
        [
            "-m", GEN_MODEL, "-v", VALUE_SET,
            "-o", str(games_dir),
            "--assistant-seat-prob", "1.0",
            "--max-per-config", str(SMOKE_SCENARIOS),
            "--batch-size", str(SMOKE_SCENARIOS),
            "--max-tokens", "8192",
        ],
        stage_index=0,
        timeout=1800,
    )
    assert generated.returncode == 0, "generation CLI failed"

    # Deliberately uses the shipped --guidedness-votes default (3). Dropping to
    # 1 to save calls makes the check as noisy as the code's own docstring
    # warns, and vetoes scenarios at random.
    filtered = run_cli(
        "conflictxgame.games.filter_game_scenarios",
        [
            "-i", str(games_dir / f"{GEN_SHORT}_games.json"),
            "-o", str(data_dir),
            "-m", JUDGE_MODEL, "-v", VALUE_SET,
            "--coverage-report", str(games_dir / "coverage_report.json"),
        ],
        stage_index=1,
        timeout=1800,
    )
    assert filtered.returncode == 0, "filter CLI failed"
    assert filtered_csv.exists(), f"filter produced no {filtered_csv}"
    return filtered_csv


@pytest.fixture(scope="session")
def kept_scenarios(corpus) -> int:
    """How many scenarios survived filtering.

    Evaluation tests run **unfiltered** so they always have data: whether the
    judge keeps a scenario is a data-quality question, not a code-health one,
    and a smoke suite that fails whenever a stochastic judge rejects a small
    corpus reports noise instead of breakage. This count gates only the one
    test that specifically exercises ``--filter``.
    """
    import pandas as pd

    return int(pd.read_csv(corpus)["keep_scenario"].sum())
