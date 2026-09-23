"""Value-set loading and per-agent-kind definition resolution.

Value sets are JSON files in the repository's ``value_sets/`` directory,
mapping value names to definition strings (or, for legacy files, to per-kind
dicts — see :func:`resolve_value_definition`). :func:`find_value_sets_dir`
locates the directory whether a CLI is run from the repository root or the
package is used from an arbitrary working directory under an editable
install.

Provenance: split out of ConflictScope's ``src/utils.py`` (MIT, (c) 2025 Andy
Liu) and extended with per-agent-kind resolution: https://github.com/andyjliu/conflictscope/blob/main/src/utils.py
See ``ATTRIBUTION.md``.
"""

import json
from pathlib import Path

#: Recognized decision-maker kinds for per-kind value definitions.
AGENT_KINDS = ("human", "assistant")


def find_value_sets_dir() -> Path:
    """Locate the ``value_sets/`` directory.

    Resolution order: ``value_sets/`` under the current working directory
    (preserving the historical run-from-repo-root behavior), then
    ``value_sets/`` at the repository root located relative to this file
    (the editable src-layout install).

    Returns:
        The path of the first existing candidate.

    Raises:
        FileNotFoundError: If neither candidate directory exists.
    """
    cwd_candidate = Path.cwd() / "value_sets"
    if cwd_candidate.is_dir():
        return cwd_candidate
    # src-layout: .../repo/src/conflictxgame/common/values.py -> parents[3] = repo
    repo_candidate = Path(__file__).resolve().parents[3] / "value_sets"
    if repo_candidate.is_dir():
        return repo_candidate
    raise FileNotFoundError(
        "No value_sets/ directory found (run from the repository root, or keep "
        "the package installed in editable mode next to value_sets/)"
    )


def load_value_set(json_name: str) -> dict:
    """Load a value set for the game pipeline (fails loudly on a bad name).

    All shipped value sets are flat ``name -> definition`` mappings with one
    agent-neutral definition per value, used for every decision-maker kind.
    For backward compatibility with legacy files, an entry may also be a dict
    mapping agent kinds (see :data:`AGENT_KINDS`) to definitions, e.g.
    ``{"human": "...", "assistant": "..."}``; resolve an entry with
    :func:`resolve_value_definition` (flat strings pass through unchanged).

    Args:
        json_name: Value-set name; loads ``value_sets/{json_name}.json``.

    Returns:
        Mapping from value name to its raw entry (string or per-kind dict).

    Raises:
        FileNotFoundError / json.JSONDecodeError: If the file is missing or
            malformed (no silent fallback — the game pipeline should fail
            loudly on a bad value-set name).
    """
    json_path = find_value_sets_dir() / f"{json_name}.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_value_definition(entry, agent_kind: str = "human") -> str:
    """Resolve a value-set entry to the definition for one agent kind.

    Resolution order for dict entries: the exact ``agent_kind`` key, then
    ``"human"``, then the first entry (stable fallback for single-kind dicts).
    String entries apply to every kind unchanged.

    Args:
        entry: A value-set entry (string, or dict of kind -> definition).
        agent_kind: One of :data:`AGENT_KINDS`.

    Returns:
        The definition string.

    Raises:
        KeyError: If ``entry`` is an empty dict.
        TypeError: If ``entry`` is neither a string nor a dict.
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        if agent_kind in entry:
            return entry[agent_kind]
        if "human" in entry:
            return entry["human"]
        for definition in entry.values():
            return definition
        raise KeyError("Empty value-definition entry")
    raise TypeError(f"Value-set entry must be str or dict, got {type(entry).__name__}")


def resolve_value_dict(raw_value_set: dict, agent_kind: str = "human") -> dict:
    """Flatten a raw value set to plain definitions for one agent kind.

    Args:
        raw_value_set: Mapping from value name to raw entry (string or
            per-kind dict).
        agent_kind: One of :data:`AGENT_KINDS`.

    Returns:
        Mapping from value name to the resolved definition string.
    """
    return {
        name: resolve_value_definition(entry, agent_kind)
        for name, entry in raw_value_set.items()
    }
