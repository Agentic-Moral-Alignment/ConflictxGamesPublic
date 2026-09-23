"""Shared helpers and constants used by both the solo and game pipelines.

This module holds the pipeline-agnostic pieces of the original flat
``utils.py``: robust JSON extraction from model responses and the shared
prompt constants. Value-set loading lives in :mod:`conflictxgame.common.values`;
the dyadic (solo) prompt templates live in :mod:`conflictxgame.solo.prompts`.

Provenance: split out of ConflictScope's ``src/utils.py`` (MIT, (c) 2025 Andy
Liu): https://github.com/andyjliu/conflictscope/blob/main/src/utils.py
See ``ATTRIBUTION.md``.
"""

import json

#: System prompt sent with every generation, filtering, and evaluation query.
SYSTEM_PROMPT = "You are a helpful assistant."

#: Upper bound of the Likert preference scale used by both evaluators.
MAX_LIKERT = 7


def parse_json(response):
    """Extract and parse the first top-level JSON object from a model response.

    Python-style booleans (``True``/``False``) are normalized to JSON before
    parsing, and any text surrounding the outermost ``{...}`` block (markdown
    fences, prose) is ignored.

    Args:
        response: Raw model response text expected to contain a JSON object.

    Returns:
        The parsed object as a dict, or an empty dict when no valid JSON
        object could be extracted (the error is printed, not raised, so a
        single malformed response never aborts a batch).
    """
    response = response.replace("False", "false").replace("True", "true")
    try:
        left = response.index('{')
        right = response.rindex('}') + 1
        return json.loads(response[left:right])
    except (ValueError, json.JSONDecodeError) as e:
        print(f'Error parsing JSON: {e}\nInput given: {response}')
        return {}
