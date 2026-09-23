"""Opt-in capture of reasoning traces returned alongside chat completions.

Reasoning models served through OpenRouter answer with two fields, not one:
``message.content`` holds the reply and ``message.reasoning`` holds the chain
of thought that produced it. :func:`conflictxgame.common.model_wrappers.
completion_text` reads only the former, so the reasoning has been discarded on
every run to date. Anything that needs to inspect *how* a model reached its
answer -- eval awareness being the motivating case -- has nothing to read.

Capture is off unless ``CONFLICTXGAME_CAPTURE_REASONING`` names a file. That
default matters: an evaluation sweep issues tens of thousands of calls, the
traces are far larger than the answers, and a run that silently began writing
them would change the disk footprint of every existing pipeline. Turning it on
is therefore a deliberate act, and the same run is reproducible with it off.

Records are appended as JSON Lines so a killed run leaves a readable prefix
rather than a truncated document, and so a capture can be tailed while the
sweep is still going. Writes are serialised through a lock because
``batch_generate`` dispatches through a thread pool.

Example:
    CONFLICTXGAME_CAPTURE_REASONING=/tmp/reasoning.jsonl \\
        python -m conflictxgame.solo.evaluate_models --config ...
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

#: Environment variable naming the capture file. Unset means capture is off.
CAPTURE_ENV_VAR = 'CONFLICTXGAME_CAPTURE_REASONING'

#: Serialises appends from ``batch_generate``'s worker threads.
_WRITE_LOCK = threading.Lock()


def capture_path() -> Optional[str]:
    """The configured capture file, or ``None`` when capture is disabled.

    Read on every call rather than cached at import so a long-lived process
    picks up a change, and so tests can toggle it with ``monkeypatch.setenv``.

    Returns:
        The path from :data:`CAPTURE_ENV_VAR`, or ``None`` if it is unset or
        empty.
    """
    value = os.environ.get(CAPTURE_ENV_VAR, '').strip()
    return value or None


def extract_reasoning(response: Any) -> Optional[str]:
    """Pull the reasoning trace out of a chat-completion response.

    Providers disagree on the field. OpenRouter populates ``reasoning`` with a
    plain string and ``reasoning_details`` with a list of typed blocks; some
    gateways send only the latter. Both are read, the plain field first, and
    the blocks are joined when it is absent.

    Args:
        response: Whatever the SDK returned. Any shape that is not a
            recognisable completion yields ``None`` rather than raising --
            capture must never be able to fail a run.

    Returns:
        The reasoning text, or ``None`` when the response carries none.
    """
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, KeyError, TypeError):
        return None

    reasoning = getattr(message, 'reasoning', None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    details = getattr(message, 'reasoning_details', None)
    if isinstance(details, list):
        parts: List[str] = []
        for block in details:
            text = (
                block.get('text') if isinstance(block, dict)
                else getattr(block, 'text', None)
            )
            if isinstance(text, str) and text.strip():
                parts.append(text)
        if parts:
            return '\n'.join(parts)
    return None


def _content_of(response: Any) -> Optional[str]:
    """The assistant text, or ``None`` if the response carries none.

    Unlike :func:`~conflictxgame.common.model_wrappers.completion_text` this
    does not raise on empty content. An empty answer beside a long reasoning
    trace is the single most informative record a capture can hold -- it is a
    scenario the harness dropped -- so it must survive to the file.
    """
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _prompt_text(messages: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Flatten a conversation into one string for the record.

    Returns:
        Role-tagged turns joined by blank lines, or ``None`` when there is no
        usable conversation.
    """
    if not messages:
        return None
    turns = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get('role', '?'))
            content = message.get('content', '')
        else:
            role = str(getattr(message, 'role', '?'))
            content = getattr(message, 'content', '')
        if isinstance(content, str) and content:
            turns.append(f'{role}: {content}')
    return '\n\n'.join(turns) or None


def capture_response(
    model_name: str,
    messages: Optional[List[Dict[str, Any]]],
    response: Any,
    *,
    scenario_id: Optional[str] = None,
    condition: Optional[str] = None,
) -> bool:
    """Append one call's reasoning trace to the capture file.

    A no-op when capture is disabled, and a no-op when the response carries no
    reasoning -- a non-reasoning model would otherwise fill the file with
    records whose only content duplicates the results CSV.

    Every failure is swallowed. This runs inside the request path of a paid
    evaluation; a full disk or an unwritable directory must degrade to "no
    capture", never to a lost run.

    Args:
        model_name: Model id the request was issued against.
        messages: The conversation sent, used to record the prompt.
        response: The raw completion object.
        scenario_id: Optional identifier tying the record to a scenario row.
        condition: Optional arm label (e.g. ``"mcq-steer"``).

    Returns:
        ``True`` if a record was written, ``False`` otherwise.
    """
    path = capture_path()
    if path is None:
        return False

    reasoning = extract_reasoning(response)
    if reasoning is None:
        return False

    content = _content_of(response)
    record = {
        'timestamp': time.time(),
        'model': model_name,
        'scenario_id': scenario_id,
        'condition': condition,
        'prompt': _prompt_text(messages),
        'content': content,
        'content_empty': not (content or '').strip(),
        'reasoning': reasoning,
        'reasoning_chars': len(reasoning),
    }

    try:
        line = json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        return False

    try:
        with _WRITE_LOCK:
            with open(path, 'a', encoding='utf-8') as handle:
                handle.write(line + '\n')
    except OSError:
        return False
    return True


def load_capture(path: str) -> List[Dict[str, Any]]:
    """Read a capture file, skipping any line that is not valid JSON.

    A run killed mid-write can leave a partial final line; dropping it is
    preferable to refusing to read the thousands of intact records before it.

    Args:
        path: Capture file written by :func:`capture_response`.

    Returns:
        One dict per readable record, in file order.
    """
    records: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records
