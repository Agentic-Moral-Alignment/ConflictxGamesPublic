"""Human-readable run configurations shared by every command-line entry point.

A run is a long, expensive, resumable thing whose settings have to stay
auditable months after it finished, so every entry point accepts the same
``--config PATH`` flag holding one flat object keyed by *argparse destination
name* (underscores, not option hyphens)::

    python -m conflictxgame.solo.filter_scenarios \\
        --config configs/solo/HHH_new_solo_20260826_01_filter_judge-gpt-5.6-luna.yaml

Explicit command-line arguments still win over the file, so a config records
the run and the command line records the one-off deviation from it.

YAML (``.yaml`` or ``.yml``) is the preferred format because it is readable
and supports comments. JSONC remains supported for older runs: it accepts
``//`` line comments, ``/* ... */`` block comments, and trailing commas.

Validation is strict and happens before any client is constructed: an unknown
key, a string where an integer belongs, or a value outside an option's
``choices`` aborts the run rather than silently reverting to a default that
would quietly change what the run measures.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

import yaml


#: Suffixes recognized as configuration files. YAML is preferred; JSON and
#: JSONC remain supported so historical run records stay reproducible.
CONFIG_SUFFIXES = ('.yaml', '.yml', '.json', '.jsonc')


def strip_jsonc(text: str) -> str:
    """Return ``text`` with JSONC comments and trailing commas removed.

    Comments are replaced by spaces and block comments keep their newlines, so
    every retained character stays at its original line and column and a
    :class:`json.JSONDecodeError` still points at the line the author wrote.

    Args:
        text: Raw file contents, possibly containing ``//`` line comments,
            ``/* ... */`` block comments, and commas before a closing brace
            or bracket.

    Returns:
        Strict JSON text of the same line structure.
    """
    out: List[str] = []
    index, length = 0, len(text)
    in_string = False
    while index < length:
        char = text[index]
        if in_string:
            out.append(char)
            if char == '\\' and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == '/' and index + 1 < length:
            following = text[index + 1]
            if following == '/':
                while index < length and text[index] != '\n':
                    out.append(' ')
                    index += 1
                continue
            if following == '*':
                end = text.find('*/', index + 2)
                end = length if end == -1 else end + 2
                # Keep newlines so reported line numbers stay truthful.
                out.extend('\n' if c == '\n' else ' ' for c in text[index:end])
                index = end
                continue
        if char in '}]':
            # Drop a trailing comma: walk back over the whitespace just
            # emitted, which cannot be inside a string at this point.
            back = len(out) - 1
            while back >= 0 and out[back].isspace():
                back -= 1
            if back >= 0 and out[back] == ',':
                out[back] = ' '
        out.append(char)
        index += 1
    return ''.join(out)


def load_jsonc(path: str):
    """Parse one JSONC file.

    Args:
        path: Path to a ``.json`` or ``.jsonc`` file.

    Returns:
        The decoded value.

    Raises:
        OSError: The file could not be read.
        json.JSONDecodeError: The file is not valid JSONC.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        return json.loads(strip_jsonc(handle.read()))


def load_config(path: str):
    """Parse a YAML, JSON, or JSONC config file based on its suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in ('.yaml', '.yml'):
        with open(path, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle)
    if suffix in ('.json', '.jsonc'):
        return load_jsonc(path)
    raise ValueError(
        f'unsupported config suffix {suffix!r}; expected one of '
        f'{", ".join(CONFIG_SUFFIXES)}'
    )


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--config`` flag to ``parser``."""
    parser.add_argument(
        '--config', type=str, default=None,
        help='YAML (preferred), JSON, or JSONC file of argparse destination '
             'names (underscores, not option hyphens). Explicit command-line '
             'arguments override its values',
    )


def explicit_cli_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set:
    """Return parser destinations explicitly present in ``argv``.

    Lets a caller tell "the user asked for this on the command line" apart
    from "this came from the config file or an argparse default", which
    matters wherever one setting derives another.

    Args:
        parser: The parser whose option strings define the mapping.
        argv: Argument list, excluding the program name.

    Returns:
        The set of destination names named by ``argv``.
    """
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    explicit = set()
    for token in argv:
        option = token.split('=', 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit


def _takes_list(action: argparse.Action) -> bool:
    """Whether this action collects several values into a list."""
    return action.nargs in ('*', '+') or isinstance(action.nargs, int)


class _ScalarView:
    """One element of a list-valued action, seen as a scalar action.

    ``_coerce_value`` validates against ``type`` and ``choices``; for a list
    option those describe each element, so a view with ``nargs`` cleared lets
    the same code check the elements without recursing forever.
    """

    def __init__(self, action: argparse.Action):
        self.dest = action.dest
        self.type = action.type
        self.choices = action.choices
        self.required = False
        self.default = None
        self.nargs = None


def _coerce_value(
    parser: argparse.ArgumentParser,
    action: argparse.Action,
    value,
):
    """Strictly validate one JSON value against its argparse action.

    Args:
        parser: Parser used to report errors (exits with status 2).
        action: The action the key maps to, supplying type and choices.
        value: The decoded JSON value.

    Returns:
        The value converted to the type the action produces.
    """
    key = action.dest
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            parser.error(f'config key {key!r} must be a boolean')
        converted = value
    elif value is not None and _takes_list(action):
        # A repeatable option (nargs '*', '+' or a count) produces a list on the
        # command line, so its config form is a JSON list; each element is then
        # validated against the same action as a scalar would be.
        if not isinstance(value, list):
            parser.error(f'config key {key!r} must be a list')
        scalar = _ScalarView(action)
        converted = [_coerce_value(parser, scalar, item) for item in value]
        if isinstance(action.nargs, int) and len(converted) != action.nargs:
            parser.error(
                f'config key {key!r} must have exactly {action.nargs} items, '
                f'got {len(converted)}'
            )
        if action.nargs == '+' and not converted:
            parser.error(f'config key {key!r} must have at least one item')
        return converted
    elif value is None:
        if action.required or action.default is not None:
            parser.error(f'config key {key!r} may not be null')
        converted = None
    elif action.type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            parser.error(f'config key {key!r} must be an integer')
        converted = value
    elif action.type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            parser.error(f'config key {key!r} must be a number')
        converted = float(value)
    elif action.type is str or action.type is None:
        if not isinstance(value, str):
            parser.error(f'config key {key!r} must be a string')
        converted = value
    else:
        try:
            converted = action.type(value)
        except (TypeError, ValueError) as exc:
            parser.error(f'invalid config value for {key!r}: {exc}')
    if action.choices is not None and converted not in action.choices:
        parser.error(
            f'config key {key!r} must be one of {list(action.choices)!r}, '
            f'not {converted!r}'
        )
    return converted


def load_config_defaults(parser: argparse.ArgumentParser, path: str) -> dict:
    """Load and strictly validate config defaults for ``parser``.

    Args:
        parser: The parser the config is written against.
        path: Path to the config file.

    Returns:
        A mapping of destination name to validated value, ready for
        :meth:`argparse.ArgumentParser.set_defaults`.
    """
    try:
        raw = load_config(path)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        parser.error(f'could not read --config {path!r}: {exc}')
    if not isinstance(raw, dict):
        parser.error('--config must contain one mapping/object')

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in ('help', 'config')
    }
    unknown = sorted(set(raw) - set(actions))
    if unknown:
        parser.error(
            f'unknown config key(s): {", ".join(unknown)}; use argparse '
            'destination names such as "value_set" and "round_batch"'
        )
    return {
        key: _coerce_value(parser, actions[key], value)
        for key, value in raw.items()
    }


def parse_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, dict, set]:
    """Parse ``argv`` with config values as defaults.

    Required options are relaxed before parsing and re-checked afterwards, so
    a config file may supply them; the substitute message names both the flag
    and the config key, since either is a legitimate way to provide it.

    Args:
        parser: A parser that has been given :func:`add_config_argument`.
        argv: Argument list excluding the program name (defaults to
            ``sys.argv[1:]``).

    Returns:
        ``(args, config_defaults, explicit_cli)`` -- the parsed namespace, the
        validated mapping the config supplied (empty when no ``--config`` was
        given), and the destinations named on the command line.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    probed, _ = probe.parse_known_args(argv)
    config_defaults = (
        load_config_defaults(parser, probed.config) if probed.config else {}
    )

    required = [action for action in parser._actions if action.required]
    for action in required:
        action.required = False
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)

    missing = [
        action for action in required
        if getattr(args, action.dest, None) is None
    ]
    if missing:
        flags = ', '.join(
            '/'.join(action.option_strings) or action.dest for action in missing
        )
        keys = ', '.join(f'"{action.dest}"' for action in missing)
        parser.error(
            f'missing required setting(s): {flags} -- give them on the command '
            f'line or as config key(s) {keys}'
        )
    return args, config_defaults, explicit_cli_dests(parser, argv)
