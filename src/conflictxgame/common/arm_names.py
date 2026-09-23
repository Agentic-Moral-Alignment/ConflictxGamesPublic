"""Parse experiment-arm tokens from result CSV filenames."""
from __future__ import annotations

import os
import re
from collections.abc import Callable


RESULT_NAME = re.compile(
    r"^(?P<slug>.+?)_(?P<mode>mcq|open|interactive|play)"
    r"(?P<balanced>_balanced\d+)?(?P<arm>.*?)"
    r"(?P<tail>_user-.*|_fallback-.*)?\.csv$"
)


def split_result_name(basename: str) -> dict[str, str]:
    """Return the named filename components, rejecting non-result names."""
    name = os.path.basename(basename)
    match = RESULT_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Not a recognised result CSV filename: {name!r}")
    return {key: value or "" for key, value in match.groupdict().items()}


def arm_of(basename: str) -> str:
    """Return the experiment-arm suffix (``''`` denotes the baseline arm)."""
    return split_result_name(basename)["arm"]


def arm_token(basename: str, stem: str) -> str:
    """Return the arm suffix following an exact result-file *stem*.

    Provider/judge metadata beginning with ``_user-`` and fallback metadata
    beginning with ``_fallback-`` are not part of the experiment arm.
    """
    name = os.path.basename(basename)
    if not name.endswith(".csv"):
        raise ValueError(f"Expected a .csv result filename, got {name!r}")
    core = name[:-4]
    core = re.split(r"_(?:user|fallback)-", core, maxsplit=1)[0]
    if not core.startswith(stem):
        raise ValueError(f"Result filename {name!r} does not match stem {stem!r}")
    token = core[len(stem):]
    # Prevent a stem that is merely a prefix of the model slug from matching.
    if token and not token.startswith("_"):
        raise ValueError(f"Result filename {name!r} does not match stem {stem!r}")
    return token


def expected_arm(
    steered: bool,
    repeated: bool = False,
    states: bool = False,
) -> Callable[[str], bool]:
    """Build the exact arm predicate for one plotted condition."""
    if states:
        wanted = "_states-steer" if steered else "_states"
        return lambda arm: arm == wanted
    if repeated:
        prefix = "_steer-HHH_repeated_" if steered else "_repeated_"
        return lambda arm: arm.startswith(prefix)
    wanted = "_steer-HHH" if steered else ""
    return lambda arm: arm == wanted


def normalise_arm(arm: str) -> str:
    """Normalise the user-facing ``baseline`` spelling to its empty token."""
    return "" if arm == "baseline" else arm
