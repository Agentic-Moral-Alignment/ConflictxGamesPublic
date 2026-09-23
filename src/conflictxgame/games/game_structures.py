"""Data model for multi-agent, game-theoretic ConflictxGame scenarios.

This module defines the configuration space (structure x player count x
action-arity vector), the concrete scenario representation (:class:`GameSpec`),
config enumeration/sampling with soft-cap warnings, coverage reporting, and
payoff-tensor helpers (pure Nash equilibria, welfare, symmetry and
canonical-pattern detection).

Design notes:
    * Games are parameterized by ``(structure, n_players, arities)`` rather
      than by canonical named games (prisoner's dilemma, chicken, ...). Named
      patterns are only *detected post hoc* from a scenario's payoff tensor
      (see :func:`detect_canonical_2x2`), e.g. to gate zero-determinant
      scripted strategies.
    * Payoff tensors are **annotative**: they describe the material stakes of
      a scenario for an economically rational agent, but are not assumed to be
      the utility function of the (moral) agent under evaluation. The primary
      evaluation signal is the model's own preference ranking over actions and
      predicted outcomes.
    * Player-count and arity ranges are **soft-capped**: any size is accepted,
      but configurations whose joint-profile count exceeds
      :data:`DEFAULT_PROFILE_WARN_THRESHOLD` trigger a
      :class:`RepresentationSizeWarning` describing the representational
      blowup (tensor entries, per-profile narrative outcomes, prompt length).
"""

import itertools
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Recognized game structures. ``single`` is the classic one-agent
#: ConflictScope setting (now allowing up to 3 actions); the other three are
#: multi-agent. ``repeated`` denotes a finitely repeated simultaneous game.
STRUCTURES = ("single", "simultaneous", "sequential", "repeated")

#: Default (inclusive) player-count range used by the generation CLI. The
#: default configuration space is pinned to one-shot 2-player 2x2 games;
#: larger spaces remain available via the CLI flags.
DEFAULT_PLAYER_RANGE = (2, 2)

#: Default (inclusive) per-player action-arity range used by the generation CLI.
DEFAULT_ARITY_RANGE = (2, 2)

#: Default horizon (number of rounds) for ``repeated`` games.
DEFAULT_HORIZON = 5

#: Joint-profile count above which :meth:`GameConfig.warn_if_large` emits a
#: :class:`RepresentationSizeWarning` (27 = 3 players x 3 actions).
DEFAULT_PROFILE_WARN_THRESHOLD = 27

#: Rough per-item token costs used to estimate prompt size in
#: :meth:`GameConfig.representation_size`.
_TOKENS_PER_OUTCOME = 45
_TOKENS_PER_ACTION = 60
_TOKENS_BASE = 400


class RepresentationSizeWarning(UserWarning):
    """Warns that a game configuration is expensive to represent, not invalid."""


def profile_key(profile: Sequence[int]) -> str:
    """Serialize a joint action profile to the canonical string key.

    Args:
        profile: One action index per player, e.g. ``(0, 2)``.

    Returns:
        A comma-joined key, e.g. ``"0,2"``, used to index payoff tensors and
        outcome descriptions in :class:`GameSpec`.
    """
    return ",".join(str(int(a)) for a in profile)


def parse_profile_key(key: str) -> Tuple[int, ...]:
    """Inverse of :func:`profile_key`.

    Args:
        key: A comma-joined action-index string, e.g. ``"0,2"``.

    Returns:
        The corresponding tuple of action indices.
    """
    return tuple(int(part) for part in key.split(","))


def enumerate_joint_profiles(arities: Sequence[int]) -> List[Tuple[int, ...]]:
    """Enumerate all joint action profiles for the given per-player arities.

    Args:
        arities: Number of actions available to each player.

    Returns:
        All profiles in lexicographic order, e.g. ``[(0,0), (0,1), (1,0), (1,1)]``
        for ``[2, 2]``.
    """
    return list(itertools.product(*[range(k) for k in arities]))


@dataclass
class ActionSpec:
    """One action available to one player.

    Attributes:
        label: Short unique (per player) name shown in menus and transcripts.
        description: One-sentence description of taking the action.
        value_tag: The value (a key of the active value set) this action favors.
    """

    label: str
    description: str
    value_tag: str

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {"label": self.label, "description": self.description, "value_tag": self.value_tag}

    @classmethod
    def from_dict(cls, data: dict) -> "ActionSpec":
        """Reconstruct an :class:`ActionSpec` from :meth:`to_dict` output."""
        return cls(label=data["label"], description=data["description"], value_tag=data["value_tag"])


#: Recognized decision-maker kinds. ``human`` is an ordinary person;
#: ``assistant`` is a deployed text-only AI assistant playing a seat.
#: Value definitions, filter checks, and play prompts resolve per seat kind.
AGENT_KINDS = ("human", "assistant")


@dataclass
class PlayerSpec:
    """One role in a game scenario.

    Attributes:
        name: The character's name (concrete, per ConflictScope grounding rules).
        persona: Short biographical paragraph (mirrors the dyadic ``user_prompt``).
            For ``assistant`` seats this describes the deployment context (what
            product the assistant powers and who interacts with it).
        role_description: The player's situation, stakes, and private context.
        actions: The player's action menu; ``len(actions)`` is the player's arity.
        value_pair: The value names (from the value set) whose tension this
            role instantiates. May contain more than two entries for arity-3 menus.
        agent_kind: One of :data:`AGENT_KINDS`; defaults to ``"human"``.
    """

    name: str
    persona: str
    role_description: str
    actions: List[ActionSpec]
    value_pair: List[str] = field(default_factory=list)
    agent_kind: str = "human"

    @property
    def arity(self) -> int:
        """Number of actions available to this player."""
        return len(self.actions)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "name": self.name,
            "persona": self.persona,
            "role_description": self.role_description,
            "actions": [a.to_dict() for a in self.actions],
            "value_pair": list(self.value_pair),
            "agent_kind": self.agent_kind,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerSpec":
        """Reconstruct a :class:`PlayerSpec` from :meth:`to_dict` output.

        ``agent_kind`` defaults to ``"human"`` so scenario files generated
        before the field existed remain loadable.
        """
        return cls(
            name=data["name"],
            persona=data.get("persona", ""),
            role_description=data.get("role_description", ""),
            actions=[ActionSpec.from_dict(a) for a in data["actions"]],
            value_pair=list(data.get("value_pair", [])),
            agent_kind=data.get("agent_kind", "human"),
        )


@dataclass
class GameConfig:
    """A cell of the game configuration space.

    Attributes:
        structure: One of :data:`STRUCTURES`.
        n_players: Number of players (>= 1; ``single`` requires exactly 1,
            the multi-agent structures require >= 2).
        arities: Per-player action counts (each >= 2); asymmetric vectors such
            as ``(2, 3)`` are allowed.
        horizon: Number of rounds. Must be 1 except for ``repeated`` games,
            where it must be >= 2 (default :data:`DEFAULT_HORIZON` at sampling
            time).
    """

    structure: str
    n_players: int
    arities: Tuple[int, ...]
    horizon: int = 1

    def __post_init__(self):
        self.arities = tuple(int(k) for k in self.arities)
        if self.structure not in STRUCTURES:
            raise ValueError(f"Unknown structure '{self.structure}'; expected one of {STRUCTURES}")
        if self.n_players < 1:
            raise ValueError("n_players must be >= 1")
        if len(self.arities) != self.n_players:
            raise ValueError(
                f"arities has {len(self.arities)} entries but n_players={self.n_players}"
            )
        if any(k < 2 for k in self.arities):
            raise ValueError("Every player needs at least 2 actions")
        if self.structure == "single" and self.n_players != 1:
            raise ValueError("structure 'single' requires exactly 1 player")
        if self.structure != "single" and self.n_players < 2:
            raise ValueError(f"structure '{self.structure}' requires >= 2 players")
        if self.structure == "repeated":
            if self.horizon < 2:
                raise ValueError("repeated games require horizon >= 2")
        elif self.horizon != 1:
            raise ValueError(f"structure '{self.structure}' requires horizon == 1")

    @property
    def joint_profile_count(self) -> int:
        """Number of joint action profiles (the product of the arities)."""
        count = 1
        for k in self.arities:
            count *= k
        return count

    def representation_size(self) -> Dict[str, int]:
        """Estimate how large this configuration is to represent.

        Returns:
            Dict with:
                * ``joint_profiles``: number of joint action profiles.
                * ``tensor_entries``: payoff numbers to specify
                  (``joint_profiles * n_players``).
                * ``outcome_descriptions``: narrative outcome paragraphs the
                  generator must write (one per joint profile).
                * ``estimated_prompt_tokens``: rough token cost of presenting
                  the full game (menus + outcomes) in a single prompt.
        """
        profiles = self.joint_profile_count
        return {
            "joint_profiles": profiles,
            "tensor_entries": profiles * self.n_players,
            "outcome_descriptions": profiles,
            "estimated_prompt_tokens": (
                _TOKENS_BASE
                + _TOKENS_PER_OUTCOME * profiles
                + _TOKENS_PER_ACTION * sum(self.arities)
            ),
        }

    def warn_if_large(self, threshold: int = DEFAULT_PROFILE_WARN_THRESHOLD) -> bool:
        """Emit a :class:`RepresentationSizeWarning` if the config is large.

        Sizes are a soft cap: the configuration is *accepted* regardless, but
        callers (and users raising the CLI ranges) are warned about the cost of
        representing the game.

        Args:
            threshold: Joint-profile count above which to warn.

        Returns:
            True if a warning was emitted.
        """
        size = self.representation_size()
        if size["joint_profiles"] > threshold:
            warnings.warn(
                f"Game config {self.cell_key()} has {size['joint_profiles']} joint action "
                f"profiles (> {threshold}). Representing it requires "
                f"{size['tensor_entries']} payoff entries and "
                f"{size['outcome_descriptions']} narrative outcome descriptions, "
                f"with an estimated prompt size of ~{size['estimated_prompt_tokens']} tokens "
                f"per query. Generation quality and evaluation cost degrade with size; "
                f"consider smaller player counts or arities.",
                RepresentationSizeWarning,
                stacklevel=2,
            )
            return True
        return False

    def cell_key(self) -> str:
        """Return a stable identifier for this configuration cell.

        Examples:
            ``"simultaneous_n2_a2x3"``, ``"repeated_n2_a2x2_t5"``.
        """
        arity_part = "x".join(str(k) for k in self.arities)
        key = f"{self.structure}_n{self.n_players}_a{arity_part}"
        if self.structure == "repeated":
            key += f"_t{self.horizon}"
        return key

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "structure": self.structure,
            "n_players": self.n_players,
            "arities": list(self.arities),
            "horizon": self.horizon,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameConfig":
        """Reconstruct a :class:`GameConfig` from :meth:`to_dict` output."""
        return cls(
            structure=data["structure"],
            n_players=int(data["n_players"]),
            arities=tuple(data["arities"]),
            horizon=int(data.get("horizon", 1)),
        )


@dataclass
class GameSpec:
    """A fully instantiated game scenario.

    Attributes:
        game_id: Unique identifier (``<cell_key>-<index>`` by convention).
        config: The :class:`GameConfig` cell this scenario instantiates.
        description: Concrete narrative description of the situation
            (mirrors the dyadic pipeline's ``description`` field).
        players: One :class:`PlayerSpec` per seat.
        payoffs: Map from :func:`profile_key` to per-player numeric payoffs.
            **Annotative** (see module docstring): a reference for economic
            rationality analysis, not ground-truth utility.
        outcome_descriptions: Map from :func:`profile_key` to a narrative
            description of what happens under that joint action profile.
        turn_order: Seat indices in move order (used by ``sequential`` games;
            defaults to seat order).
        value_set: Name of the value set the value tags come from.
        annotative_payoffs: Flag recording the payoff-tensor stance above.
        metadata: Free-form extras (generating model, prompt type, etc.).
    """

    game_id: str
    config: GameConfig
    description: str
    players: List[PlayerSpec]
    payoffs: Dict[str, List[float]]
    outcome_descriptions: Dict[str, str] = field(default_factory=dict)
    turn_order: List[int] = field(default_factory=list)
    value_set: str = ""
    annotative_payoffs: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.turn_order:
            self.turn_order = list(range(self.config.n_players))

    # ------------------------------------------------------------------ #
    # Validation and lookups
    # ------------------------------------------------------------------ #

    def validate(self) -> List[str]:
        """Check internal consistency.

        Returns:
            A list of human-readable problems; empty when the spec is valid.
        """
        problems = []
        if len(self.players) != self.config.n_players:
            problems.append(
                f"{len(self.players)} players but config expects {self.config.n_players}"
            )
        for seat, player in enumerate(self.players):
            if seat < len(self.config.arities) and player.arity != self.config.arities[seat]:
                problems.append(
                    f"player {seat} has {player.arity} actions but config expects "
                    f"{self.config.arities[seat]}"
                )
            labels = [a.label for a in player.actions]
            if len(set(labels)) != len(labels):
                problems.append(f"player {seat} has duplicate action labels")
            if any(not a.value_tag for a in player.actions):
                problems.append(f"player {seat} has actions without value tags")
            if player.agent_kind not in AGENT_KINDS:
                problems.append(
                    f"player {seat} has unknown agent_kind '{player.agent_kind}' "
                    f"(expected one of {AGENT_KINDS})"
                )
        expected_keys = {profile_key(p) for p in enumerate_joint_profiles(self.config.arities)}
        actual_keys = set(self.payoffs.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            if missing:
                problems.append(
                    f"payoffs missing profiles: {missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            if extra:
                problems.append(
                    f"payoffs has unexpected profiles: {extra[:5]}{'...' if len(extra) > 5 else ''}"
                )
        for key, values in self.payoffs.items():
            if len(values) != self.config.n_players:
                problems.append(
                    f"payoffs[{key}] has {len(values)} entries, expected {self.config.n_players}"
                )
                break
        if self.outcome_descriptions:
            missing_outcomes = expected_keys - set(self.outcome_descriptions.keys())
            if missing_outcomes:
                problems.append(
                    f"outcome_descriptions missing profiles: {sorted(missing_outcomes)[:5]}"
                )
        if sorted(self.turn_order) != list(range(self.config.n_players)):
            problems.append(f"turn_order {self.turn_order} is not a permutation of seats")
        return problems

    @property
    def agent_mix(self) -> str:
        """Composition of seat kinds: ``all_human``, ``all_assistant``, or ``mixed``."""
        kinds = {p.agent_kind for p in self.players}
        if kinds == {"human"}:
            return "all_human"
        if kinds == {"assistant"}:
            return "all_assistant"
        return "mixed"

    def payoff(self, profile: Sequence[int]) -> List[float]:
        """Return the per-player payoffs for a joint action profile."""
        return list(self.payoffs[profile_key(profile)])

    def outcome(self, profile: Sequence[int]) -> str:
        """Return the narrative outcome description for a joint action profile."""
        return self.outcome_descriptions.get(profile_key(profile), "")

    def welfare(self, profile: Sequence[int]) -> float:
        """Return total (utilitarian) welfare of a profile under the annotative payoffs."""
        return float(sum(self.payoff(profile)))

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "game_id": self.game_id,
            "config": self.config.to_dict(),
            "description": self.description,
            "players": [p.to_dict() for p in self.players],
            "payoffs": {k: list(v) for k, v in self.payoffs.items()},
            "outcome_descriptions": dict(self.outcome_descriptions),
            "turn_order": list(self.turn_order),
            "value_set": self.value_set,
            "annotative_payoffs": self.annotative_payoffs,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Return the spec serialized as a JSON string (for CSV columns)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "GameSpec":
        """Reconstruct a :class:`GameSpec` from :meth:`to_dict` output."""
        return cls(
            game_id=data["game_id"],
            config=GameConfig.from_dict(data["config"]),
            description=data["description"],
            players=[PlayerSpec.from_dict(p) for p in data["players"]],
            payoffs={k: [float(x) for x in v] for k, v in data["payoffs"].items()},
            outcome_descriptions=dict(data.get("outcome_descriptions", {})),
            turn_order=list(data.get("turn_order", [])),
            value_set=data.get("value_set", ""),
            annotative_payoffs=bool(data.get("annotative_payoffs", True)),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "GameSpec":
        """Reconstruct a :class:`GameSpec` from :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------- #
# Payoff-tensor analysis helpers (all annotative)
# ---------------------------------------------------------------------- #

def max_welfare_profile(spec: GameSpec) -> Tuple[int, ...]:
    """Return the joint profile maximizing total welfare.

    Ties are broken by the highest minimum per-player payoff, then
    lexicographically, so the result is deterministic.
    """
    profiles = enumerate_joint_profiles(spec.config.arities)
    return max(
        profiles,
        key=lambda p: (spec.welfare(p), min(spec.payoff(p)), tuple(-a for a in p)),
    )


def min_welfare_profile(spec: GameSpec) -> Tuple[int, ...]:
    """Return the joint profile minimizing total welfare (deterministic tie-break)."""
    profiles = enumerate_joint_profiles(spec.config.arities)
    return min(profiles, key=lambda p: (spec.welfare(p), -min(spec.payoff(p)), p))


def cooperative_actions(spec: GameSpec) -> Tuple[int, ...]:
    """Per-player actions played in the welfare-maximizing profile.

    Used by scripted strategies (tit-for-tat's opening move, grim trigger's
    trigger condition) as the operational definition of "cooperating".
    """
    return max_welfare_profile(spec)


def punishment_actions(spec: GameSpec) -> Tuple[int, ...]:
    """Per-player actions played in the welfare-minimizing profile.

    Used by scripted strategies as the operational definition of "punishing".
    """
    return min_welfare_profile(spec)


def best_responses(spec: GameSpec, profile: Sequence[int], seat: int) -> List[int]:
    """Actions for ``seat`` maximizing its own payoff, holding others fixed.

    Args:
        spec: The game.
        profile: A joint profile supplying the other players' actions.
        seat: The player index whose best responses are computed.

    Returns:
        All payoff-maximizing action indices for ``seat``.
    """
    profile = list(profile)
    best_value = None
    best = []
    for action in range(spec.config.arities[seat]):
        candidate = list(profile)
        candidate[seat] = action
        value = spec.payoff(candidate)[seat]
        if best_value is None or value > best_value + 1e-12:
            best_value = value
            best = [action]
        elif abs(value - best_value) <= 1e-12:
            best.append(action)
    return best


def is_best_response(spec: GameSpec, profile: Sequence[int], seat: int) -> bool:
    """Whether ``seat``'s action in ``profile`` is a best response to the others."""
    return list(profile)[seat] in best_responses(spec, profile, seat)


def pure_nash_equilibria(spec: GameSpec) -> List[Tuple[int, ...]]:
    """Enumerate pure-strategy Nash equilibria by brute force.

    At the default scale (<= 3 players x 3 actions => <= 27 profiles) this is
    trivial; it remains exact (just slower) for larger soft-capped configs.

    Returns:
        All profiles where every player's action is a best response.
    """
    equilibria = []
    for profile in enumerate_joint_profiles(spec.config.arities):
        if all(is_best_response(spec, profile, seat) for seat in range(spec.config.n_players)):
            equilibria.append(profile)
    return equilibria


def is_symmetric_two_by_two(spec: GameSpec) -> bool:
    """Whether the game is a 2-player, 2x2, payoff-symmetric game.

    Symmetry here means the game looks identical from both seats:
    ``payoff((i, j))[0] == payoff((j, i))[1]`` for all action pairs.
    """
    if spec.config.n_players != 2 or tuple(spec.config.arities) != (2, 2):
        return False
    for i in range(2):
        for j in range(2):
            if abs(spec.payoff((i, j))[0] - spec.payoff((j, i))[1]) > 1e-9:
                return False
    return True


def pd_payoffs(spec: GameSpec) -> Optional[Dict[str, float]]:
    """Extract prisoner's-dilemma-ordered payoffs from a symmetric 2x2 game.

    Tries both labelings of which action is "cooperate" and returns the one
    satisfying the strict PD ordering ``T > R > P > S``.

    Returns:
        ``{"R": ..., "S": ..., "T": ..., "P": ..., "cooperative_action": c}``
        if the game is symmetric 2x2 with a PD ordering, else None. This is
        the applicability gate for zero-determinant strategies (which assume
        a symmetric memory-one game).
    """
    if not is_symmetric_two_by_two(spec):
        return None
    for c in (0, 1):
        d = 1 - c
        R = spec.payoff((c, c))[0]
        S = spec.payoff((c, d))[0]
        T = spec.payoff((d, c))[0]
        P = spec.payoff((d, d))[0]
        if T > R > P > S:
            return {"R": R, "S": S, "T": T, "P": P, "cooperative_action": c}
    return None


def detect_canonical_2x2(spec: GameSpec) -> Optional[str]:
    """Annotate a symmetric 2x2 game with the canonical pattern it resembles.

    Purely post-hoc annotation (games are never generated from named types):

    * ``prisoners_dilemma_like``: T > R > P > S
    * ``stag_hunt_like``: R > T and P > S
    * ``chicken_like``: T > R > S > P
    * ``harmony_like``: R > T and S > P

    Returns:
        The pattern name, or None if the game is not symmetric 2x2 or matches
        no pattern under either labeling of the "cooperative" action.
    """
    if not is_symmetric_two_by_two(spec):
        return None
    for c in (0, 1):
        d = 1 - c
        R = spec.payoff((c, c))[0]
        S = spec.payoff((c, d))[0]
        T = spec.payoff((d, c))[0]
        P = spec.payoff((d, d))[0]
        if T > R > P > S:
            return "prisoners_dilemma_like"
        if T > R > S > P:
            return "chicken_like"
        if R > T and P > S:
            return "stag_hunt_like"
        if R > T and S > P:
            return "harmony_like"
    return None


def as_repeated(spec: GameSpec, horizon: int) -> GameSpec:
    """Return a repeated-structure copy of a one-shot simultaneous spec.

    Used at evaluation time to play one-shot scenarios as N-round repeated
    sequences (see ``--rounds`` in :mod:`conflictxgame.games.evaluate_games`):
    the copy has ``structure="repeated"`` and the given ``horizon``, passes
    the usual :class:`GameConfig` validation, and records
    ``metadata["derived_from_structure"] = "simultaneous"`` so analysis can
    distinguish natively repeated scenarios from converted one-shots. The
    original spec is not modified.

    Args:
        spec: A ``simultaneous`` (one-shot) spec.
        horizon: Number of rounds for the repeated copy (>= 2).

    Returns:
        A new, validated :class:`GameSpec`.

    Raises:
        ValueError: If ``spec`` is not simultaneous or ``horizon < 2``.
    """
    if spec.config.structure != "simultaneous":
        raise ValueError(
            f"as_repeated requires a simultaneous spec, got '{spec.config.structure}'"
        )
    if horizon < 2:
        raise ValueError("as_repeated requires horizon >= 2")
    data = spec.to_dict()
    data["config"]["structure"] = "repeated"
    data["config"]["horizon"] = horizon
    clone = GameSpec.from_dict(data)
    clone.metadata["derived_from_structure"] = "simultaneous"
    return clone


# ---------------------------------------------------------------------- #
# Config enumeration / sampling
# ---------------------------------------------------------------------- #

def enumerate_configs(
    structures: Iterable[str],
    player_range: Tuple[int, int] = DEFAULT_PLAYER_RANGE,
    arity_range: Tuple[int, int] = DEFAULT_ARITY_RANGE,
    horizon: int = DEFAULT_HORIZON,
) -> List[GameConfig]:
    """Enumerate all configuration cells implied by the requested ranges.

    Arity vectors are enumerated as multisets (non-decreasing), since seat
    relabelings of the same arity multiset describe the same cell; asymmetric
    vectors such as ``(2, 3)`` are therefore included exactly once.

    ``single`` cells are produced only when the player range includes 1; the
    multi-agent structures use player counts >= 2. There is no hard upper
    bound — see :meth:`GameConfig.warn_if_large` for the soft cap.

    Args:
        structures: Subset of :data:`STRUCTURES` to include.
        player_range: Inclusive (min, max) player counts.
        arity_range: Inclusive (min, max) per-player action counts.
        horizon: Round count assigned to ``repeated`` cells.

    Returns:
        The list of :class:`GameConfig` cells, in a stable order.
    """
    min_players, max_players = player_range
    min_arity, max_arity = arity_range
    if min_players < 1 or max_players < min_players:
        raise ValueError(f"Invalid player range {player_range}")
    if min_arity < 2 or max_arity < min_arity:
        raise ValueError(f"Invalid arity range {arity_range}")

    configs = []
    for structure in structures:
        if structure not in STRUCTURES:
            raise ValueError(f"Unknown structure '{structure}'")
        if structure == "single":
            if min_players > 1:
                continue
            player_counts = [1]
        else:
            player_counts = list(range(max(2, min_players), max_players + 1))
        for n in player_counts:
            for arities in itertools.combinations_with_replacement(
                range(min_arity, max_arity + 1), n
            ):
                configs.append(
                    GameConfig(
                        structure=structure,
                        n_players=n,
                        arities=tuple(arities),
                        horizon=horizon if structure == "repeated" else 1,
                    )
                )
    return configs


class CoverageReport:
    """Tracks what was (and was not) generated across configuration cells.

    The configuration space is covered by *sampling*, not exhaustion; this
    report makes the sampling explicit. One row per cell records the target
    count, the number of scenarios actually generated, the number kept after
    filtering, and free-form notes (e.g. "skipped: representation too large",
    "zd_extort inapplicable").
    """

    def __init__(self):
        self.cells: Dict[str, dict] = {}

    def ensure_cell(self, config: GameConfig, target: int = 0) -> dict:
        """Register a cell (idempotent) and return its mutable row."""
        key = config.cell_key()
        if key not in self.cells:
            self.cells[key] = {
                "cell_key": key,
                "structure": config.structure,
                "n_players": config.n_players,
                "arities": "x".join(str(k) for k in config.arities),
                "horizon": config.horizon,
                "target": target,
                "generated": 0,
                "kept": 0,
                "notes": [],
            }
        elif target:
            self.cells[key]["target"] = target
        return self.cells[key]

    def record_generated(self, config: GameConfig, count: int) -> None:
        """Add ``count`` to the cell's generated tally."""
        self.ensure_cell(config)["generated"] += count

    def record_kept(self, cell_key: str, count: int) -> None:
        """Add ``count`` to the kept-after-filter tally for a cell (by key)."""
        if cell_key in self.cells:
            self.cells[cell_key]["kept"] += count

    def add_note(self, config: GameConfig, note: str) -> None:
        """Attach a free-form note (skip reason, warning, ...) to a cell."""
        self.ensure_cell(config)["notes"].append(note)

    def to_json(self, path: str) -> None:
        """Write the report as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(self.cells.values()), f, indent=2, ensure_ascii=False)

    def to_csv(self, path: str) -> None:
        """Write the report as CSV (notes joined with '; ')."""
        import csv

        fieldnames = [
            "cell_key", "structure", "n_players", "arities", "horizon",
            "target", "generated", "kept", "notes",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.cells.values():
                out = dict(row)
                out["notes"] = "; ".join(row["notes"])
                writer.writerow(out)

    @classmethod
    def from_json(cls, path: str) -> "CoverageReport":
        """Load a report previously written by :meth:`to_json`."""
        report = cls()
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                report.cells[row["cell_key"]] = row
        return report

    def summary(self) -> str:
        """Return a short human-readable summary of coverage."""
        lines = ["cell_key: generated/target (kept)"]
        for row in self.cells.values():
            lines.append(
                f"  {row['cell_key']}: {row['generated']}/{row['target']} ({row['kept']} kept)"
                + (f" — {'; '.join(row['notes'])}" if row["notes"] else "")
            )
        return "\n".join(lines)
