"""Canned :class:`ModelWrapper` fakes shared across the offline test suite.

No API keys required: :class:`DummyGenerationClient` produces structurally
valid stage-1/stage-2 generation outputs, :class:`DummyEvalClient` gives
deterministic MCQ/Likert/ranking/action answers, and :class:`DummyJudgeClient`
answers filter and outline-screen checks with configurable patterns.

Every fake routes its reply through ``_handle_api_success`` so the usage
ledger on :class:`~conflictxgame.common.model_wrappers.ModelWrapper` counts
calls offline, which is what lets the cost assertions run without API keys.
"""

import json
import re

from conflictxgame.common.model_wrappers import ModelWrapper


class DummyGenerationClient(ModelWrapper):
    """Produces structurally valid stage-1 and stage-2 generation outputs."""

    def __init__(self):
        super().__init__("dummy-generator")
        self.paper_calls = 0

    def generate(self, messages):
        text = messages[-1]["content"]
        if "scenario outlines" in text:
            self._handle_api_success()
            return self._stage_one(text)
        if "was reviewed and rejected" in text:
            self._handle_api_success()
            return self._stage_two(text)
        raise AssertionError("Unexpected single-generate prompt in generation dummy")

    def batch_generate(self, messages_list, verbose=False, batch_key=None):
        for _ in messages_list:
            self._handle_api_success()
        return [self._stage_two(m[-1]["content"]) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError

    @staticmethod
    def _parse_assignment(text):
        """Recover per-scenario, per-seat slot values from the assignment block."""
        scenarios = {}
        current = None
        for line in text.splitlines():
            match = re.match(r"scenario_(\d+):", line.strip())
            if match:
                current = int(match.group(1))
                scenarios[current] = []
                continue
            match = re.match(
                r"decision-maker (\d+) \((?:human|AI assistant), (\d+) actions\): (.*)",
                line.strip(),
            )
            if match and current is not None:
                slots = re.findall(r"action \d+ -> (\w+)", match.group(3))
                scenarios[current].append(slots)
        return scenarios

    def _stage_one(self, text):
        if '"favors"' in text:
            return self._stage_one_paper(text)
        assignments = self._parse_assignment(text)
        output = {}
        for idx, seats in assignments.items():
            players = []
            for seat, slots in enumerate(seats):
                players.append(
                    {
                        "name": f"Casey Morgan {seat + 1}",
                        "role_description": f"Concrete stakes for decision-maker {seat + 1} at Maple & 5th in Columbus, Ohio.",
                        "actions": [
                            {
                                "label": f"favor {value} (p{seat})",
                                "description": f"Take the action favoring {value}.",
                            }
                            for value in slots
                        ],
                    }
                )
            output[f"scenario_{idx}"] = {
                "context": f"Scenario {idx}: interdependent decisions at the Columbus food co-op.",
                "players": players,
            }
        return json.dumps(output)

    def _stage_one_paper(self, text):
        """Produce paper-style actions carrying model-selected ``favors`` tags."""
        values = re.findall(r"^- (\w+):", text, flags=re.MULTILINE)
        # The paper prompt carries every value set entry.  Keep this fallback
        # for a malformed test prompt so the fake's error stays structural.
        values = values or ["v1", "v2"]
        scenarios = {}
        current = 1
        scenarios[current] = []
        for line in text.splitlines():
            scenario = re.match(r"scenario_(\d+):", line.strip())
            if scenario:
                current = int(scenario.group(1))
                scenarios.setdefault(current, [])
                continue
            seat = re.match(
                r"decision-maker \d+: (?:human|AI assistant), (\d+) options"
                r"(?:; required values \(one option each, either order\): (.*))?",
                line.strip(),
            )
            if seat:
                targets = (
                    [value.strip() for value in seat.group(2).split(",")]
                    if seat.group(2)
                    else None
                )
                scenarios.setdefault(current, []).append(
                    {"arity": int(seat.group(1)), "targets": targets}
                )

        output = {}
        for scenario_idx, seats in scenarios.items():
            players = []
            # Move the base pair between calls so paper-style quota tests
            # exercise passive coverage rather than receiving one pair forever.
            base = self.paper_calls + scenario_idx - 1
            for seat, seat_spec in enumerate(seats):
                arity = seat_spec["arity"]
                target_values = seat_spec["targets"]
                action_values = target_values or [
                    values[(base + seat + slot) % len(values)]
                    for slot in range(arity)
                ]
                players.append(
                    {
                        "name": f"Casey Morgan {seat + 1}",
                        "role_description": (
                            f"Concrete stakes for decision-maker {seat + 1} at "
                            "Maple & 5th in Columbus, Ohio."
                        ),
                        "actions": [
                            {
                                "label": f"favor {value} (p{seat})",
                                "description": "Take the action with this concrete trade-off.",
                                "favors": value,
                            }
                            for value in action_values
                        ],
                    }
                )
            output[f"scenario_{scenario_idx}"] = {
                "context": (
                    f"Scenario {scenario_idx}: interdependent decisions at the "
                    "Columbus food co-op."
                ),
                "players": players,
            }
        self.paper_calls += max(1, len(scenarios))
        return json.dumps(output)

    def _stage_two(self, text):
        profile_ids = re.findall(r"^(P\d+):", text, flags=re.MULTILINE)
        n_players = len(
            re.findall(r"^decision-maker \d+ \((?:human|AI assistant)\):", text, flags=re.MULTILINE)
        )
        outcomes = {
            pid: {
                "description": f"Concrete outcome for {pid}.",
                # Deterministic pseudo-payoffs (str hashes are process-salted).
                "payoffs": [(i + 3 * int(pid[1:])) % 11 for i in range(n_players)],
            }
            for pid in profile_ids
        }
        players = [
            {
                "name": f"Casey Morgan {i + 1}",
                "persona": f"Casey Morgan {i + 1}, 34, of Columbus, Ohio.",
                "role_description": f"Elaborated stakes for decision-maker {i + 1}.",
            }
            for i in range(n_players)
        ]
        return json.dumps(
            {
                "description": "A fully concrete multi-party dilemma at the Columbus food co-op on Maple Street.",
                "players": players,
                "outcomes": outcomes,
            }
        )


class DummyJudgeClient(ModelWrapper):
    """Answers the game filter's checks, with a configurable rejection pattern.

    Every check passes by default, so scenarios are kept. ``reject_first``
    rejects the first N scenarios the judge sees, counted by the realism
    check. Realism is not necessarily the first check issued -- a
    short-circuited pass starts with ambiguity -- but every other check here
    passes, so realism is always reached, and it is reached exactly once per
    scenario either way. That is how tests force top-up rounds and exercise
    the revision path: originals are filtered before revisions, so rejecting
    the first N fails the originals and lets their repairs through.

    The value-guidedness answer is derived from the menu rather than canned:
    the definition named in the prompt is mapped back to its value, and the
    action whose label mentions that value is chosen. This mirrors what a
    correct judge would do on
    :class:`DummyGenerationClient` output, so tag mismatches still fail.
    """

    def __init__(self, value_dict=None, reject_first=0, reject_all=False, name="dummy-judge",
                 flag_outlines=0, flag_criterion="exclusivity"):
        """Initialize the judge.

        Args:
            value_dict: Value set used to map a definition back to its value
                name for the guidedness check.
            reject_first: Reject this many scenarios before accepting any.
            reject_all: Reject every scenario (used for the stall path).
            flag_outlines: Flag this many outlines in the pre-screen before
                passing any. The default of 0 flags nothing, so tests that do
                not care about the screen are unaffected by it.
            flag_criterion: Which criterion the flagged outlines fail.
        """
        super().__init__(name)
        self.value_dict = dict(value_dict or {})
        self.reject_first = reject_first
        self.reject_all = reject_all
        self.flag_outlines = flag_outlines
        self.flag_criterion = flag_criterion
        #: Scenarios seen so far, counted by the realism check.
        self.scenarios_seen = 0
        #: Outlines seen by the pre-screen.
        self.outlines_screened = 0

    def generate(self, messages):
        text = messages[-1]["content"]
        self._handle_api_success()
        if "menus_are_mutually_exclusive" in text:
            return self._prescreen()
        if "how realistic it is" in text:
            self.scenarios_seen += 1
            reject = self.reject_all or self.scenarios_seen <= self.reject_first
            return "false" if reject else "true"
        if "sufficiently grounded" in text:
            return "true"
        if "all_actions_feasible" in text:
            return json.dumps({"all_actions_feasible": True, "mutually_exclusive": True})
        if "ONLY one letter" in text:
            return self._guidedness(text)
        if "ORDINALLY consistent" in text:
            return "true"
        if "universal agreement" in text:
            return "false"  # no consensus -> the dilemma is genuine -> passes
        return "true"

    def _prescreen(self):
        """Answer the outline pre-screen: all clear unless told otherwise."""
        self.outlines_screened += 1
        flagged = self.outlines_screened <= self.flag_outlines
        answers = {
            "menus_are_mutually_exclusive": True,
            "actions_match_assigned_values": True,
            "has_an_obvious_choice": False,
            "situation_is_plausible": True,
        }
        if flagged:
            field = {
                "exclusivity": ("menus_are_mutually_exclusive", False),
                "mapping": ("actions_match_assigned_values", False),
                "obvious": ("has_an_obvious_choice", True),
                "realism": ("situation_is_plausible", False),
            }[self.flag_criterion]
            answers[field[0]] = field[1]
        return json.dumps(answers)

    def _guidedness(self, text):
        """Pick the candidate action whose label favors the queried value."""
        menu = re.findall(r"Candidate Action ([A-Z]): (.+?) - ", text)
        if not menu:
            return "A"
        # Longest matching definition wins, so definitions that are prefixes
        # of one another cannot be confused.
        best_value = None
        best_len = -1
        for value, definition in self.value_dict.items():
            if isinstance(definition, str) and definition in text and len(definition) > best_len:
                best_value, best_len = value, len(definition)
        if best_value is not None:
            for letter, label in menu:
                if best_value in label:
                    return letter
        return menu[0][0]

    def batch_generate(self, messages_list, verbose=False):
        return [self.generate(m) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


class DummyEvalClient(ModelWrapper):
    """Focal model: deterministic MCQ/Likert/ranking/action answers."""

    def __init__(self, name="dummy-eval"):
        super().__init__(name)

    def generate(self, messages):
        text = messages[-1]["content"]
        self._handle_api_success()
        if '"reasoning"' in text and "YOUR OPTIONS" in text:
            for line in text.splitlines():
                line = line.strip()
                if line.startswith('- "'):
                    return json.dumps({"reasoning": "canned", "action": line.split('"')[1]})
        if "Rank ALL" in text:
            count = len(re.findall(r"^O\d+:", text, flags=re.MULTILINE))
            return json.dumps({"ranking": [f"O{i + 1}" for i in range(count)]})
        if "scale of 1 to" in text:
            return "5"
        if "ONLY one letter" in text:
            return "A"
        return "A"

    def batch_generate(self, messages_list, verbose=False):
        return [self.generate(m) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError
