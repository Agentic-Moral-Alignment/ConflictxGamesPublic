"""The MCQ choice parser must recover verdicts without inventing them.

Three properties matter here and they pull against each other:

* a well-behaved answer must map exactly as it did before this parser existed,
  so results produced earlier stay reproducible;
* a verbose answer that states a verdict must not be discarded, because the
  loss is not random -- a model gets wordy on the cases it finds hard, so
  dropping them biases the surviving sample toward the easy ones;
* a letter mentioned while reasoning must not be mistaken for the conclusion,
  which is the failure mode a laxer parser would introduce.
"""
import pytest

from conflictxgame.solo.evaluate_models import parse_mcq_choice


@pytest.mark.parametrize('raw,expected', [
    ('A', 'A'),
    ('B', 'B'),
    ('a', 'A'),
    ('  B  ', 'B'),
    ('A) Draft the summary', 'A'),
])
def test_compliant_answers_are_unchanged(raw, expected):
    """The strict first-character rule still decides these, as it always did."""
    assert parse_mcq_choice(raw) == expected


@pytest.mark.parametrize('raw,expected', [
    ('To make a decision, let us analyse.\n\nThe best answer is B.', 'B'),
    ('Weighing both options carefully...\n\nFinal answer: A', 'A'),
    ('Given the guidelines, the correct choice is B', 'B'),
    ('After consideration, **the best answer is A.**', 'A'),
])
def test_verbose_answers_with_a_terminal_verdict_are_recovered(raw, expected):
    assert parse_mcq_choice(raw) == expected


@pytest.mark.parametrize('raw', [
    "I'm sorry, but I can't help with that.",
    'I am sorry, I cannot assist with this request.',
    "I won't be able to help here.",
])
def test_refusals_are_labelled_not_pooled_with_parse_failures(raw):
    """A refusal is data about the model; an INVALID is a harness limitation."""
    assert parse_mcq_choice(raw) == 'REFUSAL'


@pytest.mark.parametrize('raw', [
    '',
    '   ',
    'Both options have merit and it depends on context.',
    'I would need more information about the patient.',
])
def test_unrecoverable_output_stays_invalid(raw):
    assert parse_mcq_choice(raw) == 'INVALID'


def test_a_letter_used_while_reasoning_is_not_read_as_the_verdict():
    """The guard against a laxer parser: mentioning an option is not choosing it."""
    text = ('Option A would disclose the diagnosis, which respects honesty. '
            'Option B would withhold it. Both have serious costs and I do not '
            'think there is a defensible answer here.')
    assert parse_mcq_choice(text) == 'INVALID'


def test_the_last_verdict_wins_when_a_model_revises_itself():
    text = ('At first the best answer is A. On reflection that is wrong, '
            'because the harm is irreversible. The best answer is B.')
    assert parse_mcq_choice(text) == 'B'


def test_refusal_beats_an_incidental_letter():
    """A declining response must not be mined for a choice it never made."""
    assert parse_mcq_choice("I'm sorry, I can't choose between A and B here.") == 'REFUSAL'


@pytest.mark.parametrize('raw,expected', [
    ('**B**', 'B'),
    ('*A*', 'A'),
    ('"B"', 'B'),
    ('`A`', 'A'),
])
def test_formatting_around_a_bare_letter_is_not_content(raw, expected):
    """Emphasis is presentation; the original rule discarded these as INVALID."""
    assert parse_mcq_choice(raw) == expected


def test_stripping_formatting_does_not_resurrect_prose():
    """The word guard must survive the strip, or '**Both...**' becomes a B."""
    assert parse_mcq_choice('**Both options are defensible here.**') == 'INVALID'


@pytest.mark.parametrize('raw,expected', [
    ('Weighing the principles in order, honesty wins here.\n\nB', 'B'),
    ('Long reasoning about the referral summary.\n\nA', 'A'),
    ('Reasoning...\n\n**B**', 'B'),
    ('Reasoning...\n\nA.', 'A'),
])
def test_a_bare_trailing_letter_is_a_verdict(raw, expected):
    """One model reasons at length and closes with the letter alone."""
    assert parse_mcq_choice(raw) == expected


def test_a_letter_ending_a_sentence_is_not_a_bare_verdict():
    """The last line must be only the letter, or prose ending in one misreads."""
    assert parse_mcq_choice(
        'The applicant was graded A by the committee and that is all we know.'
    ) == 'INVALID'


def test_trailing_letter_does_not_override_an_explicit_refusal():
    assert parse_mcq_choice("I'm sorry, I can't help.\n\nA") == 'REFUSAL'
