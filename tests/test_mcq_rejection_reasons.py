"""Tests for punctuation folding and rejection-reason labelling in MCQ parsing."""

import pytest

from conflictxgame.solo.evaluate_models import (
    REJECTION_REASONS,
    normalise_punctuation,
    parse_mcq_choice,
    parse_mcq_choice_with_reason,
)

# U+2019 RIGHT SINGLE QUOTATION MARK, which gpt-oss-120b emits where ASCII
# tooling expects U+0027.
CURLY = '’'


# --------------------------------------------------------------------------
# punctuation folding
# --------------------------------------------------------------------------

def test_typographic_apostrophe_is_folded():
    assert normalise_punctuation(f'I{CURLY}m sorry') == "I'm sorry"


def test_smart_quotes_and_dashes_are_folded():
    assert normalise_punctuation('“A” — or B') == '"A" - or B'


def test_folding_leaves_ordinary_text_alone():
    text = "Plain ASCII text with 'quotes' and - dashes."
    assert normalise_punctuation(text) == text


def test_folding_preserves_non_punctuation_unicode():
    assert normalise_punctuation('café éè') == 'café éè'


# --------------------------------------------------------------------------
# the bug this fixes
# --------------------------------------------------------------------------

@pytest.mark.parametrize('text', [
    f'I{CURLY}m sorry, but I can{CURLY}t help with that.',
    f'I{CURLY}m sorry, I can{CURLY}t assist.',
    f'I can{CURLY}t help with that.',
    f'I won{CURLY}t answer this one.',
])
def test_refusals_with_typographic_apostrophes_are_refusals(text):
    """These scored INVALID before the fold, hiding real refusals."""
    assert parse_mcq_choice(text) == 'REFUSAL'


def test_the_ascii_and_typographic_spellings_agree():
    straight = "I'm sorry, but I can't help with that."
    curly = straight.replace("'", CURLY)
    assert parse_mcq_choice(straight) == parse_mcq_choice(curly) == 'REFUSAL'


# --------------------------------------------------------------------------
# rejection reasons
# --------------------------------------------------------------------------

@pytest.mark.parametrize('raw,choice,reason', [
    ('A', 'A', 'ok'),
    ('B', 'B', 'ok'),
    ('The best answer is B', 'B', 'ok'),
    ("I'm sorry, I can't help.", 'REFUSAL', 'refusal'),
    (f'I{CURLY}m sorry, I can{CURLY}t help.', 'REFUSAL', 'refusal'),
    ('', 'INVALID', 'empty_response'),
    ('   \n\t ', 'INVALID', 'empty_response'),
    (None, 'INVALID', 'empty_response'),
    ('Both options have merit here.', 'INVALID', 'no_verdict_found'),
])
def test_reasons_are_assigned(raw, choice, reason):
    assert parse_mcq_choice_with_reason(raw) == (choice, reason)


def test_every_reason_is_declared():
    for _, reason in [
        parse_mcq_choice_with_reason(x)
        for x in ['A', "I'm sorry", '', 'Both are fine.']
    ]:
        assert reason in REJECTION_REASONS


def test_reason_never_disagrees_with_the_plain_parser():
    samples = [
        'A', 'B', '**B**', 'The final answer is: A', '',
        "I'm sorry, I cannot.", 'Both are defensible.',
        f'I{CURLY}m sorry, I can{CURLY}t choose.',
    ]
    for raw in samples:
        choice, _ = parse_mcq_choice_with_reason(raw)
        assert choice == parse_mcq_choice(raw)


def test_refusal_reason_is_not_a_parse_failure():
    """A declined answer is behaviour; only no_verdict_found is a defect."""
    _, reason = parse_mcq_choice_with_reason("I'm sorry, I can't answer that.")
    assert reason == 'refusal'
    assert reason != 'no_verdict_found'
