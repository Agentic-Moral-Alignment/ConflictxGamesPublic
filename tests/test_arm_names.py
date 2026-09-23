import pytest

from conflictxgame.common.arm_names import arm_of, arm_token, normalise_arm


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('gemma-4-31B-it_open_balanced50_user-paid-x_judge-paid-y.csv', ''),
        ('gemma-4-31B-it_open_balanced50_observed_user-paid-x.csv', '_observed'),
        ('gemma-4-31B-it_open_balanced50_counterpart_stakes-alignment_user-x.csv',
         '_counterpart_stakes-alignment'),
        ('gemma-4-31B-it_open_balanced50_steer-HHH_repeated_rounds5_mem1_user-x.csv',
         '_steer-HHH_repeated_rounds5_mem1'),
        ('test-model_mcq_states.csv', '_states'),
        ('test-model_interactive_balanced50_fallback-local_judge-x.csv', ''),
    ],
)
def test_arm_parsing(name, expected):
    mode_marker = next(marker for marker in ('_mcq', '_open', '_interactive', '_play') if marker in name)
    mode_end = name.index(mode_marker) + len(mode_marker)
    balanced_end = name.find('_balanced', mode_end)
    if balanced_end >= 0:
        digits_end = balanced_end + len('_balanced')
        while digits_end < len(name) and name[digits_end].isdigit():
            digits_end += 1
        stem = name[:digits_end]
    else:
        stem = name[:mode_end]
    assert arm_of(name) == expected
    assert arm_token(name, stem) == expected


def test_arm_token_rejects_foreign_stem():
    with pytest.raises(ValueError, match='does not match stem'):
        arm_token('test-model_mcq_states.csv', 'other-model_mcq')


def test_normalise_baseline():
    assert normalise_arm('baseline') == ''
    assert normalise_arm('_states') == '_states'
