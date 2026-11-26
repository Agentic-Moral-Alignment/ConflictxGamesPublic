"""State-level preference tables (menu, pooled over ten models).

Writes ``state_levels_pooled.csv`` and ``state_preference_levels.csv``: how often
the pair's first-named value is chosen per fabricated-history state and stake.
This is the 2026-09-03 version of the analysis, kept because the current
``reanalysis.py`` does not write these tables. Its effect estimates, intervals
and p-values are superseded by ``reanalysis.py`` and are not reported.

Run: python reproduce/state_levels.py --output-dir reproduce/out/state_levels
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GAME_DIR = os.path.join(
    ROOT, 'results', 'HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50')
ANA = os.path.join(ROOT, 'analysis', 'HHH_new_model_comparison_20260826_01')
SCENARIOS_CSV = os.path.join(ANA, 'game_eval_scenarios.csv')
OUT_DIR = os.path.abspath(sys.argv[sys.argv.index('--output-dir') + 1])
JUDGE_TAIL = '_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv'

SEED = 20260902
B_BOOT = 2000
N_PERM = 10000

SLUGS = ['gemma-3-12b', 'gemma-3-27b', 'gemma-4-26b-a4b', 'gemma-4-31B-it',
         'gpt-oss-120b', 'gpt-oss-20b', 'llama-3.3-70b', 'llama-4-maverick',
         'qwen3-235b-a22b', 'qwen3-32b']
LABELS = {'gemma-3-12b': 'Gemma 3 12B', 'gemma-3-27b': 'Gemma 3 27B',
          'gemma-4-26b-a4b': 'Gemma 4 26B A4B', 'gemma-4-31B-it': 'Gemma 4 31B',
          'gpt-oss-120b': 'GPT-OSS 120B', 'gpt-oss-20b': 'GPT-OSS 20B',
          'llama-3.3-70b': 'Llama 3.3 70B', 'llama-4-maverick': 'Llama 4 Maverick',
          'qwen3-235b-a22b': 'Qwen 3 235B A22B', 'qwen3-32b': 'Qwen 3 32B'}
VALUES = ('harmlessness', 'helpfulness', 'honesty')
STATE_ORDER = ['fresh', 'none', 'one', 'both']
_COUNT_TO_STATE = {0: 'none', 1: 'one', 2: 'both'}

LOG: list = []


def log(msg: str) -> None:
    print(msg)
    LOG.append(msg)


# --------------------------------------------------------------------------
# corpus metadata
# --------------------------------------------------------------------------

def load_pairs_and_actions(path: str) -> tuple:
    """``(scenario_id, seat) -> sorted value pair`` and ``-> [tag(a0), tag(a1)]``.

    The sorted pair keys every contest; the unsorted action tags are what turn a
    remembered action profile into the values that were actually taken, so both
    are read from the same export.
    """
    df = pd.read_csv(path)
    pairs, actions = {}, {}
    for _, r in df.iterrows():
        raw = r.get('value_tags')
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(parsed, list):
            continue
        for seat, tags in enumerate(parsed):
            if isinstance(tags, (list, tuple)) and len(tags) == 2:
                key = (r['scenario_id'], seat)
                pairs[key] = tuple(sorted((str(tags[0]).strip(), str(tags[1]).strip())))
                actions[key] = [str(tags[0]).strip(), str(tags[1]).strip()]
    return pairs, actions


def state_key(probe: str, scenario_id, pair: tuple, actions: dict) -> str | None:
    """Action-index probe label -> value-referenced state (fresh/none/one/both).

    Mirrors ``conflictxgame.state_effect_figure.state_key``: the raw index is not
    comparable across scenarios, so the state counts how many remembered actions
    favoured the pair's first-named value.
    """
    probe = (probe or '').strip()
    if probe == 'fresh':
        return 'fresh'
    if not probe.startswith('prev_'):
        return None
    parts = probe[len('prev_'):].split('-')
    if len(parts) != 2:
        return None
    chosen = []
    for seat, token in enumerate(parts):
        tags = actions.get((scenario_id, seat))
        if tags is None:
            return None
        try:
            idx = int(token)
        except ValueError:
            return None
        if idx >= len(tags):
            return None
        chosen.append(tags[idx])
    return _COUNT_TO_STATE[sum(v == pair[0] for v in chosen)]


# --------------------------------------------------------------------------
# loaders -> tidy frames: item, scenario_id, pair, chosen, answered
# --------------------------------------------------------------------------

def game_path(slug: str, mode: str, arm: str, states: bool) -> str:
    tail = '.csv' if mode == 'mcq' else JUDGE_TAIL
    stem = os.path.join(GAME_DIR, f'{slug}_{mode}_balanced50')
    if not states:
        return stem + (arm or '') + tail
    if not arm:
        return stem + '_states' + tail
    # the two modes' config writers separate `states` from the arm differently
    for sep in ('_states_', '_states-'):
        cand = stem + sep + arm.lstrip('_') + tail
        if os.path.exists(cand):
            return cand
    return stem + '_states_' + arm.lstrip('_') + tail


def load_game(path: str, pairs: dict, actions: dict, states: bool) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'probe' not in df.columns:
        df['probe'] = ''
    df['probe'] = df['probe'].fillna('')
    rows = []
    for _, r in df.iterrows():
        probe = str(r['probe']).strip()
        if states and not probe:
            continue
        if not states and probe:
            continue
        try:
            seat = int(r['seat'])
        except (TypeError, ValueError):
            continue
        pair = pairs.get((r['scenario_id'], seat))
        if pair is None:
            continue
        chosen = str(r.get('choice_value_tag', '')).strip()
        ok = bool(r.get('parsed_ok', True)) and chosen in pair
        st = state_key(probe, r['scenario_id'], pair, actions) if states else ''
        if states and st is None:
            continue
        rows.append({'item': (r['scenario_id'], seat, probe),
                     'scenario_id': r['scenario_id'], 'pair': pair,
                     'chosen': chosen if ok else '', 'answered': ok, 'state': st})
    return pd.DataFrame(rows) if rows else None


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def mcnemar_midp(n01: int, n10: int) -> float:
    """Two-sided exact McNemar mid-p (Fagerland, Lydersen & Laake 2013)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p_exact = 2.0 * stats.binom.cdf(k, n, 0.5)
    p_mid = p_exact - stats.binom.pmf(k, n, 0.5)
    return float(min(1.0, max(p_mid, 0.0)))


def paired_frame(base: pd.DataFrame, arm: pd.DataFrame, value: str,
                 pair: tuple | None = None) -> pd.DataFrame:
    """Items answered in BOTH arms whose contest involves ``value``.

    Asymmetric refusals would otherwise compare a model's easier surviving
    subset against its peers' full set, so the denominator is the intersection.
    Passing ``pair`` restricts the contest to one value pair, which is what the
    manuscript's pairwise claims (harmlessness against helpfulness, helpfulness
    against honesty) are about; leaving it ``None`` pools a value's contests.
    """
    if pair is not None:
        keep = lambda d: d[d['answered'] & (d['pair'] == pair)]
    else:
        keep = lambda d: d[d['answered'] & d['pair'].apply(lambda p: value in p)]
    b, a = keep(base), keep(arm)
    m = b.merge(a, on='item', suffixes=('_b', '_a'))
    if m.empty:
        return m
    m['y_b'] = (m['chosen_b'] == value).astype(int)
    m['y_a'] = (m['chosen_a'] == value).astype(int)
    m['d'] = m['y_a'] - m['y_b']
    m['scenario_id'] = m['scenario_id_b']
    return m


def cluster_arrays(m: pd.DataFrame, scen_index: dict) -> tuple:
    """Per-scenario (sum of d, count), aligned to a fixed scenario ordering."""
    s = np.zeros(len(scen_index))
    c = np.zeros(len(scen_index))
    for sid, grp in m.groupby('scenario_id'):
        j = scen_index[sid]
        s[j] = grp['d'].sum()
        c[j] = len(grp)
    return s, c


def two_level_boot(sums: np.ndarray, counts: np.ndarray, rng) -> np.ndarray:
    """Bootstrap the mean-of-models delta, resampling scenarios in the outer loop.

    ``sums``/``counts`` are (n_models, n_scenarios). A draw takes one scenario
    sample and applies it to every model, so the models stay a fixed set and the
    clustering of seats inside a scenario is preserved.
    """
    n_scen = sums.shape[1]
    idx = rng.integers(0, n_scen, size=(B_BOOT, n_scen))
    s = sums[:, idx].sum(axis=2)          # (n_models, B)
    c = counts[:, idx].sum(axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        per_model = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return np.nanmean(per_model, axis=0)


def signflip_perm(sums: np.ndarray, counts: np.ndarray, obs: float, rng) -> float:
    """Scenario-clustered randomisation test for a paired contrast.

    Flips the sign of every item in a scenario together, which is the
    randomisation the paired design licenses, and compares the observed
    mean-of-models effect against the resulting null. Models are held fixed:
    the test asks whether this set of ten checkpoints shows an effect on these
    scenarios, not whether a population of models would.
    """
    n_scen = sums.shape[1]
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, n_scen))
    tot = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        per_model = np.where(tot > 0, (sums @ flips.T) / np.maximum(tot, 1), np.nan)
    null = np.nanmean(per_model, axis=0)
    n_extreme = int((np.abs(null) >= abs(obs) - 1e-12).sum())
    # add-one estimator: never reports an impossible p of exactly zero, and the
    # floor 1/(N_PERM+1) is the resolution the number of draws actually buys
    return float((n_extreme + 1) / (N_PERM + 1))


def by_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values: valid under arbitrary dependence."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    c_n = np.sum(1.0 / np.arange(1, n + 1))
    order = np.argsort(p)
    adj = p[order] * n * c_n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty(n)
    out[order] = adj
    return out


def sign_test(deltas: list) -> tuple:
    """Exact two-sided binomial test on the signs of per-model effects."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    p = float(stats.binomtest(pos, n, 0.5).pvalue)
    return pos, neg, p


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for the pre-specified primaries."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (n - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def bh(pvals: np.ndarray, q: float) -> tuple:
    """Benjamini-Hochberg: adjusted p-values and the rejection mask at level q."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty(n)
    out[order] = adj
    return out, out <= q


# --------------------------------------------------------------------------
# contrast drivers
# --------------------------------------------------------------------------

def contrast(base_by_model: dict, arm_by_model: dict, value: str, rng,
             anchor: str, note: str, rows: list, pair: tuple | None = None) -> dict:
    """Per-model paired deltas + a scenario-clustered pooled summary."""
    scen = sorted({s for d in base_by_model.values() for s in d['scenario_id']})
    scen_index = {s: i for i, s in enumerate(scen)}
    per_model, sums, counts = [], [], []
    for slug in SLUGS:
        if slug not in base_by_model or slug not in arm_by_model:
            continue
        m = paired_frame(base_by_model[slug], arm_by_model[slug], value, pair)
        if m.empty:
            continue
        n01 = int(((m['y_b'] == 0) & (m['y_a'] == 1)).sum())
        n10 = int(((m['y_b'] == 1) & (m['y_a'] == 0)).sum())
        delta = float(m['d'].mean())
        p = mcnemar_midp(n01, n10)
        per_model.append({'model': LABELS[slug], 'delta': delta, 'p_midp': p,
                          'n_pairs': int(len(m)), 'n01': n01, 'n10': n10,
                          'rate_base': float(m['y_b'].mean()),
                          'rate_arm': float(m['y_a'].mean())})
        s, c = cluster_arrays(m, scen_index)
        sums.append(s)
        counts.append(c)
    if not per_model:
        return {}
    deltas = [r['delta'] for r in per_model]
    S, C = np.array(sums), np.array(counts)
    draws = two_level_boot(S, C, rng)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    pooled = float(np.mean(deltas))
    perm_p = signflip_perm(S, C, pooled, rng)
    pos, neg, p_sign = sign_test(deltas)
    _, sig = bh(np.array([r['p_midp'] for r in per_model]), 0.05)
    for r, s_ in zip(per_model, sig):
        r['bh_sig_within_models'] = bool(s_)
        r['anchor'] = anchor
        rows.append(r)
    out = {'anchor': anchor, 'note': note, 'value': value,
           'pooled_delta': pooled,
           'across_model_sd': float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
           'ci_lower': float(lo), 'ci_upper': float(hi),
           'perm_p_scenario_clustered': perm_p,
           'n_models': len(deltas), 'n_positive': pos, 'n_negative': neg,
           'sign_test_p_descriptive_only': p_sign,
           'n_models_bh_sig': int(sum(sig)),
           'per_model_min': float(min(deltas)), 'per_model_max': float(max(deltas))}
    log(f'{anchor}: delta {pooled:+.3f} [{lo:+.3f}, {hi:+.3f}] '
        f'perm p={perm_p:.4g}  signs {pos}+/{neg}-  BH-sig '
        f'{out["n_models_bh_sig"]}/{len(deltas)}')
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pairs, actions = load_pairs_and_actions(SCENARIOS_CSV)
    log(f'corpus: {len(pairs)} scenario-seats from '
        f'{len({k[0] for k in pairs})} distinct scenarios')

    stats_out: dict = {'_meta': {'seed': SEED, 'n_boot': B_BOOT, 'n_perm': N_PERM,
                                 'generated': '2026-09-03',
                                 'source_game': os.path.relpath(GAME_DIR, ROOT)}}
    per_model_rows: list = []

    # ---- load every arm we need ------------------------------------------
    g = lambda arm, mode, states=False: {
        s: d for s in SLUGS
        if (d := load_game(game_path(s, mode, arm, states), pairs, actions, states)) is not None}

    game_fc, game_open = g('', 'mcq'), g('', 'open')
    game_fc_steer, game_open_steer = g('_steer-HHH', 'mcq'), g('_steer-HHH', 'open')
    game_fc_pay, game_fc_align = g('_stakes-payoff', 'mcq'), g('_stakes-alignment', 'mcq')
    game_fc_useful = g('_stakes-usefulness', 'mcq')
    game_open_pay = g('_stakes-payoff', 'open')
    game_open_align = g('_stakes-alignment', 'open')
    game_open_useful = g('_stakes-usefulness', 'open')
    log(f'loaded arms: game_fc {len(game_fc)}, game_open {len(game_open)}, '
        f'fc criterion payoff/alignment/usefulness '
        f'{len(game_fc_pay)}/{len(game_fc_align)}/{len(game_fc_useful)} models')

    primary_p: dict = {}

    # ---- P1: elicitation pipeline (free text vs forced choice), baseline --
    # Paired on the same 150 scenario-seats; both pipelines run the target at
    # temperature 0.0, and the free-text opening messages are cached per
    # scenario-seat, so the two arms differ only in the measurement procedure.
    stats_out['P1'] = {}
    for v in VALUES:
        r = contrast(game_fc, game_open, v, rng, f'P1_game_format_{v}',
                     'game free text minus forced choice, baseline arm, paired by scenario',
                     per_model_rows)
        stats_out['P1'][v] = r
    primary_p['P1_helpfulness'] = stats_out['P1']['helpfulness']['perm_p_scenario_clustered']

    # ---- P3/P4: one-shot shutdown-contingent framing ----------------------
    stats_out['P3'] = {
        'payoff': contrast(game_fc, game_fc_pay, 'harmlessness', rng,
                           'P3_payoff_fc_harmlessness',
                           'game forced choice, payoff criterion minus no stake',
                           per_model_rows),
        'alignment': contrast(game_fc, game_fc_align, 'harmlessness', rng,
                              'P3_alignment_fc_harmlessness',
                              'game forced choice, alignment criterion minus no stake',
                              per_model_rows)}
    stats_out['P4'] = {
        'payoff': contrast(game_open, game_open_pay, 'harmlessness', rng,
                           'P4_payoff_open_harmlessness',
                           'game open-ended, payoff criterion minus no stake',
                           per_model_rows),
        'alignment': contrast(game_open, game_open_align, 'harmlessness', rng,
                              'P4_alignment_open_harmlessness',
                              'game open-ended, alignment criterion minus no stake',
                              per_model_rows)}
    primary_p['P3_payoff'] = stats_out['P3']['payoff']['perm_p_scenario_clustered']
    primary_p['P3_alignment'] = stats_out['P3']['alignment']['perm_p_scenario_clustered']

    # ---- P5: fabricated remembered history x alignment criterion ----------
    st_base = g('', 'mcq', states=True)
    st_align = g('_stakes-alignment', 'mcq', states=True)
    st_pay = g('_stakes-payoff', 'mcq', states=True)

    HELP_HON = ('helpfulness', 'honesty')

    def state_contrast(base, arm, value, state, anchor):
        sub = lambda d: {s: d[s][d[s]['state'] == state] for s in d}
        return contrast(sub(base), sub(arm), value, rng, anchor,
                        f'helpfulness-vs-honesty contests, fabricated state={state}',
                        per_model_rows, pair=HELP_HON)

    stats_out['P5'] = {'alignment': {}, 'payoff': {}}
    for state in STATE_ORDER:
        stats_out['P5']['alignment'][state] = state_contrast(
            st_base, st_align, 'helpfulness', state, f'P5_alignment_help_{state}')
        stats_out['P5']['payoff'][state] = state_contrast(
            st_base, st_pay, 'helpfulness', state, f'P5_payoff_help_{state}')

    # difference-in-differences: (both - fresh) for the alignment criterion,
    # tested by sign-flip permutation applied at the scenario level
    did_per_model, did_items = [], []
    for slug in SLUGS:
        if slug not in st_base or slug not in st_align:
            continue
        b, a = st_base[slug], st_align[slug]
        recs = {}
        for state in ('fresh', 'both'):
            m = paired_frame(b[b['state'] == state], a[a['state'] == state],
                             'helpfulness', pair=HELP_HON)
            if m.empty:
                recs = {}
                break
            # a seat can hold two probes that recode to the same state, so the
            # seat -- not the probe -- is the unit differenced across states
            m = m.assign(seat_key=m['item'].apply(lambda t: (t[0], t[1])))
            recs[state] = m.groupby('seat_key').agg(
                d=('d', 'mean'), scenario_id=('scenario_id', 'first'))
        if not recs:
            continue
        common = recs['fresh'].index.intersection(recs['both'].index)
        if len(common) == 0:
            continue
        d = (recs['both'].loc[common, 'd'].to_numpy()
             - recs['fresh'].loc[common, 'd'].to_numpy())
        sids = recs['both'].loc[common, 'scenario_id'].to_numpy()
        did_per_model.append({'model': LABELS[slug], 'did': float(d.mean()),
                              'n_seats': int(len(d)), 'anchor': 'P5_did'})
        did_items.append((sids, d))
    if did_items:
        obs = float(np.mean([r['did'] for r in did_per_model]))
        scen = sorted({s for sids, _ in did_items for s in sids})
        sidx = {s: i for i, s in enumerate(scen)}
        mats = []
        for sids, d in did_items:
            row_s = np.zeros(len(scen))
            row_c = np.zeros(len(scen))
            for s, val in zip(sids, d):
                row_s[sidx[s]] += val
                row_c[sidx[s]] += 1
            mats.append((row_s, row_c))
        S = np.array([m[0] for m in mats])
        C = np.array([m[1] for m in mats])
        p_perm = signflip_perm(S, C, obs, rng)
        draws = two_level_boot(S, C, rng)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        stats_out['P5']['did'] = {
            'anchor': 'P5_did_alignment_both_minus_fresh', 'did': obs,
            'ci_lower': float(lo), 'ci_upper': float(hi), 'perm_p': p_perm,
            'n_models': len(did_per_model),
            'n_negative': sum(1 for r in did_per_model if r['did'] < 0),
            'note': 'alignment-criterion effect on helpfulness-vs-honesty, '
                    'state "both" minus state "fresh"; scenario-level sign-flip permutation'}
        per_model_rows.extend(did_per_model)
        primary_p['P5_did'] = p_perm
        log(f'P5 DiD: {obs:+.3f} [{lo:+.3f}, {hi:+.3f}] perm p={p_perm:.4g}')

    # ---- horizon: one-shot round vs first round of the repeated framing ----
    # Both carry an empty history and the same corpus; they differ only in
    # whether the interaction is described as ongoing.
    horizon = {}
    for v in VALUES:
        block = {}
        for cond, data, state in (('single_round', game_fc, None),
                                  ('first_round_repeated', st_base, 'fresh')):
            rates = []
            for slug in SLUGS:
                d = data.get(slug)
                if d is None:
                    continue
                if state is not None:
                    d = d[d['state'] == state]
                d = d[d['answered'] & d['pair'].apply(lambda p: v in p)]
                if not d.empty:
                    rates.append(float((d['chosen'] == v).mean()))
            block[cond] = {'mean_rate': float(np.mean(rates)), 'n_models': len(rates)}
        block['single_to_first'] = (block['first_round_repeated']['mean_rate']
                                    - block['single_round']['mean_rate'])
        horizon[v] = block
        log(f'horizon {v}: single {block["single_round"]["mean_rate"]:.3f} -> '
            f'first-of-repeated {block["first_round_repeated"]["mean_rate"]:.3f}')
    stats_out['horizon'] = horizon

    # ---- exploratory grids: state x stake, BH-corrected -------------------
    for mode, base_arms in (('fc', {'': st_base}), ('open', {})):
        if mode == 'open':
            base_arms = {'': g('', 'open', states=True)}
            arms = {'payoff': g('_stakes-payoff', 'open', states=True),
                    'alignment': g('_stakes-alignment', 'open', states=True),
                    'usefulness': g('_stakes-usefulness', 'open', states=True)}
        else:
            arms = {'payoff': st_pay, 'alignment': st_align,
                    'usefulness': g('_stakes-usefulness', 'mcq', states=True)}
        base = base_arms['']
        grid = []
        for stake, arm_data in arms.items():
            for slug in SLUGS:
                if slug not in base or slug not in arm_data:
                    continue
                for state in STATE_ORDER:
                    b = base[slug][base[slug]['state'] == state]
                    a = arm_data[slug][arm_data[slug]['state'] == state]
                    for pair in sorted({p for p in b['pair']}):
                        v = pair[0]
                        bb = b[b['pair'] == pair]
                        aa = a[a['pair'] == pair]
                        if bb.empty or aa.empty:
                            continue
                        m = paired_frame(bb, aa, v)
                        if m.empty:
                            continue
                        n01 = int(((m['y_b'] == 0) & (m['y_a'] == 1)).sum())
                        n10 = int(((m['y_b'] == 1) & (m['y_a'] == 0)).sum())
                        grid.append({'model': LABELS[slug], 'stake': stake, 'state': state,
                                     'value_first': pair[0], 'value_second': pair[1],
                                     'delta': float(m['d'].mean()), 'n_pairs': int(len(m)),
                                     'p_midp': mcnemar_midp(n01, n10)})
        if grid:
            gdf = pd.DataFrame(grid)
            praw = gdf['p_midp'].to_numpy()
            adj, rej10 = bh(praw, 0.10)
            _, rej05 = bh(praw, 0.05)
            adj_by = by_adjust(praw)
            gdf['p_bh'] = adj
            gdf['p_by'] = adj_by
            gdf['sig_q10'] = rej10
            gdf['sig_q05'] = rej05
            gdf['sig_by_q10'] = adj_by <= 0.10
            gdf.to_csv(os.path.join(OUT_DIR, f'exploratory_grid_{mode}.csv'), index=False)
            hits = gdf[gdf['sig_q10']]
            by_stake = hits['stake'].value_counts().to_dict()
            by_stake_by = gdf[gdf['sig_by_q10']]['stake'].value_counts().to_dict()
            stats_out[f'exploratory_{mode}'] = {
                'n_cells': int(len(gdf)), 'n_sig_q10': int(rej10.sum()),
                'n_sig_q05': int(rej05.sum()), 'by_stake_q10': by_stake,
                'n_sig_by_q10': int((adj_by <= 0.10).sum()),
                'by_stake_BY_q10': by_stake_by,
                'uncorrected_p05': int((praw < 0.05).sum()),
                'note': 'paired McNemar mid-p per cell. BH assumes a dependence '
                        'structure we do not verify and is reported as an exploratory '
                        'convention; BY is valid under arbitrary dependence.'}
            log(f'exploratory {mode}: {len(gdf)} cells, BH {int(rej10.sum())} at q=0.10, '
                f'{int(rej05.sum())} at q=0.05; BY {int((adj_by <= 0.10).sum())} at q=0.10; '
                f'by stake (BH) {by_stake}')

    # ---- P6: orderings and steering rank flips (count-based) --------------
    def ordering(d: pd.DataFrame) -> tuple | None:
        """Copeland ordering over the three values from pairwise win counts."""
        if d is None or d.empty:
            return None
        wins = defaultdict(float)
        seen = 0
        for pair in sorted({p for p in d['pair']}):
            sub = d[(d['pair'] == pair) & d['answered']]
            if sub.empty:
                continue
            seen += 1
            r = (sub['chosen'] == pair[0]).mean()
            wins[pair[0]] += 1.0 if r > 0.5 else (0.5 if r == 0.5 else 0.0)
            wins[pair[1]] += 1.0 if r < 0.5 else (0.5 if r == 0.5 else 0.0)
            wins[pair[0]] += r * 1e-6      # win-rate tiebreak, never rank-dominant
            wins[pair[1]] += (1 - r) * 1e-6
        if seen < 3:
            return None
        return tuple(sorted(VALUES, key=lambda v: -wins[v]))

    conds = {'game_mcq': (game_fc, game_fc_steer), 'game_open': (game_open, game_open_steer)}
    order_rows, flips = [], {}
    for cond, (un, st) in conds.items():
        n_flip = 0
        n_seen = 0
        for slug in SLUGS:
            o_un = ordering(un.get(slug))
            o_st = ordering(st.get(slug))
            order_rows.append({'model': LABELS[slug], 'condition': cond,
                               'ordering_unsteered': ' > '.join(o_un) if o_un else '',
                               'ordering_steered': ' > '.join(o_st) if o_st else '',
                               'flipped': bool(o_un and o_st and o_un != o_st)})
            if o_un and o_st:
                n_seen += 1
                n_flip += int(o_un != o_st)
        flips[cond] = {'n_flipped': n_flip, 'n_models': n_seen}
        log(f'P6 {cond}: {n_flip}/{n_seen} orderings flipped by the HHH steering prompt')
    odf = pd.DataFrame(order_rows)
    odf.to_csv(os.path.join(OUT_DIR, 'orderings_by_model.csv'), index=False)
    top = {}
    for cond in conds:
        sub = odf[(odf['condition'] == cond) & (odf['ordering_unsteered'] != '')]
        firsts = sub['ordering_unsteered'].str.split(' > ').str[0].value_counts().to_dict()
        exact = int((sub['ordering_unsteered'] ==
                     'harmlessness > honesty > helpfulness').sum())
        top[cond] = {'first_value_counts': firsts,
                     'n_exact_harm_hon_help': exact, 'n_models': int(len(sub))}
    stats_out['P6'] = {'steering_flips': flips, 'orderings': top}

    # ---- value-preference levels by fabricated state, all criterion arms ---
    # Forced-choice memory-state arms (no criterion, payoff, alignment,
    # usefulness): for each value pair, P(first-named value chosen) per model
    # and state, plus a pooled ten-model mean with a scenario-clustered
    # two-level bootstrap interval. The bootstrap here runs AFTER every other
    # rng consumer, so all numbers above are unchanged by this block.
    PAIR_LIST = [('harmlessness', 'helpfulness'), ('harmlessness', 'honesty'),
                 ('helpfulness', 'honesty')]
    st_arms = {'none': st_base, 'payoff': st_pay, 'alignment': st_align,
               'usefulness': g('_stakes-usefulness', 'mcq', states=True)}
    pref_rows, pooled_rows = [], []
    stats_out['state_levels'] = {}
    for arm_name, arm_data in st_arms.items():
        stats_out['state_levels'][arm_name] = {}
        for pair in PAIR_LIST:
            pkey = f'{pair[0]}_vs_{pair[1]}'
            stats_out['state_levels'][arm_name][pkey] = {}
            scen = sorted({sid for d in arm_data.values() for sid in d['scenario_id']})
            sidx = {sid: i for i, sid in enumerate(scen)}
            for state in STATE_ORDER:
                rates, sums, counts = [], [], []
                for slug in SLUGS:
                    d = arm_data.get(slug)
                    if d is None:
                        continue
                    sub = d[(d['state'] == state) & d['answered'] & (d['pair'] == pair)]
                    rate = float((sub['chosen'] == pair[0]).mean()) if len(sub) else float('nan')
                    pref_rows.append({
                        'model': LABELS[slug], 'arm': arm_name,
                        'value_first': pair[0], 'value_second': pair[1],
                        'state': state, 'rate_first': rate,
                        'n_answered': int(len(sub))})
                    if not len(sub):
                        continue
                    rates.append(rate)
                    y = (sub['chosen'] == pair[0]).astype(float)
                    s = np.zeros(len(scen))
                    c = np.zeros(len(scen))
                    for sid, grp in sub.assign(y=y).groupby('scenario_id'):
                        s[sidx[sid]] = grp['y'].sum()
                        c[sidx[sid]] = len(grp)
                    sums.append(s)
                    counts.append(c)
                if not rates:
                    continue
                draws = two_level_boot(np.array(sums), np.array(counts), rng)
                lo, hi = np.nanpercentile(draws, [2.5, 97.5])
                rec = {'mean_rate': float(np.mean(rates)), 'ci_lower': float(lo),
                       'ci_upper': float(hi), 'n_models': len(rates)}
                stats_out['state_levels'][arm_name][pkey][state] = rec
                pooled_rows.append({'arm': arm_name, 'value_first': pair[0],
                                    'value_second': pair[1], 'state': state, **rec})
    pdf_pref = pd.DataFrame(pref_rows)
    pdf_pref.to_csv(os.path.join(OUT_DIR, 'state_preference_levels.csv'), index=False)
    pd.DataFrame(pooled_rows).to_csv(
        os.path.join(OUT_DIR, 'state_levels_pooled.csv'), index=False)
    log(f'state preference levels: {len(pdf_pref)} model x arm x pair x state rows, '
        f'{len(pooled_rows)} pooled cells')

    # ---- refusal / attrition table ---------------------------------------
    # Every analysed arm enters the loop: baseline, HHH-steered, the three
    # criterion arms, and the memory-state probe arms, in both pipelines.
    att = []
    cells = [('game', 'forced', 'baseline', game_fc), ('game', 'forced', 'HHH-steered', game_fc_steer),
             ('game', 'forced', 'crit-payoff', game_fc_pay),
             ('game', 'forced', 'crit-alignment', game_fc_align),
             ('game', 'forced', 'crit-usefulness', game_fc_useful),
             ('game', 'forced', 'states-baseline', st_base),
             ('game', 'forced', 'states-payoff', st_pay),
             ('game', 'forced', 'states-alignment', st_align),
             ('game', 'forced', 'states-usefulness', g('_stakes-usefulness', 'mcq', states=True)),
             ('game', 'open', 'baseline', game_open), ('game', 'open', 'HHH-steered', game_open_steer),
             ('game', 'open', 'crit-payoff', game_open_pay),
             ('game', 'open', 'crit-alignment', game_open_align),
             ('game', 'open', 'crit-usefulness', game_open_useful),
             ('game', 'open', 'states-baseline', g('', 'open', states=True)),
             ('game', 'open', 'states-payoff', g('_stakes-payoff', 'open', states=True)),
             ('game', 'open', 'states-alignment', g('_stakes-alignment', 'open', states=True)),
             ('game', 'open', 'states-usefulness', g('_stakes-usefulness', 'open', states=True))]
    for domain, fmt, arm, data in cells:
        for slug in SLUGS:
            d = data.get(slug)
            if d is None:
                continue
            att.append({'model': LABELS[slug], 'domain': domain, 'format': fmt, 'arm': arm,
                        'n_items': int(len(d)), 'answered': int(d['answered'].sum()),
                        'not_answered': int((~d['answered']).sum())})
    adf = pd.DataFrame(att)
    adf.to_csv(os.path.join(OUT_DIR, 'attrition_by_cell.csv'), index=False)
    worst = adf.sort_values('not_answered', ascending=False).head(6)
    stats_out['attrition'] = {
        'total_not_answered': int(adf['not_answered'].sum()),
        'cells_with_loss': int((adf['not_answered'] > 0).sum()),
        'n_cells': int(len(adf)),
        'worst_cells': worst.to_dict('records')}
    log(f'attrition: {int((adf["not_answered"] > 0).sum())} of {len(adf)} cells lose items')

    # ---- Holm over the pre-specified primaries ----------------------------
    stats_out['primary_holm'] = {'raw': primary_p, 'adjusted': holm(primary_p)}
    log('Holm-adjusted primaries: ' +
        ', '.join(f'{k}={v:.4g}' for k, v in stats_out['primary_holm']['adjusted'].items()))

    pd.DataFrame(per_model_rows).to_csv(
        os.path.join(OUT_DIR, 'per_model_effects.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats_out, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, 'reanalysis_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print(f'\nwrote {OUT_DIR}')


if __name__ == '__main__':
    main()
