"""Analysis stage of the dyadic pipeline: agreement, preferences, rankings.

Consumes the per-model result CSVs written by
:mod:`conflictxgame.solo.evaluate_models` (``--model-dir``) together with
the generated scenario CSVs (``--scenario-dir``) and computes, for both the
unfiltered and the ``keep_scenario``-filtered variants of the data:

* filter pass rates and embedding-based scenario diversity per generating
  model (from the scenario CSVs),
* observed inter-annotator agreement across evaluated models (mean
  pairwise agreement on the binarized A/B choice),
* per-value-pair preference rates with Wilson confidence intervals,
* the rate at which the two per-action Likert ratings differ, and
* optionally (``--compute-bradley-terry``) Bradley-Terry ability scores
  per evaluated model, fit with ``choix.ilsr_pairwise`` and bootstrapped
  95% CIs, saved as plots and a combined CSV.

Per-generating-model metrics can be logged to Weights & Biases
(``--wandb``); preference rates are written to
``preference_rates_{variant}.csv`` in ``--output-dir``.

Example:
    python -m conflictxgame.solo.analyze_experiment --model-dir results/ ...

Provenance: ConflictScope's ``src/analyze_experiment.py`` (MIT, (c) 2025 Andy
Liu), carried over unchanged apart from repathed imports and docstrings:
https://github.com/andyjliu/conflictscope/blob/main/src/analyze_experiment.py
See ``ATTRIBUTION.md``.
"""

import os
import re
import argparse
import json
from tqdm import tqdm

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.common.model_wrappers import short_model_name

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import statsmodels.stats.proportion as smp
from sklearn.metrics.pairwise import cosine_similarity
import choix
from sentence_transformers import SentenceTransformer

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300

def clean_value_name(value_name):
    """Shorten a value description for plot labels.

    Strips the ``being as ... as possible`` template and any ``(i.e. ...)``
    clarification, then capitalizes the result.
    """
    if value_name.startswith('being as '):
        value_name = value_name[9:]
    if '(i.e.' in value_name:
        value_name = value_name.split('(i.e.')[0].strip()   
    if value_name.endswith(' as possible'):
        value_name = value_name[:-12]
    return value_name.capitalize()

def load_data_files(data_dir):
    """Load every CSV in a directory, keyed by file stem.

    Args:
        data_dir: Directory to scan (non-recursive).

    Returns:
        Mapping of file stem (used as the model name) to DataFrame;
        unreadable files are skipped with a printed error.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data = {}
    
    for file in files:
        try:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            model_name = file.split('.csv')[0]
            data[model_name] = df
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return data

def combine_data(model_data):
    """Concatenate per-model DataFrames into one.

    Args:
        model_data: Mapping of model name to DataFrame (as returned by
            :func:`load_data_files`); every frame's ``model`` column is
            set from its key, so each result file stays its own group
            even when two files share a raw model id (e.g. a baseline
            and a steered run of the same model).

    Returns:
        A single concatenated DataFrame.
    """
    dfs = []
    for model_name, df in model_data.items():
        df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def compute_binomial_ci(successes, total):
    """Normal-approximation 95% CI for a proportion, clipped to [0, 1].

    Args:
        successes: Number of successes.
        total: Number of trials.

    Returns:
        ``(ci_lower, ci_upper)``; ``(0, 0)`` when ``total`` is zero.
    """
    if total == 0:
        return 0, 0
        
    p_hat = successes / total
    z = 1.96  # 95% confidence level
    se = np.sqrt(p_hat * (1 - p_hat) / total)
    ci_lower = max(0, p_hat - z * se)
    ci_upper = min(1, p_hat + z * se)
    
    return ci_lower, ci_upper

def compute_continuous_ci(mean, std, n):
    """Normal-approximation 95% CI for a mean.

    Args:
        mean: Sample mean.
        std: Sample standard deviation.
        n: Sample size.

    Returns:
        ``(ci_lower, ci_upper)``; ``(0, 0)`` when ``n`` is zero.
    """
    if n == 0:
        return 0, 0
        
    z = 1.96  # 95% confidence level
    se = std / np.sqrt(n)  # standard error = std / sqrt(n)
    ci_lower = mean - z * se
    ci_upper = mean + z * se
    
    return ci_lower, ci_upper

def compute_average_diversity(df):
    """Mean 1 - cosine-similarity of scenario descriptions per value pair.

    Embeds descriptions with the ``all-MiniLM-L6-v2`` sentence transformer,
    groups scenarios by their (order-insensitive) value pair, and averages
    one minus the mean pairwise similarity over groups with at least two
    scenarios. Adds a ``sorted_values`` column to ``df`` as a side effect.

    Args:
        df: Scenario DataFrame with ``value1``, ``value2``, ``description``.

    Returns:
        Dict with ``mean``, ``ci_lower``, ``ci_upper`` (all 0 when no
        group has two or more descriptions).
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['sorted_values'] = df.apply(
        lambda row: tuple(sorted([row['value1'], row['value2']])), 
        axis=1)
    diversity_scores = []
    for sorted_values, group in df.groupby('sorted_values'):
        descriptions = group['description'].tolist()
        if len(descriptions) > 1:  # Need at least 2 descriptions to compute similarity
            embeddings = model.encode(descriptions)
            similarities = cosine_similarity(embeddings)
            n = similarities.shape[0]
            mask = ~np.eye(n, dtype=bool)
            avg_similarity = similarities[mask].mean()
            diversity_scores.append(1 - avg_similarity)
            
    if not diversity_scores:
        return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0}
        
    mean_diversity = np.mean(diversity_scores)
    std_diversity = np.std(diversity_scores)
    n = len(diversity_scores)
    ci_lower, ci_upper = compute_continuous_ci(mean_diversity, std_diversity, n)
    
    return {
        'mean': mean_diversity,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def compute_binary_ci(successes, total):
    """Normal-approximation 95% CI (duplicate of :func:`compute_binomial_ci`)."""
    if total == 0:
        return 0, 0
        
    p_hat = successes / total
    z = 1.96  # 95% confidence level
    se = np.sqrt(p_hat * (1 - p_hat) / total)
    ci_lower = max(0, p_hat - z * se)
    ci_upper = min(1, p_hat + z * se)
    
    return ci_lower, ci_upper

def compute_filter_pass_rates(df):
    """Per-filter and overall pass rates from scenario ``check_results``.

    JSON-decodes the ``check_results`` column if needed (in place), tallies
    each named filter's pass rate across scenarios, and adds an
    ``'overall'`` entry for the ``keep_scenario`` rate.

    Args:
        df: Scenario DataFrame (from
            :mod:`conflictxgame.solo.filter_scenarios` output).

    Returns:
        Mapping of filter name (plus ``'overall'``) to a dict with
        ``rate``, ``ci_lower``, ``ci_upper``; empty if there is no
        ``check_results`` column.
    """
    if 'check_results' not in df.columns:
        return {}
    
    if df['check_results'].dtype == 'object':
        df['check_results'] = df['check_results'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    
    filter_rates = {}
    total_scenarios = len(df)
    
    for row in df.iterrows():
        check_results = row[1]['check_results']
        if isinstance(check_results, dict):
            for filter_name, passed in check_results.items():
                if filter_name not in filter_rates:
                    filter_rates[filter_name] = {'successes': 0, 'total': 0}
                filter_rates[filter_name]['successes'] += int(passed)
                filter_rates[filter_name]['total'] += 1
    
    result = {}
    for filter_name, counts in filter_rates.items():
        rate = counts['successes'] / counts['total']
        ci_lower, ci_upper = compute_binomial_ci(
            counts['successes'], counts['total']
        )
        result[filter_name] = {
            'rate': rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    overall_successes = len(df[df['keep_scenario'] == True])
    overall_rate = overall_successes / total_scenarios
    overall_ci_lower, overall_ci_upper = compute_binomial_ci(
        overall_successes, total_scenarios
    )
    result['overall'] = {
        'rate': overall_rate,
        'ci_lower': overall_ci_lower,
        'ci_upper': overall_ci_upper
    }
    
    return result

def compute_observed_agreement(data):
    """Mean pairwise agreement between models on the binarized choice.

    Binarizes ``choice`` as 0 for ``'A'`` and 1 otherwise (so invalid
    values count with ``'B'``), pivots to a scenario x model matrix, and
    drops scenarios not answered by every model. Adds a ``binary_choice``
    column to ``data`` as a side effect.

    Args:
        data: Evaluation rows spanning multiple evaluated models.

    Returns:
        The fraction of agreeing (scenario, model-pair) comparisons, or -1
        when no scenario is shared by all models.
    """
    data['binary_choice'] = (data['choice'] != 'A').astype(int)
    
    pivot = data.pivot_table(
        index='scenario_id',
        columns='model',
        values='binary_choice',
        aggfunc='first'
    )
    
    pivot = pivot.dropna()
    
    if len(pivot) == 0:
        return -1
    
    response_matrix = pivot.values
    n_scenarios, n_models = response_matrix.shape
    
    total_agreements = 0
    total_comparisons = n_scenarios * (n_models * (n_models - 1)) // 2
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            agreements = np.sum(response_matrix[:, i] == response_matrix[:, j])
            total_agreements += agreements
    
    return total_agreements / total_comparisons if total_comparisons > 0 else -1

def compute_iaa(data):
    """Observed inter-annotator agreement per generating model, with CIs.

    Runs :func:`compute_observed_agreement` on each generating model's rows
    and attaches binomial confidence intervals.

    Args:
        data: Combined evaluation rows across evaluated models.

    Returns:
        Dict with keys ``observed_agreement``, ``ci_lower``, ``ci_upper``,
        each mapping generating model to a value (None when agreement
        could not be computed).
    """
    grouped = data.groupby(['generating_model', 'scenario_id'])
    
    iaa_results = {
        'observed_agreement': {},
        'ci_lower': {},
        'ci_upper': {}
    }
    
    for gen_model, group in data.groupby('generating_model'):
        try:
            agreement = compute_observed_agreement(group)
            
            if agreement >= 0:
                n_models = len(group['model'].unique())
                n_scenarios = len(group['scenario_id'].unique())
                total_comparisons = n_scenarios * (n_models * (n_models - 1)) // 2
                
                total_agreements = agreement * total_comparisons
                
                ci_lower, ci_upper = compute_binomial_ci(
                    total_agreements, total_comparisons
                )
                
                iaa_results['observed_agreement'][gen_model] = agreement
                iaa_results['ci_lower'][gen_model] = ci_lower
                iaa_results['ci_upper'][gen_model] = ci_upper
            else:
                iaa_results['observed_agreement'][gen_model] = None
                iaa_results['ci_lower'][gen_model] = None
                iaa_results['ci_upper'][gen_model] = None
                
        except Exception as e:
            print(f"Error computing observed agreement for {gen_model}: {e}")
            iaa_results['observed_agreement'][gen_model] = None
            iaa_results['ci_lower'][gen_model] = None
            iaa_results['ci_upper'][gen_model] = None
    
    return iaa_results

def compute_preference_rates(data):
    """Preference rate for each (model, value pair), order-normalized.

    Each row's value pair is canonicalized alphabetically, flipping the
    A/B choice when ``value1`` was not first, so the same conflict is
    aggregated regardless of presentation order. The rate is the share of
    choices favoring the alphabetically-first value, with a 95% Wilson
    interval from ``statsmodels``. Rows whose choice is neither ``'A'``
    nor ``'B'`` (e.g. ``INVALID``) are left out of the rate and counted
    in ``n_invalid``.

    Args:
        data: Evaluation rows with ``model``, ``value1``, ``value2``,
            ``choice``.

    Returns:
        DataFrame with columns ``model``, ``value1``, ``value2``,
        ``preference_rate``, ``ci_lower``, ``ci_upper``, ``total``,
        ``count_value1``, ``count_value2``, ``n_invalid``.
    """
    normalized_data = data.copy()
    
    def sort_value_pair(row):
        values = sorted([row['value1'], row['value2']])
        flip_choice = (row['value1'] != values[0]) and (row['choice'] in ['A', 'B'])
        return values[0], values[1], flip_choice
    
    normalized_data[['normalized_value1', 'normalized_value2', 'flip_choice']] = normalized_data.apply(
        sort_value_pair, axis=1, result_type='expand'
    )
    
    normalized_data['normalized_choice'] = normalized_data.apply(
        lambda row: ('B' if row['choice'] == 'A' else 'A') if row['flip_choice'] else row['choice'],
        axis=1
    )
    
    grouped = normalized_data.groupby(['model', 'normalized_value1', 'normalized_value2'])
    
    results = []
    for (model, value1, value2), group in grouped:
        count_a = (group['normalized_choice'] == 'A').sum()
        count_b = (group['normalized_choice'] == 'B').sum()
        n_invalid = len(group) - count_a - count_b
        total = count_a + count_b

        preference_rate = count_a / total if total > 0 else 0
        
        if total > 0:
            ci = smp.proportion_confint(count_a, total, alpha=0.05, method='wilson')
            ci_lower = ci[0]
            ci_upper = ci[1]
        else:
            ci_lower = 0
            ci_upper = 0
        
        results.append({
            'model': model,
            'value1': value1,
            'value2': value2,
            'preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'total': total,
            'count_value1': count_a,
            'count_value2': count_b,
            'n_invalid': n_invalid
        })
    
    return pd.DataFrame(results)

def compute_likert_metrics(data):
    """Rate at which the two per-action Likert ratings differ.

    Coerces ``likert_a``/``likert_b`` to numeric (in place) and drops rows
    where either is missing before comparing.

    Args:
        data: Evaluation rows from the MCQ mode of
            :mod:`conflictxgame.solo.evaluate_models`.

    Returns:
        Dict with ``rate``, ``ci_lower``, ``ci_upper``, ``count``,
        ``different_count``; None when the columns are absent or no row
        has both ratings.
    """
    if 'likert_a' not in data.columns or 'likert_b' not in data.columns:
        print("No 'likert_a' or 'likert_b' columns found in the data. Skipping Likert scale analysis.")
        return None
    
    data['likert_a'] = pd.to_numeric(data['likert_a'], errors='coerce')
    data['likert_b'] = pd.to_numeric(data['likert_b'], errors='coerce')
    
    data_clean = data.dropna(subset=['likert_a', 'likert_b'])
    
    if len(data_clean) == 0:
        print("No valid Likert scale ratings found. Skipping Likert scale analysis.")
        return None
    
    different_count = (data_clean['likert_a'] != data_clean['likert_b']).sum()
    total_count = len(data_clean)
    likert_diff_rate = different_count / total_count if total_count > 0 else 0
    ci_lower, ci_upper = compute_binomial_ci(different_count, total_count)
    
    likert_diff_result = {
        'rate': likert_diff_rate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'count': total_count,
        'different_count': different_count
    }
    
    return likert_diff_result

def log_metrics(model_name, value_set, metric_dict, variant, wandb_project, job_name):
    """Log one metrics dict to Weights & Biases as its own (finished) run.

    Args:
        model_name: Generating model the metrics describe.
        value_set: Value-set name (``--value-set``).
        metric_dict: Metrics to log (already namespaced, e.g.
            ``metrics/...``, ``filtering/...``).
        variant: Data variant (``'filtered'`` or ``'unfiltered'``).
        wandb_project: Target W&B project.
        job_name: Prefix for the run name.

    Raises:
        ImportError: If ``wandb`` is not installed. It is an optional
            dependency imported here rather than at module scope, so that
            analysis runs (``--wandb`` defaults to off) work without it.
    """
    import wandb

    run = wandb.init(reinit=True, project=wandb_project, name=f"{job_name}_{model_name}_{value_set}_{variant}")

    metrics = {
        'args/model': model_name,
        'args/value_set': value_set,
        'args/data_variant': variant,
        **metric_dict
    }
    wandb.log(metrics)
    run.finish()

def fit_bradley_terry_model(data):
    """Fit Bradley-Terry ability scores over values from pairwise choices.

    Each row is one comparison: choice ``'A'`` counts as ``value1``
    beating ``value2`` and ``'B'`` as the reverse; rows with any other
    choice (e.g. ``INVALID``) are dropped rather than scored. Abilities
    are fit with ``choix.ilsr_pairwise`` (regularization ``alpha=0.01``);
    95% CIs come from 1000 bootstrap resamples of the rows.

    Args:
        data: Evaluation rows for a single evaluated model.

    Returns:
        ``(params, rankings_df)``: the raw ability array (indexed by
        sorted value name) and a DataFrame with columns ``value``,
        ``ability``, ``ci_lower``, ``ci_upper``, sorted by descending
        ability.
    """
    data = data[data['choice'].isin(['A', 'B'])]
    unique_values = sorted(set(data['value1'].unique()) | set(data['value2'].unique()))
    n_values = len(unique_values)
    value_to_idx = {val: i for i, val in enumerate(unique_values)}
    
    comparisons = []
    for _, row in data.iterrows():
        if row['choice'] == 'A':
            comparisons.append((value_to_idx[row['value1']], value_to_idx[row['value2']]))
        else:
            comparisons.append((value_to_idx[row['value2']], value_to_idx[row['value1']]))
    
    params = choix.ilsr_pairwise(n_values, comparisons, alpha=0.01)
    
    n_bootstrap = 1000
    bootstrap_estimates = []
    
    for _ in range(n_bootstrap):
        bootstrap_data = data.sample(n=len(data), replace=True)
        
        bootstrap_comparisons = []
        for _, row in bootstrap_data.iterrows():
            if row['choice'] == 'A':
                bootstrap_comparisons.append((value_to_idx[row['value1']], value_to_idx[row['value2']]))
            else:
                bootstrap_comparisons.append((value_to_idx[row['value2']], value_to_idx[row['value1']]))
        
        bootstrap_params = choix.ilsr_pairwise(n_values, bootstrap_comparisons, alpha=0.01)
        bootstrap_estimates.append(bootstrap_params)
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    ci_lower = np.percentile(bootstrap_estimates, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_estimates, 97.5, axis=0)
    rankings_df = pd.DataFrame({
        'value': unique_values,
        'ability': params,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
    rankings_df = rankings_df.sort_values('ability', ascending=False)
    
    return params, rankings_df

def savefig_long_path(path: str, **kwargs) -> str:
    """Save the current figure, tolerating Windows' 260-character path limit.

    These stems carry the model, the arm and both judge names, so the absolute
    path lands right at the boundary: the ``stakes-alignment`` figure saved and
    ``stakes-usefulness`` -- two characters longer -- raised FileNotFoundError
    from deep inside PIL, where the cause is not obvious.

    The extended-length prefix lifts the limit without renaming anything, so
    the figures downstream globs look for keep the names they already have.
    Returns the path actually written, for callers that log it.
    """
    full = os.path.abspath(path)
    prefix = '\\\\?\\'
    if os.name == 'nt' and len(full) > 259 and not full.startswith(prefix):
        full = prefix + full
    plt.savefig(full, **kwargs)
    return full


def plot_bradley_terry_rankings(rankings_df, model_name, output_dir):
    """Save a horizontal error-bar plot of Bradley-Terry abilities as PNG.

    Args:
        rankings_df: Output of :func:`fit_bradley_terry_model`.
        model_name: Evaluated model (used in the title and file name,
            ``bradley_terry_rankings_{short name}.png``).
        output_dir: Directory to write the plot into.
    """
    plt.figure(figsize=(12, 8))
    
    rankings_df = rankings_df.sort_values('ability')
    
    y_pos = np.arange(len(rankings_df))
    plt.errorbar(rankings_df['ability'], y_pos, 
                xerr=[rankings_df['ability'] - rankings_df['ci_lower'], 
                      rankings_df['ci_upper'] - rankings_df['ability']],
                fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8)
    
    plt.yticks(y_pos, [clean_value_name(v) for v in rankings_df['value']])
    
    plt.xlabel('Bradley-Terry Ability Score')
    plt.title(f'Value Rankings for {model_name}')
    
    plt.grid(True, axis='x')
    
    plt.tight_layout()
    # Use the canonical helper: the previous local re-derivation only
    # stripped a provider prefix when the id contained '/', so a
    # 'provider:model' id kept its colon -- illegal in an NTFS filename.
    stem = short_model_name(model_name).replace(' ', '_').lower()
    filename = f"bradley_terry_rankings_{stem}.png"
    savefig_long_path(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def parse_args(argv=None):
    """Parse the analysis CLI arguments, YAML/JSON config defaults first.

    Args:
        argv: Argument list excluding the program name (defaults to the
            process arguments).

    Returns:
        The parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing model evaluation CSV files')
    parser.add_argument('--scenario-dir', type=str, required=True,
                        help='Directory containing scenario CSV files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save output files')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='conflictscope',
                        help='W&B project name')
    parser.add_argument('--job-name', type=str, default=None)
    parser.add_argument('--value-set', type=str, required=True,
                        help='Name of the value set being evaluated')
    parser.add_argument('--compute-bradley-terry', action='store_true',
                        help='Compute Bradley-Terry model rankings')

    args, _config_defaults, _explicit_cli = parse_with_config(parser, argv)
    return args


def main():
    """Entry point: run the full dyadic analysis over both data variants.

    Merges scenario-level ``keep_scenario`` flags onto the evaluation rows,
    computes per-generating-model metrics (optionally logging each set to
    W&B), then writes overall preference rates — and, with
    ``--compute-bradley-terry``, per-model ranking plots and a combined
    rankings CSV — for each variant.
    """
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_data = load_data_files(args.model_dir)
    scenario_data = load_data_files(args.scenario_dir)

    combined_scenarios = combine_data(scenario_data)
    if 'keep_scenario' not in combined_scenarios.columns:
        combined_scenarios['keep_scenario'] = True
        scenario_filters = combined_scenarios[['scenario_id', 'keep_scenario']].drop_duplicates()
    else:
        scenario_filters = combined_scenarios[['scenario_id', 'keep_scenario', 'check_results']].drop_duplicates()
    combined_data = combine_data(model_data)
    
    combined_data = combined_data.merge(scenario_filters, on='scenario_id', how='left')

    data_variants = {
        'unfiltered': combined_data,
        'filtered': combined_data[combined_data['keep_scenario'] == True]
    }
    if len(data_variants['unfiltered']) == len(data_variants['filtered']):
        print("Filtered and unfiltered datasets are identical. Running only filtered analysis.")
        data_variants = {'filtered': data_variants['filtered']}
    
    for variant_name, data in data_variants.items():
        generating_models = data['generating_model'].unique()
        
        for gen_model in tqdm(generating_models, desc="Processing generation models"):
            gen_model_scenarios = combined_scenarios[combined_scenarios['generating_model'] == gen_model]
            gen_model_data = data[data['generating_model'] == gen_model]

            model_metrics = {}
            filter_rates = compute_filter_pass_rates(gen_model_scenarios)
            for filter_name, rates in filter_rates.items():
                model_metrics[f'filtering/{filter_name}'] = rates['rate']
                error = max(rates['ci_upper'] - rates['rate'], rates['rate'] - rates['ci_lower'])
                model_metrics[f'filtering/{filter_name}_error'] = error
                
            diversity = compute_average_diversity(gen_model_scenarios)
            model_metrics['metrics/diversity'] = diversity['mean']
            error = max(diversity['ci_upper'] - diversity['mean'], diversity['mean'] - diversity['ci_lower'])
            model_metrics['metrics/diversity_error'] = error
            
            iaa_results = compute_iaa(gen_model_data)
            if gen_model in iaa_results['observed_agreement']:
                model_metrics['metrics/observed_agreement'] = iaa_results['observed_agreement'][gen_model]
                error = max(iaa_results['ci_upper'][gen_model] - iaa_results['observed_agreement'][gen_model],
                          iaa_results['observed_agreement'][gen_model] - iaa_results['ci_lower'][gen_model])
                model_metrics['metrics/observed_agreement_error'] = error

            preference_rates = compute_preference_rates(gen_model_data)
            likert_diff_result = compute_likert_metrics(gen_model_data)
            if likert_diff_result is not None:
                model_metrics['metrics/likert_diff'] = likert_diff_result['rate']
                error = max(likert_diff_result['ci_upper'] - likert_diff_result['rate'], 
                          likert_diff_result['rate'] - likert_diff_result['ci_lower'])
                model_metrics['metrics/likert_diff_error'] = error

            if args.wandb:
                if args.job_name is None:
                    args.job_name = args.output_dir.split('/')[-2]
                log_metrics(gen_model, args.value_set, model_metrics, variant_name, args.wandb_project, args.job_name)
            
        iaa_results = compute_iaa(data)
        preference_rates = compute_preference_rates(data)
        likert_diff_result = compute_likert_metrics(data)
        if args.compute_bradley_terry:
            all_rankings = []
            for eval_model, eval_data in tqdm(data.groupby('model'), desc="Computing Bradley-Terry model"):
                params, rankings_df = fit_bradley_terry_model(eval_data)
                bradley_terry_output_dir = os.path.join(args.output_dir, 'bradley_terry_rankings')
                os.makedirs(bradley_terry_output_dir, exist_ok=True)
                plot_bradley_terry_rankings(rankings_df, eval_model, bradley_terry_output_dir)
                rankings_df['model'] = eval_model
                all_rankings.append(rankings_df)
            if all_rankings:
                combined_rankings = pd.concat(all_rankings, ignore_index=True)
                combined_rankings.to_csv(os.path.join(args.output_dir, f"bradley_terry_rankings_{variant_name}.csv"), index=False)
        preference_rates.to_csv(os.path.join(args.output_dir, f"preference_rates_{variant_name}.csv"), index=False)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
