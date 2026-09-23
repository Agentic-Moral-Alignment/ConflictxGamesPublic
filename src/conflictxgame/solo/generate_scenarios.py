"""Two-stage dyadic scenario generation for one (value1, value2) pair.

Original ConflictScope pipeline: given two values from a value set, an LLM
generates scenarios in which a text-based AI assistant must choose between
one action supporting ``value1`` and one supporting ``value2``. Generation
runs in two stages:

* **Stage one** produces scenario *skeletons*. Each requested batch is split
  evenly across the causal-graph prompt variants in
  :data:`conflictxgame.solo.prompts.CAUSAL_GRAPH_PROMPTS` (remainder assigned
  to random variants); one LLM call per variant returns a JSON dict of
  skeletons, each annotated with the two value definitions and its
  ``prompt_type``.
* **Stage two** instantiates every skeleton into a concrete scenario via a
  batched LLM call, adding ``description``, ``user_prompt``,
  ``action1``/``action2`` and ``consequence1``/``consequence2`` fields;
  skeletons whose instantiation is malformed are removed.

Between stage-one batches, near-duplicate skeletons can optionally be
dropped using sentence-embedding cosine similarity (``all-MiniLM-L6-v2``).

Output: one JSON file per (model, value pair) —
``{output_dir}/{model}_{value1}_{value2}.json`` — mapping scenario ids of the
form ``{v1}-{v2}-{prompt_type}-{index}`` to scenario dicts. These files are
consumed by :mod:`conflictxgame.solo.filter_scenarios`.

Prompt templates and :func:`conflictxgame.solo.prompts.load_value_dict` live
in :mod:`conflictxgame.solo.prompts`; shared helpers (``parse_json``,
``SYSTEM_PROMPT``) come from :mod:`conflictxgame.common.utils`.

Usage::

    python -m conflictxgame.solo.generate_scenarios -v1 honesty \\
        -v2 helpfulness -v 0217 -o data/scenarios

Provenance: ConflictScope's ``src/generate_scenarios.py`` (MIT, (c) 2025 Andy
Liu), carried over unchanged apart from repathed imports and docstrings:
https://github.com/andyjliu/conflictscope/blob/main/src/generate_scenarios.py
See ``ATTRIBUTION.md``.
"""

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.common.generation_quality import checklist_block
from conflictxgame.common.model_wrappers import (
    ModelWrapper,
    short_model_name,
    split_provider_prefix,
)
from conflictxgame.common.utils import SYSTEM_PROMPT, parse_json
from conflictxgame.solo.prompts import (
    CAUSAL_GRAPH_PROMPTS,
    STAGE_ONE_BASE_PROMPT,
    STAGE_ONE_OUTPUT_FORMAT,
    STAGE_TWO_BASE_PROMPT,
    STAGE_TWO_PROMPT,
    load_value_dict,
)
from argparse import ArgumentParser
from tqdm import tqdm
import hashlib
import os
import json
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#: The paper's HHH triple. This Message Batch arm generates only these three
#: unordered pairs, even if a larger value set is loaded by mistake.
HHH_VALUE_NAMES = ("harmlessness", "helpfulness", "honesty")
HHH_PAIRS = (
    ("harmlessness", "helpfulness"),
    ("harmlessness", "honesty"),
    ("helpfulness", "honesty"),
)

def parse_args(argv=None):
    """Parse generation CLI arguments, YAML/JSON config defaults first.

    Args:
        argv: Argument list excluding the program name (defaults to the
            process arguments).

    Returns:
        The parsed :class:`argparse.Namespace`.
    """
    parser = ArgumentParser(description=__doc__)
    add_config_argument(parser)

    parser.add_argument('--value1', '-v1', type=str, default=None)
    parser.add_argument('--value2', '-v2', type=str, default=None)
    parser.add_argument('--value-set', '-v', type=str, required=True)
    parser.add_argument(
        '--all-hhh-pairs',
        action='store_true',
        help='Generate the three HHH pairs (harmlessness/helpfulness/honesty) '
             'in one process. Mutually exclusive with -v1/-v2. The loaded '
             'value set must be exactly that triple (HHH or HHH_new).',
    )
    parser.add_argument('--num-scenarios', '-n', type=int, default=40)
    parser.add_argument('--batch-size', '-b', type=int, default=40)
    parser.add_argument('--max-extra-tries', '-e', type=int, default=5)
    parser.add_argument('--deduplicate', '-d', action='store_true')
    parser.add_argument('--deduplication_threshold', '-dt', type=float, default=0.9)
    parser.add_argument('--model', '-m', type=str, default='gpt-4o-mini')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--add-to-existing', '-a', action='store_true')
    parser.add_argument('--criteria-in-context', action='store_true',
                        help='Append the distilled filter checklist to both generation '
                             'stages, raising the share of scenarios that survive '
                             'filtering. The judge is unchanged either way. Off by '
                             'default, preserving the original prompts.')
    parser.add_argument(
        '--generation-guidance',
        type=str,
        default=None,
        help='Comma-separated generation guidance parts (recipes, nearmiss, '
             'checklist, stage2). This extends --criteria-in-context while '
             'leaving the independent filter judge unchanged.',
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Seed for splitting a stage-one batch across causal-graph prompt '
             'variants. Required for resumable Anthropic Message Batches.',
    )
    parser.add_argument(
        '--anthropic-message-batches', action='store_true',
        help='Submit stage-one prompt-type calls and stage-two instantiations '
             'through the Anthropic Message Batches API. Opt-in; requires '
             '--message-batch-state-dir and an Anthropic-routed generator.',
    )
    parser.add_argument(
        '--message-batch-state-dir', type=str, default=None,
        help='Durable provider manifests for --anthropic-message-batches. '
             'Must be private to this solo run, not shared with a games arm.',
    )
    parser.add_argument(
        '--message-batch-poll-seconds', type=float, default=60.0,
        help='Seconds between Anthropic Message Batch status polls (default 60).',
    )
    parser.add_argument(
        '--recover-stage-one-state-dir', type=str, default=None,
        help='Import structurally valid stage-one skeletons from completed '
             'Message Batch manifests before generation. Existing skeleton '
             'content is skipped, making recovery safe to resume.',
    )

    args, _config_defaults, _explicit_cli = parse_with_config(parser, argv)
    return args


def _hhh_pairs_or_exit(value_dict: dict) -> tuple:
    """Return the three HHH pairs, or exit if this is not an HHH value set."""
    if set(value_dict) != set(HHH_VALUE_NAMES):
        raise SystemExit(
            "--all-hhh-pairs only generates the HHH triple "
            f"{list(HHH_VALUE_NAMES)}; this value set has {sorted(value_dict)}."
        )
    return HHH_PAIRS


def _solo_batch_key(state_dir: str, model: str, value_set: str, *parts: object) -> str:
    """Stable Message Batch key scoped to this solo run and logical request."""
    digest = hashlib.sha256(
        json.dumps(
            {
                "state_dir": os.path.abspath(state_dir),
                "model": model,
                "value_set": value_set,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    label = "-".join(str(part) for part in parts)
    return f"solo-{digest}-{label}"


def _generation_pairs(args, value_dict: dict):
    """Resolve one CLI pair or the three HHH pairs."""
    if args.all_hhh_pairs:
        if args.value1 or args.value2:
            raise SystemExit("--all-hhh-pairs cannot be combined with -v1/-v2.")
        return _hhh_pairs_or_exit(value_dict)
    if not args.value1 or not args.value2:
        raise SystemExit("Pass -v1 and -v2, or --all-hhh-pairs for the HHH triple.")
    return ((args.value1, args.value2),)

def standardize_dict_keys(d: dict) -> dict:
    """Normalize dict keys: lower-case them and replace spaces with underscores.

    Args:
        d: Dictionary with arbitrary string keys (LLM-produced field names).

    Returns:
        A new dict with normalized keys and unchanged values.
    """
    return {k.lower().replace(' ', '_') : v for k, v in d.items()}


def _next_scenario_index(scenarios: dict) -> int:
    """Return an unused numeric suffix for a resumed scenario corpus.

    Scenario files can contain gaps when stage-two validation removes an
    entry.  Using ``len(scenarios)`` after such a gap can overwrite an existing
    id instead of extending the corpus, so resume after the largest suffix.
    """
    suffixes = []
    for scenario_id in scenarios:
        try:
            suffixes.append(int(scenario_id.rsplit('-', 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(suffixes, default=-1) + 1


def _skeleton_fingerprint(scenario: dict) -> str | None:
    """Fingerprint the stage-one portion retained in a full scenario."""
    fields = (
        'context', 'action_opportunity',
        'necessary_strong_harm', 'necessary_strong_good',
        'strong_harm', 'strong_good', 'mild_harm', 'mild_good',
    )
    payload = {key: scenario[key] for key in fields if key in scenario}
    if not {'context', 'action_opportunity'} <= payload.keys():
        return None
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()


def _recover_stage_one_scenarios(
    state_dir: str, v1: str, v2: str, start_idx: int,
    value_dict: dict, existing_scenarios: dict,
) -> dict:
    """Recover valid, non-duplicate skeletons from completed batch manifests."""
    recovered = {}
    seen = {
        fingerprint
        for fingerprint in map(_skeleton_fingerprint, existing_scenarios.values())
        if fingerprint is not None
    }
    next_idx = start_idx
    pair_marker = f'-{v1}-{v2}-stage1-'

    for name in sorted(os.listdir(state_dir)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(state_dir, name)
        with open(path, 'r') as handle:
            manifest = json.load(handle)
        if pair_marker not in manifest.get('batch_key', ''):
            continue
        if manifest.get('status') != 'ended':
            continue
        for result in manifest.get('results', []):
            if result.get('type') != 'succeeded':
                continue
            outputs = parse_json(result.get('text', ''))
            if not isinstance(outputs, dict):
                continue
            for output_id, output in outputs.items():
                if not isinstance(output, dict):
                    continue
                normalized = standardize_dict_keys(output)
                fingerprint = _skeleton_fingerprint(normalized)
                if fingerprint is None or fingerprint in seen:
                    continue
                prefix = f'{v1}-{v2}-'
                prompt_type = 'recovered'
                if output_id.startswith(prefix):
                    prompt_type = output_id[len(prefix):].rsplit('-', 1)[0]
                normalized['value1'] = value_dict[v1]
                normalized['value2'] = value_dict[v2]
                normalized['prompt_type'] = prompt_type
                scenario_id = f'{v1}-{v2}-{prompt_type}-{next_idx}'
                recovered[scenario_id] = normalized
                seen.add(fingerprint)
                next_idx += 1

    return recovered

def generate_stage_one(v1, v2, client, num_scenarios, start_idx, value_dict,
                       criteria_in_context=False, batch_key=None, seed=None,
                       guidance_parts=None):
    """Generate a batch of scenario skeletons for one value pair.

    The batch is split evenly across the causal-graph prompt variants in
    :data:`conflictxgame.solo.prompts.CAUSAL_GRAPH_PROMPTS` (any remainder is
    assigned to randomly chosen variants). For each variant with a nonzero
    quota, one LLM call is made with ``STAGE_ONE_BASE_PROMPT`` + the variant
    prompt + ``STAGE_ONE_OUTPUT_FORMAT``, and the parsed JSON skeletons are
    collected up to that variant's quota. Each kept skeleton gets its keys
    normalized via :func:`standardize_dict_keys` and is annotated with the
    two value *definitions* (``value1``/``value2``) and its ``prompt_type``;
    outputs that are not dicts are silently skipped.

    Args:
        v1: Short name of the first value (key into ``value_dict``).
        v2: Short name of the second value.
        client: :class:`conflictxgame.common.model_wrappers.ModelWrapper`
            used for generation.
        num_scenarios: Number of skeletons requested in this batch.
        start_idx: Offset used when numbering scenario ids, so that ids stay
            unique across successive batches.
        value_dict: Mapping of short value names to full definitions, from
            :func:`conflictxgame.solo.prompts.load_value_dict`.
        criteria_in_context: Append the distilled filter checklist
            (:data:`conflictxgame.common.generation_quality.FILTER_CHECKLISTS`)
            to the prompt, raising the share of scenarios that survive
            filtering. Off by default, preserving the original prompt; the
            filter judge is unchanged either way.
        batch_key: When set, send every prompt-type call through
            ``client.batch_generate`` under this stable Message Batch key.
        seed: When set, remainder prompt-type assignment is deterministic,
            which Message Batch resume requires.

    Returns:
        Dict mapping ids of the form ``{v1}-{v2}-{prompt_type}-{index}`` to
        skeleton dicts.
    """
    scenarios = {}
    prompt_types = list(CAUSAL_GRAPH_PROMPTS.keys())
    prompt_types_count = {prompt_type: int(num_scenarios / len(prompt_types)) for prompt_type in prompt_types}
    assign = random.Random(seed).choice if seed is not None else random.choice
    for i in range(num_scenarios - sum(prompt_types_count.values())):
        prompt_types_count[assign(prompt_types)] += 1

    requests = []
    for prompt_type in prompt_types:
        if prompt_types_count[prompt_type] == 0:
            continue
        base_prompt = STAGE_ONE_BASE_PROMPT.format(
            v1=v1,
            v2=v2,
            value1_definition=value_dict[v1],
            value2_definition=value_dict[v2]
        )
        base_prompt += "\n" + CAUSAL_GRAPH_PROMPTS[prompt_type].format(v1, v2)
        base_prompt += "\n" + STAGE_ONE_OUTPUT_FORMAT.format(
            num_scenarios=prompt_types_count[prompt_type],
            v1=v1,
            v2=v2,
            prompt_type=prompt_type
        )
        if guidance_parts is None:
            base_prompt += checklist_block('solo', criteria_in_context)
        else:
            from conflictxgame.common.generation_quality import guidance_block
            base_prompt += guidance_block('solo', 'stage_one', guidance_parts)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": base_prompt}]
        requests.append((prompt_type, prompt_types_count[prompt_type], messages))

    if not requests:
        return scenarios
    if batch_key is not None:
        texts = client.batch_generate(
            [messages for _, _, messages in requests],
            verbose=True,
            batch_key=batch_key,
        )
    else:
        texts = [client.generate(messages) for _, _, messages in requests]

    for (prompt_type, quota, _), text in zip(requests, texts):
        scenario_length = len(scenarios)
        outputs = parse_json(text)
        for idx, output in enumerate(outputs):
            if idx >= quota:
                break
            try:
                outputs[output]['value1'] = value_dict[v1]
                outputs[output]['value2'] = value_dict[v2]
                outputs[output]['prompt_type'] = prompt_type
                scenarios[f'{v1}-{v2}-{prompt_type}-{start_idx + scenario_length + idx}'] = standardize_dict_keys(outputs[output])
            except TypeError:
                pass

    return scenarios

def generate_stage_two(scenarios, client, value_dict,
                       criteria_in_context=False, batch_key=None,
                       guidance_parts=None):
    """Instantiate stage-one skeletons into full scenarios via batched LLM calls.

    Skeletons that already have a ``description`` are left untouched. For
    every other skeleton, its remaining fields are rendered into a template
    and combined with ``STAGE_TWO_BASE_PROMPT``/``STAGE_TWO_PROMPT`` (the
    short value names are recovered by reverse lookup of the stored
    definitions in ``value_dict``); all prompts are sent through
    ``client.batch_generate``. The parsed output's ``description``,
    ``user_prompt``, ``action1``, ``action2``, ``consequence1`` and
    ``consequence2`` fields are copied into the skeleton; scenarios whose
    output is missing any of these keys are deleted.

    Args:
        scenarios: Skeletons from :func:`generate_stage_one` (modified in
            place).
        client: Model wrapper used for generation.
        value_dict: Mapping of short value names to full definitions.
        criteria_in_context: Append the distilled filter checklist to each
            prompt (see :func:`generate_stage_one`).
        batch_key: When set, send the instantiations through
            ``client.batch_generate`` under this stable Message Batch key.

    Returns:
        The same ``scenarios`` dict with instantiated scenarios and failed
        entries removed.
    """
    scenario_list = []
    for scenario_id, scenario in scenarios.items():
        if 'description' in scenario:
            continue
        else:
            keys = [k for k in list(scenario.keys()) if k not in ['value1', 'value2', 'prompt_type']]
            template = '\n'.join(f'{k}: {scenario[k]}' for k in keys)
            reverse_dict = {v: k for k, v in value_dict.items()}

            base_prompt = STAGE_TWO_BASE_PROMPT.format(
                v1=reverse_dict[scenario['value1']],
                v2=reverse_dict[scenario['value2']],
                value1_definition=scenario['value1'],
                value2_definition=scenario['value2']
            )
            base_prompt += "\n" + STAGE_TWO_PROMPT.format(
                template = template,
                value1 = scenario['value1'],
                value2 = scenario['value2']
            )
            if guidance_parts is None:
                base_prompt += checklist_block('solo', criteria_in_context)
            else:
                from conflictxgame.common.generation_quality import guidance_block
                base_prompt += guidance_block('solo', 'stage_two', guidance_parts)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": base_prompt}]
            scenario_list.append((scenario_id, messages))
            
    if not scenario_list:
        return scenarios
    batch_kwargs = {}
    if batch_key is not None:
        batch_kwargs['batch_key'] = batch_key
    outputs = [parse_json(output) for output in client.batch_generate(
        [scenario[1] for scenario in scenario_list], verbose=True, **batch_kwargs
    )]
    for (scenario_id, _), output in zip(scenario_list, outputs):
        try:
            keys = ['description', 'user_prompt', 'action1', 'action2', 'consequence1', 'consequence2']
            for key in keys:
                scenarios[scenario_id][key] = output[key]

        except KeyError:
            print(f'Removing the following scenario due to instantiation error: {scenarios[scenario_id]}')
            del scenarios[scenario_id]

    return scenarios

def deduplicate_scenarios(embeddings, generated_scenarios, embedding_model, threshold):
    """Drop newly generated skeletons that near-duplicate earlier ones.

    Each new skeleton is embedded from its ``context`` plus
    ``action_opportunity`` text. A new skeleton is removed when its cosine
    similarity to *any* earlier row — a previously kept scenario or an
    earlier skeleton in the same batch — exceeds ``threshold``.

    Args:
        embeddings: Matrix of embeddings for all previously kept scenarios
            (may have zero rows).
        generated_scenarios: New skeletons from :func:`generate_stage_one`.
        embedding_model: A ``SentenceTransformer`` used to embed scenarios.
        threshold: Cosine-similarity cutoff above which a scenario counts as
            a duplicate.

    Returns:
        Tuple of (surviving scenarios dict, updated embedding matrix
        containing the old rows plus the surviving new rows).
    """
    new_embeddings = embedding_model.encode([scenario.get('context', '') + ' ' + scenario.get('action_opportunity', '') for scenario in generated_scenarios.values()])
    all_embeddings = np.vstack((embeddings, new_embeddings))
    all_similarities = cosine_similarity(all_embeddings)

    to_remove = set()
    for i in range(len(embeddings), len(all_embeddings)):
        try:
            if max(all_similarities[i][:i]) > threshold:
                to_remove.add(i)
        except ValueError:
            continue

    saved_ids = [k for idx, k in enumerate(generated_scenarios.keys()) if idx + len(embeddings) not in to_remove]
    saved_scenarios = {k:v for k,v in generated_scenarios.items() if k in saved_ids}
    saved_indices = list(range(len(embeddings))) + [i for i in range(len(embeddings), len(all_embeddings)) if i not in to_remove]
    return saved_scenarios, all_embeddings[saved_indices]
    
def main():
    """Entry point: generate scenarios for one value pair or the HHH triple.

    Runs stage one in batches until ``--num-scenarios`` new scenarios exist,
    optionally deduplicating between batches, stopping early after
    ``--max-extra-tries`` consecutive batches that add nothing. Then runs
    stage two and writes ``{model}_{value1}_{value2}.json`` into
    ``--output-dir``. An existing output file is only extended when
    ``--add-to-existing`` is set; otherwise that pair is skipped.
    """
    args = parse_args()
    if args.anthropic_message_batches:
        if not args.message_batch_state_dir:
            raise SystemExit(
                '--anthropic-message-batches requires --message-batch-state-dir'
            )
        if args.message_batch_poll_seconds <= 0:
            raise SystemExit('--message-batch-poll-seconds must be greater than zero')
        if args.seed is None:
            raise SystemExit(
                '--anthropic-message-batches requires --seed so stage-one '
                'prompt-type splits can be resumed without submitting a second batch'
            )
        provider, _ = split_provider_prefix(args.model)
        if provider not in (None, 'anthropic') or (
            provider is None and 'claude' not in args.model.lower()
        ):
            raise SystemExit(
                '--anthropic-message-batches requires an Anthropic-routed generation '
                'model (for example claude-sonnet-5).'
            )
        args.message_batch_state_dir = os.path.abspath(args.message_batch_state_dir)
        os.makedirs(args.message_batch_state_dir, exist_ok=True)

    client_kwargs = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
    }
    if args.anthropic_message_batches:
        client_kwargs.update(
            use_message_batches=True,
            batch_state_dir=args.message_batch_state_dir,
            batch_poll_interval=args.message_batch_poll_seconds,
        )
    client = ModelWrapper.create(args.model, **client_kwargs)
    value_dict = load_value_dict(args.value_set)
    pairs = _generation_pairs(args, value_dict)

    from conflictxgame.common.generation_quality import GUIDANCE_PARTS, parse_guidance
    guidance_parts = list(parse_guidance(args.generation_guidance or ''))
    if args.criteria_in_context and 'checklist' not in guidance_parts:
        guidance_parts.append('checklist')
    guidance_parts = tuple(part for part in GUIDANCE_PARTS if part in guidance_parts)

    embedding_model = None
    if args.deduplicate:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    os.makedirs(args.output_dir, exist_ok=True)
    model_stem = short_model_name(args.model)

    for v1, v2 in pairs:
        _generate_one_pair(
            args, client, value_dict, embedding_model, model_stem, v1, v2,
            guidance_parts=guidance_parts,
        )


def _generate_one_pair(
    args, client, value_dict, embedding_model, model_stem, v1, v2,
    guidance_parts=None,
):
    """Generate and write one (value1, value2) JSON file."""
    embeddings = np.empty((0, 384))
    output_path = os.path.join(
        args.output_dir, f'{model_stem}_{v1}_{v2}.json'
    )

    if not os.path.exists(output_path):
        scenarios = {}
    else:
        if args.add_to_existing:
            with open(output_path, 'r') as f:
                scenarios = json.load(f)
        else:
            print(f'Output file {output_path} already exists. Exiting this pair.')
            return

    if args.recover_stage_one_state_dir:
        recovered = _recover_stage_one_scenarios(
            args.recover_stage_one_state_dir,
            v1, v2, _next_scenario_index(scenarios), value_dict, scenarios,
        )
        scenarios.update(recovered)
        print(
            f'Recovered {len(recovered)} valid stage-one scenarios for '
            f'{v1}|{v2} from {args.recover_stage_one_state_dir}.'
        )

    if embedding_model is not None and scenarios:
        embeddings = embedding_model.encode([
            scenario.get('context', '') + ' ' + scenario.get('action_opportunity', '')
            for scenario in scenarios.values()
        ])

    target_length = args.num_scenarios + len(scenarios)
    tries = 0
    round_index = 0

    progress_bar = tqdm(
        total=target_length,
        desc=f"Generating {v1}|{v2}: stage one",
    )
    progress_bar.update(len(scenarios))

    while len(scenarios) < target_length:
        initial_count = len(scenarios)
        effective_batch_size = min(args.batch_size, target_length - len(scenarios))
        start_idx = _next_scenario_index(scenarios)
        stage_one_key = None
        if args.anthropic_message_batches:
            round_index += 1
            stage_one_key = _solo_batch_key(
                args.message_batch_state_dir,
                args.model,
                args.value_set,
                v1, v2, 'stage1', round_index, effective_batch_size, len(scenarios),
            )
        generated = generate_stage_one(
            v1, v2, client, effective_batch_size,
            start_idx, value_dict,
            criteria_in_context=args.criteria_in_context,
            batch_key=stage_one_key,
            seed=args.seed,
            guidance_parts=guidance_parts,
        )
        if embedding_model is not None and len(generated) > 0:
            generated, embeddings = deduplicate_scenarios(
                embeddings, generated, embedding_model, args.deduplication_threshold
            )

        scenarios = scenarios | generated
        new_scenarios = len(scenarios) - initial_count
        progress_bar.update(new_scenarios)

        if new_scenarios == 0:
            tries += 1
        if tries >= args.max_extra_tries:
            break

    progress_bar.close()

    stage_two_key = None
    if args.anthropic_message_batches:
        stage_two_key = _solo_batch_key(
            args.message_batch_state_dir,
            args.model,
            args.value_set,
            v1, v2, 'stage2', len(scenarios),
        )
    scenarios = generate_stage_two(
        scenarios, client, value_dict,
        criteria_in_context=args.criteria_in_context,
        batch_key=stage_two_key,
        guidance_parts=guidance_parts,
    )
    with open(output_path, 'w') as f:
        json.dump(scenarios, f, indent=4)

if __name__ == '__main__':
    main()
