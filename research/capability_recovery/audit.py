"""Streaming historical dataset/provenance audit; no private example text in output."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .common import digest, rows, write_json

RUNS = {'v1': 'sft-run-20261013', 'v2': 'sft-v2-run-20261015',
        'v3': 'sft-v3-run-20261017', 'v4': 'sft-v4-run-20261018',
        'v5': 'sft-v5-run-20261018'}


def audit_dataset(path, tokenizer=None):
    counts = Counter()
    sources = Counter()
    source_tokens = Counter()
    targets = Counter()
    weighted = Counter()
    hist = Counter()
    prompts = Counter()
    groups = set()
    language = Counter()
    errors = Counter()
    nrows = ntokens = ntargets = maxlen = 0
    for r in rows(path):
        ids, labels = r['input_ids'], r['labels']
        nrows += 1
        n = sum(v != -100 for v in labels[1:])
        family = r.get('component', r.get('family', 'unclassified'))
        source = r.get('source', 'not_stored')
        counts[family] += 1
        sources[source] += 1
        source_tokens[source] += n
        targets[family] += n
        weighted[r.get('objective_component', family)] += n * r.get('objective_weight', 1.)
        ntokens += len(ids)
        ntargets += n
        maxlen = max(maxlen, len(ids))
        bucket = next((b for b in [128, 256, 512, 1024, 2048, 3072, 4096] if len(ids) <= b), 'over4096')
        hist[str(bucket)] += 1
        if len(ids) != len(labels) or not labels or labels[0] != -100:
            errors['length_or_first_label'] += 1
        if any(y != -100 and x != y for x, y in zip(ids, labels)):
            errors['label_token_mismatch'] += 1
        if n == 0:
            errors['no_shifted_targets'] += 1
        if 519 not in labels:
            errors['missing_assistant_end_token'] += 1
        if r.get('n_target_tokens', n) != n:
            errors['stored_target_count_mismatch'] += 1
        h = r.get('prompt_hash')
        if h:
            prompts[h] += 1
        if r.get('group'):
            groups.add(r['group'])
        # This is declared/inherited language, not a language detector certification.
        lang = r.get('language')
        if not lang:
            lang = 'en_source_inferred' if any(v in source for v in ['ultrachat', 'openai/gsm8k', 'smoltalk']) else 'unknown'
        language[lang] += n
        if tokenizer is not None:
            from sft_v5.prepare import messages, check_labels
            try:
                check_labels(tokenizer, r, messages(tokenizer, r))
            except (AssertionError, ValueError, KeyError):
                errors['independent_decoded_mask_or_eos'] += 1
    return {'file_sha256': digest(path), 'rows': nrows, 'input_tokens': ntokens,
            'assistant_target_tokens': ntargets, 'counts_by_family': dict(counts),
            'target_tokens_by_family': dict(targets),
            'target_shares': {k: v/max(ntargets, 1) for k, v in targets.items()},
            'source_counts': dict(sources), 'source_target_tokens': dict(source_tokens),
            'global_weighted_tokens': dict(weighted),
            'length_histogram_upper_bounds': dict(hist), 'max_length': maxlen,
            'duplicate_stored_prompt_hashes': sum(v-1 for v in prompts.values()),
            'unique_stored_groups': len(groups), 'language_target_tokens': dict(language),
            'errors': dict(errors), 'all_masks_independently_decoded': tokenizer is not None,
            'near_duplicate_semantic_audit_complete': False,
            'source_truth_independently_verified': False}


def exposure(root):
    """Recompute V5 batch-normalized exposure, not the global weighted mass shortcut."""
    import random
    metadata = [{'family': r['family'], 'target_tokens': sum(y != -100 for y in r['labels'][1:]),
                 'input_tokens': len(r['input_ids']),
                 'component': r['objective_component'], 'weight': r['objective_weight']}
                for r in rows(root/'train.jsonl.gz')]
    order = list(range(len(metadata)))
    random.Random(20261018).shuffle(order)
    mass = Counter()
    composition = Counter()
    selected_mass = Counter()
    for step, offset in enumerate(range(0, len(order), 32), 1):
        batch = [metadata[i] for i in order[offset:offset+32]]
        denominator = sum(r['target_tokens']*r['weight'] for r in batch)
        for r in batch:
            value = r['target_tokens']*r['weight']/denominator
            mass[r['component']] += value
            if step <= 1600:
                selected_mass[r['component']] += value
        composition[','.join(sorted({r['component'] for r in batch}))] += 1
    steps = (len(order)+31)//32
    return {'optimizer_steps': steps, 'batch_normalized_objective_shares': {k: v/steps for k, v in mass.items()},
            'selected1600_rows': len(order[:1600*32]),
            'selected1600_input_tokens': sum(metadata[i]['input_tokens'] for i in order[:1600*32]),
            'selected1600_assistant_tokens': sum(metadata[i]['target_tokens'] for i in order[:1600*32]),
            'selected1600_objective_shares': {k: v/1600 for k, v in selected_mass.items()},
            'step_composition_counts': dict(composition), 'actual_gradient_contributions_measured': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--home', type=Path, default=Path('/home/serda'))
    p.add_argument('--output', default='research/capability_recovery/results/data_history.json')
    p.add_argument('--decode-v5', action='store_true')
    args = p.parse_args()
    tok = None
    if args.decode_v5:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.home/RUNS['v5']/'tokenizer', local_files_only=True)
    result = {}
    for name, directory in RUNS.items():
        root = args.home/directory
        result[name] = {'manifest_sha256': digest(root/'manifest.json'),
                        'config': json.loads((root/'config.json').read_text()),
                        'manifest': json.loads((root/'manifest.json').read_text())}
        for split in ['train', 'validation']:
            result[name][split] = audit_dataset(root/(split+'.jsonl.gz'), tok if name == 'v5' else None)
        train_hash = {r['prompt_hash'] for r in rows(root/'train.jsonl.gz') if 'prompt_hash' in r}
        val_hash = {r['prompt_hash'] for r in rows(root/'validation.jsonl.gz') if 'prompt_hash' in r}
        result[name]['split_stored_prompt_hash_overlap'] = len(train_hash & val_hash) if train_hash and val_hash else None
        print(name, result[name]['train']['rows'], result[name]['train']['assistant_target_tokens'], flush=True)
        write_json(args.output, result)
    result['v5']['batch_exposure'] = exposure(args.home/RUNS['v5'])
    write_json(args.output, result)


if __name__ == '__main__':
    main()
