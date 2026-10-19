"""Scoring, paired uncertainty, source-group splitting and sealed-test guard."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from decimal import Decimal, InvalidOperation


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def shingles(text, n=13):
    words = normalize(text).split()
    return {hashlib.sha256(' '.join(words[i:i+n]).encode()).hexdigest()
            for i in range(max(0, len(words)-n+1))}


def group_split(cases, seed=20261019):
    """Join same group, exact prompts and shared 13-word spans before splitting.

    This is conservative and can join stock instructions. It is not a semantic
    translation detector; explicit translation/source group IDs remain required.
    """
    parent = list(range(len(cases)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(j)] = find(i)
    seen = {}
    for i, case in enumerate(cases):
        # Include context separately so paraphrased questions on one document join.
        prompt = '\n'.join(m['content'] for m in case['messages'])
        keys = {'group:' + case['group'], 'exact:' + normalize(prompt)}
        keys.update('span:' + s for s in shingles(prompt))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    clusters = defaultdict(list)
    for i in range(len(cases)):
        clusters[find(i)].append(i)
    assignment = {}
    for indices in clusters.values():
        # Stable cluster identity, independent of incoming row order.
        group = min(cases[i]['group'] for i in indices)
        h = hashlib.sha256(f'{seed}/{group}'.encode()).hexdigest()
        split = 'dev' if int(h[:8], 16) % 2 == 0 else 'final'
        for i in indices:
            assignment[cases[i]['id']] = (split, group)
    return assignment


def wilson(correct, n, z=1.959963984540054):
    if n == 0:
        return None
    p = correct/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [center-radius, center+radius]


def paired_difference(reference, candidate, seed=20261019, repeats=2000):
    a, b = [{r['id']: r for r in data} for data in [reference, candidate]]
    if len(a) != len(reference) or len(b) != len(candidate) or a.keys() != b.keys():
        raise ValueError('Incomplete/duplicate paired cases')
    groups = defaultdict(list)
    for key in sorted(a):
        if a[key]['group'] != b[key]['group']:
            raise ValueError('Mismatched source groups')
        if a[key]['correct'] is None or b[key]['correct'] is None:
            continue
        groups[a[key]['group']].append(int(b[key]['correct'])-int(a[key]['correct']))
    if not groups:
        return {'scored': 0, 'difference': None, 'ci95': None}
    values = list(groups.values())
    rng = random.Random(seed)
    boot = []
    for _ in range(repeats):
        sampled = [values[rng.randrange(len(values))] for _ in values]
        boot.append(sum(map(sum, sampled))/sum(map(len, sampled)))
    boot.sort()
    diffs = [v for group in values for v in group]
    return {'scored': len(diffs), 'groups': len(groups), 'difference': statistics.mean(diffs),
            'gained': diffs.count(1), 'lost': diffs.count(-1),
            'ci95': [boot[int(.025*repeats)], boot[min(repeats-1, int(.975*repeats))]],
            'method': 'paired source-group bootstrap; not independent-item bootstrap'}


def repetition(text):
    words = normalize(text).split()
    grams = [tuple(words[i:i+8]) for i in range(max(0, len(words)-7))]
    return 1-len(set(grams))/len(grams) if grams else 0.


def constraint_score(contracts, response):
    results = []
    for contract in contracts:
        kind, value = contract['kind'], contract['value']
        if kind == 'exact':
            good = response.strip() == value
        elif kind == 'json':
            try:
                def pairs(items):
                    if len({k for k,_ in items}) != len(items):
                        raise ValueError('Duplicate JSON keys')
                    return dict(items)
                def constant(_):
                    raise ValueError('Nonstandard JSON number')
                parsed = json.loads(response, object_pairs_hook=pairs, parse_constant=constant)
                good = json_equal(parsed,value)
            except ValueError:
                good = False
        elif kind == 'endswith':
            good = response.rstrip().endswith(value)
        elif kind == 'contains':
            good = value in response
        elif kind == 'forbidden':
            good = value not in response
        elif kind == 'word_range':
            good = value[0] <= len(response.split()) <= value[1]
        elif kind == 'paragraphs':
            good = len([x for x in response.strip().split(contract.get('separator', '\n\n')) if x.strip()]) == value
        elif kind == 'lines':
            good = len(response.strip().splitlines()) == value
        else:
            raise ValueError('Unsupported constraint: ' + kind)
        results.append(bool(good))
    return all(results) if results else None


def json_equal(a,b):
    if isinstance(a,bool) or isinstance(b,bool):
        return type(a) is type(b) and a==b
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(json_equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(json_equal(x,y) for x,y in zip(a,b))
    return a==b


def score_case(case, response):
    # Never promote format-only correctness into semantic correctness.
    contracts = constraint_score(case.get('contracts', []), response)
    scorer = case['scorer']
    if scorer == 'exact':
        correct = normalize(response) in {normalize(a) for a in case['answers']}
    elif scorer == 'number':
        numbers = re.findall(r'-?\d+(?:[.,]\d+)?', response.rsplit('####', 1)[-1])
        try:
            correct = bool(numbers) and Decimal(numbers[-1].replace(',', '.')) == Decimal(str(case['answer']))
        except InvalidOperation:
            correct = False
    elif scorer == 'boxed_exact':
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', response)
        correct = bool(boxed) and normalize(boxed[-1]) == normalize(case['answer'])
    elif scorer == 'contracts_with_verified_content':
        correct = contracts if case.get('content_verifier') == 'exact_json_or_literal' else None
    elif scorer in ('manual', 'code'):
        correct = None
    else:
        raise ValueError('Unknown scorer')
    errors = []
    if correct is False:
        errors.append('incorrect_under_declared_scorer')
    if correct is None:
        errors.append('semantic_or_execution_review_pending')
    if contracts is False:
        errors.append('format_violation')
    return {'correct': correct, 'format_compliant': contracts, 'error_categories': errors,
            'reasoning_valid': None, 'semantic_scope': case.get('semantic_scope', scorer),
            'repetition_8gram': repetition(response), 'characters': len(response),
            'words': len(response.split())}


def summarize(records):
    output = {}
    for family in sorted({r['family'] for r in records}):
        group = [r for r in records if r['family'] == family]
        scored = [r for r in group if r['correct'] is not None]
        correct = sum(bool(r['correct']) for r in scored)
        lengths = sorted(r['output_tokens'] for r in group)
        output[family] = {'n': len(group), 'scored_n': len(scored), 'correct': correct,
                          'ci95': wilson(correct, len(scored)),
                          'accuracy': correct/len(scored) if scored else None,
                          'source_groups': len({r['group'] for r in group}),
                          'truncated': sum(r['truncated'] for r in group),
                          'eos_terminated': sum(r['eos_terminated'] for r in group),
                          'correct_and_eos_terminated': sum(r['eos_terminated'] and r['correct'] is True for r in group),
                          'repetitive': sum(r['repetition_8gram'] > .2 for r in group),
                          'length_tokens': {'min': min(lengths), 'median': statistics.median(lengths),
                                            'p95': lengths[min(len(lengths)-1, int(.95*len(lengths)))],
                                            'max': max(lengths)},
                          'manual_review_pending': sum(r['correct'] is None for r in group)}
        output[family]['format_scored'] = sum(r.get('format_compliant') is not None for r in group)
        output[family]['format_correct'] = sum(r.get('format_compliant') is True for r in group)
        output[family]['error_categories'] = dict(Counter(e for r in group for e in r.get('error_categories', [])))
        output[family]['reasoning_validity_scored'] = sum(r.get('reasoning_valid') is not None for r in group)
    return output


def pareto_front(metrics):
    """All values already oriented higher-is-better; no arbitrary summed score."""
    if not metrics:
        return []
    keys = set(next(iter(metrics.values())))
    if any(set(v) != keys or any(not math.isfinite(x) for x in v.values()) for v in metrics.values()):
        raise ValueError('Incomplete or nonfinite objective vector')
    return [name for name, v in metrics.items() if not any(
        all(other[k] >= v[k] for k in keys) and any(other[k] > v[k] for k in keys)
        for other in metrics.values())]
