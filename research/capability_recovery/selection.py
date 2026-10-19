"""Development-only non-regression checks, Pareto set, fail-closed final readiness."""
import argparse
import json
from pathlib import Path

from .common import digest, write_json
from .evaluation import paired_difference, pareto_front

REQUIRED = ['instruction', 'multiconstraint', 'termination', 'math', 'reading',
            'summary', 'grounded_qa', 'dialogue', 'code', 'knowledge', 'english_retention']


def compare(reference, candidate, policy):
    if reference.get('split') != 'dev' or candidate.get('split') != 'dev':
        raise ValueError('Historical/final test data cannot select checkpoints')
    if not reference.get('complete') or not candidate.get('complete'):
        raise ValueError('Incomplete evaluation')
    if reference['manifest_sha256'] != candidate['manifest_sha256']:
        raise ValueError('Mismatched evaluation manifest')
    result = {}
    for family in policy['families']:
        a = [r for r in reference['records'] if r['family'] == family]
        b = [r for r in candidate['records'] if r['family'] == family]
        paired = paired_difference(a, b)
        adequate = paired.get('scored', 0) >= policy['minimum_scored_per_family'] and paired.get('groups', 0) >= policy['minimum_groups_per_family']
        passed = bool(adequate and paired['difference'] >= -policy['maximum_accuracy_drop']
                      and paired['ci95'][0] >= -policy['maximum_ci_lower_drop'])
        result[family] = {**paired, 'adequate': adequate, 'nonregression': passed}
    # Score completion independently from content; format/shortening cannot replace math.
    amap = {r['id']: r for r in reference['records']}
    bmap = {r['id']: r for r in candidate['records']}
    if amap.keys() != bmap.keys():
        raise ValueError('Different evaluated IDs')
    stopping = sum(r['truncated'] for r in bmap.values()) <= sum(r['truncated'] for r in amap.values())
    return {'families': result, 'stopping_nonregression': stopping,
            'eligible_for_selection': all(r['nonregression'] for r in result.values()) and stopping,
            'language_retention_verified': False,
            'note': 'Language NLL, semantic/manual scoring and independent final gates still required'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--candidates', nargs='+', required=True)
    p.add_argument('--policy', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    policy = json.loads(Path(args.policy).read_text())['development']
    reference = json.loads(Path(args.reference).read_text())
    results, objectives = {}, {}
    for path in args.candidates:
        candidate = json.loads(Path(path).read_text())
        results[path] = compare(reference, candidate, policy)
        if results[path]['eligible_for_selection']:
            objectives[path] = {k: v['difference'] for k, v in results[path]['families'].items()}
    write_json(args.output, {'policy_sha256': digest(args.policy), 'comparisons': results,
                            'pareto_candidates': pareto_front(objectives),
                            'automatic_production_replacement': False})


if __name__ == '__main__':
    main()
