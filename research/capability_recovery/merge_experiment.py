"""Six-model development comparison; no rental or training side effects."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .common import digest, write_json
from .evaluation import paired_difference

FAMILIES = ['math', 'instruction', 'multiconstraint', 'reading', 'dialogue', 'termination']


def compare_results(results):
    """Pre-registered two-parent gates; no historical test or aggregate score."""
    comparisons = {}
    for name, candidate in results.items():
        if name in ['v2', 'v5']:
            continue
        by_parent = {}
        for parent in ['v2', 'v5']:
            ref = results[parent]
            if not candidate['complete'] or not ref['complete'] or ref['manifest_sha256'] != candidate['manifest_sha256']:
                raise ValueError('Incomplete or mismatched evaluation')
            families = {}
            for family in FAMILIES:
                a, b = [[r for r in x['records'] if family in r.get('analysis_families', [r['family']])]
                        for x in [ref, candidate]]
                if family == 'termination':
                    # Correct content and a real EOS are both required. Do not
                    # reward an empty/incorrect answer just because it stops.
                    a, b = [[{**r, 'correct': r['correct'] is True and r['eos_terminated'] and
                              r['repetition_8gram'] <= .2} for r in rs] for rs in [a, b]]
                difference = paired_difference(a, b)
                adequate = difference['scored'] >= 64 and difference.get('groups', 0) >= 32
                families[family] = {**difference, 'adequate': adequate,
                    'nonregression': bool(adequate and difference['difference'] >= -.03 and difference['ci95'][0] >= -.08)}
            by_parent[parent] = families
        # Clear superiority needs a positive paired CI versus BOTH parents in
        # at least one same family, and no family failing either retention gate.
        gains = [f for f in FAMILIES if all(by_parent[p][f]['ci95'] and by_parent[p][f]['ci95'][0] > 0
                                          for p in ['v2', 'v5'])]
        eligible = all(v['nonregression'] for fs in by_parent.values() for v in fs.values())
        comparisons[name] = {'parents': by_parent, 'nonregression_both': eligible,
                              'clear_gain_families_both': gains,
                              'development_superior': eligible and bool(gains)}
    superior = [k for k, v in comparisons.items() if v['development_superior']]
    return {'comparisons': comparisons, 'superior': superior,
            'decision': 'candidate_requires_independent_confirmation' if superior else 'no_demonstrated_superior_merge',
            'production_accepted': False,
            'scope': 'Instruction contract success, numeric math answers, short extractive reading and dialogue memory; not full semantic/general ability acceptance',
            'multiplicity': 'Exploratory paired CIs across 4 candidates and 6 overlapping families, no multiplicity-adjusted significance claim'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent-a', required=True)
    p.add_argument('--parent-b', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    a = p.parse_args()
    from .gpu_experiments import evaluate
    a.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {'parents': ['v2', 'v5'], 'alphas': [.15, .30, .50, .70],
                'manifest_sha256': digest(a.manifest), 'batch_size': a.batch_size,
                'family_gates': {'min_scored': 64, 'min_groups': 32, 'max_drop': .03, 'max_ci_lower_drop': .08},
                'superiority': 'A paired CI lower bound >0 against both parents in the same family; both-parent nonregression in every family',
                'historical_test_used': False, 'training_enabled': False}
    path = a.output_dir/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Refusing to change experiment protocol on resume')
    write_json(path, protocol)
    results = {}
    for name, model, parent_b, alpha in [('v2', a.parent_a, None, 0.), ('v5', a.parent_b, None, 1.),
            *[(f'alpha_{x:.2f}', a.parent_a, a.parent_b, x) for x in protocol['alphas']]]:
        output = a.output_dir/(name+'.json')
        args = SimpleNamespace(model=model, parent_b=parent_b, alpha=alpha, seed=20261019,
                               manifest=a.manifest, split='dev', selection=None,
                               batch_size=a.batch_size, code_image=None, output=str(output))
        # Do not silently skip incomplete generations or rerun a different manifest.
        if output.exists():
            result = json.loads(output.read_text())
            if result.get('complete') and result.get('manifest_sha256') == protocol['manifest_sha256']:
                results[name] = result
                continue
        print('EVALUATING', name, flush=True)
        evaluate(args)
        results[name] = json.loads(output.read_text())
    report = compare_results(results)
    write_json(a.output_dir/'comparison.json', report)
    print(json.dumps({'decision': report['decision'], 'superior': report['superior']}), flush=True)


if __name__ == '__main__':
    main()
