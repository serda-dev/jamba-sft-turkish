"""Recompute saved results on CPU; does not run inference or select checkpoints."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'benchmarks')]
from numeric_score import numeric
from scoring import score
from sft_v4.data import verify


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    load = lambda name: json.loads((root / name).read_text())
    cases = {r['id']: r for r in load('dev_cases.json') + load('retention_cases.json')}
    checked = 0
    for path in sorted(root.glob('*.json')):
        if not path.name.startswith(('baseline_dev', 'baseline_math', 'baseline_reading', 'dev_', 'control_')):
            continue
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or 'cases' not in report:
            continue
        assert len(report['cases']) == report['n'], path.name
        for result in report['cases']:
            flags = verify(cases[result['id']], result['response'])
            assert all(result[k] == v for k, v in flags.items()), (path.name, result['id'])
            checked += 1
        for key, field in [('correct', 'correct'), ('format_correct', 'format'), ('content_correct', 'content'), ('truncated', 'truncated')]:
            assert report[key] == sum(r[field] for r in report['cases']), (path.name, key)
    rows = [json.loads(line) for line in (root / 'final_benchmark.jsonl').read_text().splitlines()]
    frozen = {r['id']: r for r in map(json.loads, (root / 'benchmark_cases.jsonl').read_text().splitlines())}
    assert len(rows) == len(frozen) == len({r['id'] for r in rows}) == 2320
    for row in rows:
        assert all(row[k] == v for k, v in frozen[row['id']].items()), row['id']
        if row['type'] == 'mcq':
            prediction = max(range(len(row['choice_logits'])), key=row['choice_logits'].__getitem__)
            correct = prediction == row['gold']
            assert prediction == row['prediction']
        elif row['type'] == 'math':
            response = row['response']
            numbers = re.findall(r'-?\d+(?:[.,]\d+)*', response.rsplit('####', 1)[-1] if '####' in response else response)
            prediction = numbers[-1] if numbers else None
            correct = prediction is not None and numeric(prediction) == numeric(row['gold'])
            assert prediction == row['prediction']
        else:
            flags = score(row['rules'], row['response'])
            assert flags == row['checks']
            correct = all(flags)
        assert correct == row['correct'], row['id']
    summary = load('final_benchmark_summary.json')
    baseline = {r['id']: r for r in map(json.loads, (args.baseline / 'final_benchmark.jsonl').read_text().splitlines())}
    transitions = {}
    for task, stats in summary.items():
        group = [r for r in rows if r['task'] == task]
        assert stats['n'] == len(group)
        assert stats['correct'] == sum(r['correct'] for r in group)
        assert stats['truncated'] == sum(r.get('truncated', False) for r in group)
        transitions[task] = dict(Counter(('both_correct' if baseline[r['id']]['correct'] and r['correct'] else 'lost' if baseline[r['id']]['correct'] else 'gained' if r['correct'] else 'both_wrong') for r in group))
    output = {
        'development_responses_rescored': checked,
        'benchmark_responses_rescored': len(rows),
        'mismatches': 0,
        'benchmark_sha256': hashlib.sha256((root / 'final_benchmark.jsonl').read_bytes()).hexdigest(),
        'paired_transitions_from_v2': transitions,
        'scope': 'Saved-response scoring and fixed-case identity audit; no new inference; truncation flags cannot be independently reconstructed from decoded text.',
    }
    (root / 'postrun_scoring_audit.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
