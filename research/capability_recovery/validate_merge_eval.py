"""CPU preflight on real evaluation artifacts; optional pre-generation budgets."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .common import digest, rows, write_json
from .evaluation import score_case
from .instruction_scoring import output_budget
from .prepare_merge_eval import coverage


def validate(path, prepare_budgets=False):
    path = Path(path)
    manifest = json.loads(path.read_text())
    initial = digest(path)
    if prepare_budgets and (path.parent/'FROZEN.json').exists():
        raise ValueError('Evaluation protocol already frozen')
    all_ids, groups, reports = set(), {}, {}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/home/serda/sft-v5-run-20261018/tokenizer', local_files_only=True)
    for split, spec in manifest['splits'].items():
        file = path.parent/spec['file']
        if digest(file) != spec['sha256']:
            raise ValueError('Case checksum mismatch')
        data = list(rows(file))
        if len({c['id'] for c in data}) != len(data) or all_ids & {c['id'] for c in data}:
            raise ValueError('Duplicate case identity across splits')
        all_ids.update(c['id'] for c in data)
        groups[split] = {c['group'] for c in data}
        lengths = []
        changed = 0
        for c in data:
            if score_case(c, '')['correct'] is not False:
                raise ValueError('Unscored or empty-success case')
            if c['scorer'] == 'ifeval_contract':
                score_case(c, 'Bu yalnızca puanlayıcının çalıştırılabildiğini sınayan bir cümledir.')
                budget = output_budget(c['rules'])
                if prepare_budgets and c['max_new_tokens'] != budget:
                    c['max_new_tokens'] = budget; changed += 1
                if c['max_new_tokens'] != budget:
                    raise ValueError('Word requirement exceeds registered output budget')
            else:
                gold = '#### '+c['answer'] if c['scorer'] == 'number' else c['answers'][0]
                if not score_case(c, gold)['correct']:
                    raise ValueError('Gold scorer roundtrip failed')
            if c['family'] == 'dialogue' and c['answers'][0] not in '\n'.join(m['content'] for m in c['messages'][:-1]):
                raise ValueError('Dialogue memory answer unsupported by source')
            text = tok.apply_chat_template(c['messages'], tokenize=False, add_generation_prompt=True)
            lengths.append(len(tok.encode(text, add_special_tokens=False)))
        if max(lengths) > 4096:
            raise ValueError('Prompt exceeds registered input bound')
        if prepare_budgets:
            file.write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in data))
            manifest['splits'][split]['sha256'] = digest(file)
        reports[split] = {'coverage': coverage(data), 'prompt_tokens_max': max(lengths),
                          'output_budget_counts': dict(Counter(c['max_new_tokens'] for c in data)),
                          'math_sources': dict(Counter(c['source'] for c in data if c['family'] == 'math')),
                          'budgets_prepared': changed, 'all_scorers_executed': len(data)}
    if groups['dev'] & groups['final']:
        raise ValueError('Cross-split source group leakage')
    if not all(v['ready'] for v in reports['dev']['coverage'].values()):
        raise ValueError('Development coverage gate failed')
    if prepare_budgets:
        manifest['readiness_blockers'] = ['Final multi-constraint family has fewer than 64 scored cases/32 groups',
            'Instruction scoring is contract compliance, not factual correctness',
            'Dialogue memory does not certify free-conversation fluency',
            'No full eleven-family release acceptance or TR/EN NLL measurements']
        manifest['output_budget_policy'] = '768 default; instruction min words 165..512:2048; above512:4096; identical for every model, fixed before outputs'
        write_json(path, manifest)
    report = {'passed': True, 'manifest_sha256': digest(path), 'initial_manifest_sha256': initial,
              'budgets_prepared_before_outputs': prepare_budgets, 'splits': reports,
              'source_group_disjoint': True, 'production_acceptance_ready': False}
    write_json(path.parent/'cpu_preflight.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--prepare-budgets', action='store_true')
    args = p.parse_args()
    print(json.dumps(validate(args.manifest, args.prepare_budgets), ensure_ascii=False, indent=2))
