"""Source-backed Turkish merge development suite, with explicit scoring scope."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .common import digest, rows, write_json
from .evaluation import group_split, normalize, shingles
from .instruction_scoring import checks, output_budget
from .prepare_eval import make_case, exclude_historical

SOURCES = [
    ('bezir/IFEval_tr', '07e8c2cf4ae66a86fba57d389a73cbc10e865d5d', 'data/train-00000-of-00001.parquet'),
    ('AdnanElAssadi/Google-Translated_Turkish_ifeVal_Dataset', '7d16e5e4f5b14022742bd82771e296bc9656d506', 'data/train-00000-of-00001.parquet'),
    ('barandinho/amc_turkish', 'f4926237ac9300327c01badbdaf375b09f9428ab', 'data/test-00000-of-00001.parquet'),
    ('songbo/multi3woz', '72b45678a1fd22d5138491ad8e8c328d79de429a', 'turkish/data.json'),
    ('songbo/multi3woz', '72b45678a1fd22d5138491ad8e8c328d79de429a', 'turkish/testListFile.txt'),
    ('husnu/tquad-v1v2', '34d25445b10a49c177bc5fc6179fa260fb74f8c0', 'tquad_dev_data_v1-v2.json'),
    ('malhajar/gsm8k-tr', '1046702fc5d8e14330ae90fdfd98735194ffa4ce', 'main/test-1318.jsonl'),
]


def fetch():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    files, receipts = {}, []
    for repo, revision, name in SOURCES:
        p = Path(hf_hub_download(repo, name, revision=revision, repo_type='dataset'))
        files[(repo, name)] = p
        receipts.append({'repo': repo, 'revision': revision, 'file': name, 'sha256': digest(p)})
    return files, receipts


def pools(files, historical):
    import pandas as pd
    cases, rejected = [], Counter()
    bezir = pd.read_parquet(files[(SOURCES[0][0], SOURCES[0][2])]).to_dict('records')
    old_keys = {int(bezir[r['source_row']]['key']) for r in rows(historical)
                if r['task'] == 'ifeval_tr_subset'}
    used = set()
    for source in SOURCES[:2]:
        for row in pd.read_parquet(files[(source[0], source[2])]).to_dict('records'):
            key = int(row['key'])
            if key in old_keys or key in used:
                rejected['historical_or_parallel_ifeval_key'] += 1
                continue
            ids = ast.literal_eval(row['instruction_id_list']) if isinstance(row['instruction_id_list'], str) else list(row['instruction_id_list'])
            kwargs = ast.literal_eval(row['kwargs']) if isinstance(row['kwargs'], str) else list(row['kwargs'])
            kwargs = json.loads(json.dumps(kwargs, default=lambda v: v.tolist()))
            rule = {'instruction_id_list': ids, 'kwargs': kwargs, 'prompt': row['prompt']}
            # Additional translation: textual arguments must survive in the prompt.
            # English-output requests are not a Turkish instruction probe.
            if source[0] != SOURCES[0][0]:
                if any(k.startswith(('change_case:', 'language:')) for k in ids) or 'İngilizce' in row['prompt']:
                    rejected['translation_language_or_case'] += 1
                    continue
                if any(isinstance(v, str) and k not in ['relation', 'capital_relation', 'let_relation']
                       and v and v not in row['prompt'] for kw in kwargs for k, v in kw.items() if v is not None):
                    rejected['translated_literal_mismatch'] += 1
                    continue
                if any(isinstance(kw.get('keywords'), (list, tuple)) and
                       any(v not in row['prompt'] for v in kw['keywords']) for kw in kwargs):
                    rejected['translated_keyword_mismatch'] += 1
                    continue
            try:
                checks(rule)
            except (ValueError, KeyError, TypeError):
                rejected['unsupported_or_underspecified_contract'] += 1
                continue
            used.add(key)
            tags = ['instruction', 'termination'] + (['multiconstraint'] if len(ids) >= 2 else [])
            cases.append(make_case('ifeval-new/'+str(key), 'instruction', 'tr', row['prompt'], 'ifeval_contract',
                group='ifeval-original/'+str(key), rules=rule, source=source[0], analysis_families=tags,
                semantic_scope='Declared instruction contracts only; topical/factual correctness unscored',
                max_new_tokens=output_budget(rule)))
    source = SOURCES[2]
    for row in pd.read_parquet(files[(source[0], source[2])]).to_dict('records'):
        if not __import__('re').fullmatch(r'-?\d+(?:\.\d+)?', str(row['answer'])):
            rejected['non_numeric_amc'] += 1
            continue
        visual = ['\\begin{asy}', '[asy]', '\\includegraphics', '![', 'diagram', 'figure below']
        if len(row['problem_translated']) < 50 or any(v in row['problem'].lower() for v in visual):
            rejected['visual_or_short_amc'] += 1
            continue
        group = 'amc/'+row['url'].split('/wiki/index.php/')[-1]
        cases.append(make_case('amc/'+row['id'], 'math', 'tr', row['problem_translated']+
            '\nÇözümü açıkla. Son yanıtı ayrı bir satırda #### sayı biçiminde yaz.', 'number',
            group=group, answer=row['answer'], source=source[0], source_text=row['problem_translated'],
            parallel_text=row['problem'], analysis_families=['math', 'termination'],
            semantic_scope='Numerical final answer against source key; intermediate proof not independently verified',
            max_new_tokens=768))
    docs = json.loads(files[(SOURCES[3][0], SOURCES[3][2])].read_text())
    test_ids = set(files[(SOURCES[4][0], SOURCES[4][2])].read_text().split())
    slots = {'Name': 'yerin adını', 'Depart': 'kalkış yerini', 'Dest': 'varış yerini',
             'Day': 'günü', 'Time': 'saati', 'Leave': 'hareket saatini',
             'Arrive': 'varış saatini', 'People': 'kişi sayısını', 'Stay': 'gece sayısını',
             'Ref': 'rezervasyon kodunu', 'Stars': 'yıldız sayısını', 'Phone': 'telefon numarasını'}
    for key in sorted(test_ids):
        if key not in docs:
            continue
        log = docs[key]['log']
        eligible = []
        for i, turn in enumerate(log[:-2]):
            if turn['speaker'] != 'USER':
                continue
            for span in turn.get('span_info', []):
                act, slot, value, start, end = span
                # Require literal source agreement; translated English metadata is
                # NOT assumed to be an independently verified Turkish answer.
                if slot in slots and len(value) >= 1 and value in turn['text']:
                    eligible.append((i, slot, value, act))
        if not eligible:
            continue
        i, slot, value, act = random.Random('dialogue/'+key).choice(eligible)
        # End on an actual assistant turn and ask a new memory question. Reject
        # later updates to the same slot; the reference must remain unambiguous.
        stop = min(len(log), i+6)
        if stop % 2:
            stop -= 1
        if any(s[1] == slot and s[2] != value for t in log[i+1:stop]
               if t['speaker'] == 'USER' for s in t.get('span_info', [])):
            continue
        history = [{'role': 'user' if t['speaker'] == 'USER' else 'assistant', 'content': t['text']}
                   for t in log[:stop]]
        history.append({'role': 'user', 'content': 'Daha önce belirttiğim '+slots[slot]+
                        ' tekrar söyler misin? Yalnızca o bilgiyi yaz.'})
        c = make_case('multi3woz/'+key, 'dialogue', 'tr', '', 'exact', group='multi3woz/'+key,
                      answers=[value], source=SOURCES[3][0], memory_slot=slot,
                      analysis_families=['dialogue', 'termination'],
                      semantic_scope='Literal information recall in real human multi-turn Turkish dialogue; not a free-dialogue fluency score',
                      max_new_tokens=768)
        c['messages'] = history
        cases.append(c)
    for doc in json.loads(files[(SOURCES[5][0], SOURCES[5][2])].read_text())['data']:
        for para in doc['paragraphs']:
            context = para['context']
            group = 'tquad/'+hashlib.sha256(normalize(context).encode()).hexdigest()
            for qa in para['qas'][:1]:
                answers = sorted({a['text'] for a in qa.get('answers', []) if a['text'] in context})
                if not answers:
                    continue
                cases.append(make_case(group+'/'+str(qa['id']), 'reading', 'tr',
                    f'Bağlam:\n{context}\n\nSoru: {qa["question"]}\nYalnızca bağlamdaki ilgili kısa ifadeyi yaz.',
                    'exact', group=group, answers=answers, source=SOURCES[5][0], source_text=context,
                    analysis_families=['reading', 'termination'], max_new_tokens=768))
    # Prior V5 sealed and retention questions have already influenced research.
    # Exclude their full source prompts too, not merely the frozen 200 benchmark.
    prior_math = []
    for path in [Path('/home/serda/sft-v5-run-20261018/sealed_math_cases.json'),
                 Path('/home/serda/sft-v5-run-20261018/retention_cases.json')]:
        if not path.exists():
            raise FileNotFoundError('Prior math provenance required: '+str(path))
        prior_math.extend('\n'.join(m['content'] for m in c['messages']) for c in json.loads(path.read_text()))
    prior_spans = set().union(*(shingles(t) for t in prior_math))
    for i, row in enumerate(rows(files[(SOURCES[6][0], SOURCES[6][2])])):
        if prior_spans & shingles(row['question']) or 'cıvata' in row['question']:
            rejected['previous_math_probe_or_bad_literal_translation'] += 1
            continue
        answer = row['answer'].rsplit('####', 1)[-1].strip().replace(',', '')
        if not __import__('re').fullmatch(r'-?\d+(?:\.\d+)?', answer):
            continue
        cases.append(make_case(f'gsm-new/{i}', 'math', 'tr', row['question']+
            '\nÇözümü açıkla. Son yanıtı ayrı bir satırda #### sayı biçiminde yaz.', 'number',
            group=f'gsm8k-source/test/{i}', answer=answer, source=SOURCES[6][0],
            source_text=row['question'], analysis_families=['math', 'termination'],
            semantic_scope='Natural translated grade-school final answer; solution validity unscored', max_new_tokens=768))
    unique = {}
    for c in cases:
        if c['id'] in unique:
            rejected['duplicate_source_id'] += 1
            continue
        unique[c['id']] = c
    return list(unique.values()), dict(rejected)


def coverage(cases):
    result = {}
    for family in ['instruction', 'multiconstraint', 'termination', 'math', 'reading', 'dialogue']:
        selected = [c for c in cases if family in c['analysis_families']]
        result[family] = {'scored_cases': len(selected), 'groups': len({c['group'] for c in selected}),
                          'ready': len(selected) >= 64 and len({c['group'] for c in selected}) >= 32}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--training-files', type=Path, nargs='+', required=True)
    p.add_argument('--historical-cases', required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise ValueError('Refusing to overwrite evaluation data')
    a.output_dir.mkdir(mode=0o700, parents=True)
    files, receipts = fetch()
    cases, rejected = pools(files, a.historical_cases)
    # Bound expensive exclusion work before the scan, independent of model outcomes.
    random.Random(20261019).shuffle(cases)
    counts = Counter(); bounded = []
    for c in cases:
        if counts[c['family']] < (1000 if c['family'] != 'instruction' else 500):
            bounded.append(c); counts[c['family']] += 1
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    cases, leakage = exclude_historical(bounded, a.training_files, tok, a.historical_cases)
    write_json(a.output_dir/'filtered_pool.json', {'cases': cases, 'sources': receipts,
                                                  'rejected': rejected, 'leakage': leakage})
    assignment = group_split(cases)
    # Instruction questions are scarce. Reserve 64 multi-constraint source groups
    # for development before any model output; final remains explicitly unqualified.
    multi_groups = sorted({assignment[c['id']][1] for c in cases if 'multiconstraint' in c['analysis_families']})
    if len(multi_groups) < 64:
        raise ValueError(f'Only {len(multi_groups)} independent multi-constraint groups')
    reserve = set(multi_groups[:64])
    selected = {'dev': [], 'final': []}
    family_counts = {'dev': Counter(), 'final': Counter()}
    selected_groups = {'dev': set(), 'final': set()}
    for c in sorted(cases, key=lambda c: c['id']):
        split, group = assignment[c['id']]
        if group in reserve:
            split = 'dev'
        c = {**c, 'group': group}
        if (c['family'], group) in selected_groups[split]:
            continue
        limit = 64 if c['family'] != 'instruction' else 256
        if c['family'] == 'math':
            source_kind = 'gsm' if c['id'].startswith('gsm-') else 'amc'
            if family_counts[split]['math_'+source_kind] >= 32:
                continue
        if family_counts[split][c['family']] >= limit:
            continue
        selected[split].append(c); family_counts[split][c['family']] += 1
        if c['family'] == 'math':
            family_counts[split]['math_'+source_kind] += 1
        selected_groups[split].add((c['family'], group))
    # Keep exactly 64 multi-constraint questions plus up to 16 single constraints.
    dev_instruction = [c for c in selected['dev'] if c['family'] == 'instruction']
    multi = [c for c in dev_instruction if 'multiconstraint' in c['analysis_families']][:64]
    singles = [c for c in dev_instruction if 'multiconstraint' not in c['analysis_families']][:16]
    selected['dev'] = [c for c in selected['dev'] if c['family'] != 'instruction'] + multi + singles
    manifest = {'seed': 20261019, 'sources': receipts, 'source_rejections': rejected,
                'leakage': leakage, 'training_hashes': {str(x): digest(x) for x in a.training_files},
                'splits': {}, 'development_ready': all(v['ready'] for v in coverage(selected['dev']).values()),
                'final_ready': False,
                'readiness_blockers': ['Independent final conversation pool not yet prepared',
                    'Instruction scores certify contracts, not factual/semantic quality',
                    'Dialogue discrimination does not certify natural generated conversation',
                    'No full eleven-family release acceptance or TR/EN NLL measurements'],
                'scope': 'Six-family merge feasibility development experiment; no production acceptance',
                'overlap_policy': 'Multi-constraint is a subset of instruction. Termination reuses all generations; never sum these overlapping family counts.',
                'selection_reservation': '64 multi-constraint source groups reserved before model outputs; remaining deterministic groups retained outside development',
                'alphas': [.15, .30, .50, .70]}
    for split, data in selected.items():
        path = a.output_dir/(split+'.jsonl')
        path.write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in data)); path.chmod(0o600)
        manifest['splits'][split] = {'file': path.name, 'sha256': digest(path), 'n': len(data), 'coverage': coverage(data)}
    write_json(a.output_dir/'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
