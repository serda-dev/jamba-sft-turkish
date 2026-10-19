"""Audited token sampler input; prepares data and exercises a real CPU DataLoader.

No training is started. Distilled reasoning is not relabelled as verified math.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .common import digest, rows, write_json
from .evaluation import normalize, shingles
from .sampling import TARGETS, training_loader


def task_family(row):
    family = row['family']
    return 'instruction' if family == 'grounded_instruction' else family if family in TARGETS else None


def evidence(context, answer, offset):
    # Source span, never a generated justification or a "verified" publisher label.
    start = max(context.rfind('.', 0, offset), context.rfind('\n', 0, offset)) + 1
    end = context.find('.', offset + len(answer))
    return context[start:end+1 if end >= 0 else len(context)].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoded', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--eval-manifest', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--token-budget', type=int, default=3_000_000)
    a = p.parse_args()
    if a.output_dir.exists():
        raise ValueError('Refusing to overwrite prepared data')
    a.output_dir.mkdir(parents=True, mode=0o700)
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from prepare import encode
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    manifest = json.loads(Path(a.eval_manifest).read_text())
    forbidden, exact = set(), set()
    for split in manifest['splits'].values():
        path = Path(a.eval_manifest).parent/split['file']
        if digest(path) != split['sha256']:
            raise ValueError('Evaluation checksum mismatch')
        for c in rows(path):
            for text in ['\n'.join(m['content'] for m in c['messages']), c.get('source_text', ''), c.get('parallel_text', '')]:
                if text:
                    forbidden.update(shingles(text)); exact.add(normalize(text))
    rejected, counts, tokens, seen = Counter(), Counter(), Counter(), set()
    source_counts = Counter()
    metadata = []
    sources = []
    path = a.output_dir/'train.jsonl'
    with path.open('wb') as out:
        def accept(row, messages=None):
            family = task_family(row)
            if family is None:
                rejected['unclassified_'+row['family']] += 1
                return
            ids, labels = row['input_ids'], row['labels']
            if len(ids) != len(labels) or len(ids) > 3072 or labels[0] != -100:
                raise ValueError('Invalid encoded row')
            text = tok.decode(ids, skip_special_tokens=False)
            if normalize(text) in exact or forbidden & shingles(text):
                rejected['evaluation_overlap'] += 1
                return
            key = hashlib.sha256(bytes(str(ids), 'ascii')).hexdigest()
            if key in seen:
                rejected['exact_duplicate'] += 1
                return
            if any(y != -100 and y != x for x, y in zip(ids, labels)):
                raise ValueError('Corrupt labels')
            eos = tok.convert_tokens_to_ids('<|im_end|>')
            # Every supervised assistant segment must contain its termination label.
            assistant_turns = text.count('<|im_start|>assistant\n')
            if sum(x == eos for x in labels) != assistant_turns:
                raise ValueError('Missing or extraneous supervised EOS')
            n = sum(x != -100 for x in labels[1:])
            if n < 2:
                rejected['empty_target'] += 1
                return
            seen.add(key)
            row = {k: v for k, v in row.items() if k not in ['objective_weight', 'objective_component']}
            row.update(family=family, sample_id=key)
            metadata.append({'offset': out.tell(), 'family': family, 'target_tokens': n,
                             'source': row['source'], 'sample_id': key})
            out.write((json.dumps(row, ensure_ascii=False)+'\n').encode())
            counts[family] += 1; tokens[family] += n
            source_counts[row['source']] += 1
        for r in rows(a.encoded):
            accept(r)
        source = 'husnu/tquad-v1v2'; revision = '34d25445b10a49c177bc5fc6179fa260fb74f8c0'
        file = Path(hf_hub_download(source, 'tquad_train_data_v1-v2.json', repo_type='dataset', revision=revision))
        sources.append({'repo': source, 'revision': revision, 'file': file.name, 'sha256': digest(file),
                        'license': 'Undeclared upstream; local research preparation only, redistribution blocked'})
        for doc in json.loads(file.read_text())['data']:
            for para in doc['paragraphs']:
                context = para['context']
                for qa in para['qas'][:3]:
                    answers = qa.get('answers', [])
                    if not answers:
                        continue
                    ans = answers[0]; value = ans['text']
                    try:
                        offset = int(ans['answer_start'])
                    except (ValueError, TypeError):
                        rejected['invalid_source_offset_type'] += 1
                        continue
                    if context[offset:offset+len(value)] != value:
                        rejected['bad_source_answer_offset'] += 1
                        continue
                    quote = evidence(context, value, offset)
                    if value not in quote:
                        rejected['incomplete_source_evidence'] += 1
                        continue
                    ms = [{'role': 'user', 'content': 'Yalnızca kaynağa dayanarak soruyu yanıtla. Ardından yanıtı destekleyen cümleyi kaynaktan aynen alıntıla.\nKaynak:\n'+context+'\nSoru: '+qa['question']},
                          {'role': 'assistant', 'content': 'Yanıt: '+value+'\nKanıt: '+quote}]
                    data = encode(ms, tok)
                    if len(data['input_ids']) > 3072:
                        rejected['source_too_long_no_truncation'] += 1
                        continue
                    data.update(family='grounded_qa', source=source+'/train',
                                group='tquad/'+hashlib.sha256(normalize(context).encode()).hexdigest())
                    accept(data, ms)
        # Human-written Turkish task-oriented multi-turn data; never label this
        # narrow role-play corpus as evidence of broad free-conversation quality.
        repo = 'songbo/multi3woz'; rev = '72b45678a1fd22d5138491ad8e8c328d79de429a'
        fetched = {}
        for name in ['turkish/data.json', 'turkish/testListFile.txt', 'turkish/valListFile.txt']:
            file = Path(hf_hub_download(repo, name, repo_type='dataset', revision=rev))
            fetched[name] = file
            sources.append({'repo': repo, 'revision': rev, 'file': name, 'sha256': digest(file),
                            'upstream_license': 'MIT in official multi3woz repository',
                            'collection_scope': 'Human-written culturally adapted task-oriented role play'})
        heldout = set(fetched['turkish/testListFile.txt'].read_text().split()) | set(fetched['turkish/valListFile.txt'].read_text().split())
        dialogues = json.loads(fetched['turkish/data.json'].read_text())
        import random
        keys = sorted(set(dialogues)-heldout)
        random.Random(20261019).shuffle(keys)
        source_name = repo+'/turkish/upstream_train'
        for key in keys:
            ms = [{'role': 'user' if t['speaker'] == 'USER' else 'assistant', 'content': t['text']}
                  for t in dialogues[key]['log']]
            if not ms or ms[-1]['role'] != 'assistant' or any(
                    m['role'] != ('user' if i % 2 == 0 else 'assistant') or not m['content'].strip()
                    for i, m in enumerate(ms)):
                rejected['invalid_source_dialogue_roles'] += 1
                continue
            data = encode(ms, tok)
            if len(data['input_ids']) > 3072:
                rejected['long_dialogue_no_truncation'] += 1
                continue
            data.update(family='general', source=source_name, language='tr', group='multi3woz/'+key,
                        collection_scope='Human-written task-oriented multi-turn dialogue')
            accept(data, ms)
            if source_counts[source_name] >= 1000:
                break
    meta = a.output_dir/'metadata.json'
    write_json(meta, metadata)
    loader, schedule = training_loader(path, metadata, a.token_budget, tok.pad_token_id,
                                      micro_batch_size=4, max_reuse=2)
    observed, step_compositions, batch_tokens = Counter(), [], Counter()
    observed_sources, source_exposures = Counter(), Counter()
    batches = 0
    for batch in loader:
        count = int((batch['labels'][:, 1:] != -100).sum())
        if count != int(batch['target_tokens'].sum()):
            raise ValueError('Padded DataLoader supervision count differs from schedule')
        for family, n in zip(batch['families'], batch['target_tokens'].tolist()):
            observed[family] += n; batch_tokens[family] += n
        for source, n in zip(batch['sources'], batch['target_tokens'].tolist()):
            observed_sources[source] += n; source_exposures[source] += 1
        batches += 1
        if batches % 8 == 0:  # 8 x 4 micro rows = 32 example accumulation window
            step_compositions.append(dict(batch_tokens)); batch_tokens.clear()
    if batch_tokens:
        step_compositions.append(dict(batch_tokens))
    if sum(observed.values()) != schedule['sampled_tokens'] or dict(observed) != schedule['tokens_by_family']:
        raise ValueError('Actual training DataLoader did not reproduce scheduled token exposure')
    report = {'prepared_sha256': digest(path), 'metadata_sha256': digest(meta),
              'parent_encoded_sha256': digest(a.encoded), 'eval_manifest_sha256': digest(a.eval_manifest),
              'sources': sources, 'unique_examples': dict(counts), 'target_tokens': dict(tokens),
              'examples_by_source': dict(source_counts),
              'rejected': dict(rejected), 'schedule': schedule,
              'actual_dataloader': {'batches': batches, 'tokens_by_family': dict(observed),
                  'target_tokens_by_source': dict(observed_sources), 'sample_exposures_by_source': dict(source_exposures),
                  'shifted_target_tokens': sum(observed.values()), 'accumulated_step_compositions': step_compositions},
              'training_started': False, 'full_data_quality_acceptance': False,
              'remaining_limits': ['General includes English/single-turn replay plus human Turkish task-oriented dialogues; broad open-domain free-conversation diversity is not established',
                  'TQuAD supports evidence extraction, not general open-domain QA',
                  'No task label claims distilled reasoning is verified mathematics',
                  'Language shares and near-duplicate source paraphrases require further auditing before training',
                  'TQuAD license absent; publishing prepared examples is blocked']}
    write_json(a.output_dir/'manifest.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'actual_dataloader'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
