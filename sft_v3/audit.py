"""Re-encode every prepared conversation and independently verify its targets."""
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from prepare import encode
from sft_v2.prepare import canon, digest
from sft_v3.data import FAMILIES,external_check,verify
ROOT=Path('/home/serda/sft-v3-run-20261017')


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    # V2 replay includes records inherited from v1; retain that second parent too.
    first_parent=Path('/home/serda/sft-run-20261013/manifest.json')
    first=json.loads(first_parent.read_text())
    used=list(manifest['counts'])
    replay_source=next(s for s in manifest['sources'] if s.get('source')=='v2_train_only_replay')
    replay_source['inherited_v1_sources']=[s for s in first['sources'] if any(s['repo'] in name for name in used)]
    replay_source['v1_parent_manifest_sha256']=hashlib.sha256(first_parent.read_bytes()).hexdigest()
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    assert hashlib.sha256((ROOT/'excluded_old_prompts.json.gz').read_bytes()).hexdigest()==manifest['excluded_old_prompt_hashes_sha256']
    with gzip.open(ROOT/'excluded_old_prompts.json.gz','rt') as stream:old=set(json.load(stream))
    forbidden=set();ngrams=set()
    cases=Path('/home/serda/sft-v2-run-20261015/benchmark_cases.jsonl')
    for line in cases.read_text().splitlines():
        words=canon(json.loads(line)['prompt']).split()
        forbidden.add(digest(' '.join(words)))
        ngrams.update(digest(' '.join(words[i:i+13])) for i in range(len(words)-12))
    hashes={};groups={};families=Counter();counts=Counter();peak=0
    for split in ['train','validation']:
        hashes[split]=set();groups[split]=set()
        with gzip.open(ROOT/f'{split}.jsonl.gz','rt') as stream:encoded=[json.loads(l) for l in stream]
        with gzip.open(ROOT/f'{split}_messages.jsonl.gz','rt') as stream:raw=[json.loads(l) for l in stream]
        assert len(encoded)==len(raw)
        for data,row in zip(encoded,raw):
            again=encode(row['messages'],tok)
            assert again['input_ids']==data['input_ids'] and again['labels']==data['labels']
            assert len(data['input_ids'])<=2048
            assert 519 in [v for v in data['labels'] if v!=-100]
            assert data['prompt_hash']==row['prompt_hash'] and data['group']==row['group']
            assert data['prompt_hash'] not in hashes[split]
            hashes[split].add(data['prompt_hash']);groups[split].add(data['group'])
            source=row['source'];answer=row['messages'][-1]['content']
            prompt='\n'.join(m['content'] for m in row['messages'] if m['role']=='user')
            words=canon(prompt).split();assert digest(' '.join(words))==data['prompt_hash']
            assert data['prompt_hash'] not in forbidden
            assert not any(digest(' '.join(words[i:i+13])) in ngrams for i in range(len(words)-12))
            if row.get('checks'):
                assert verify(row,answer);families[row['family']+'/'+split]+=1
            if 'source_constraints' in row:
                prompt=next(m['content'] for m in row['messages'] if m['role']=='user')
                assert external_check({'constraints':row['source_constraints'],'prompt':prompt,'messages':row['messages']}) is True
            if not source.startswith('replay/'):
                assert data['prompt_hash'] not in old
                length=len(tok(answer,add_special_tokens=False)['input_ids']);assert length<=384
                peak=max(peak,length)
            counts[source+'/'+split]+=1
    assert not hashes['train']&hashes['validation']
    assert not groups['train']&groups['validation']
    assert not hashes['validation']&old
    assert set(families)=={f+'/'+s for f in FAMILIES for s in ['train','validation']}
    assert dict(counts)==manifest['counts']
    report={'verified':True,'counts':{s:len(v) for s,v in hashes.items()},'source_counts':dict(counts),'procedural_family_counts':dict(families),'split_prompt_overlap':0,'split_group_overlap':0,'validation_previous_seen':0,'all_masks_reencoded':True,'all_procedural_targets_checked':True,'all_external_advertised_single_constraints_checked':True,'max_new_target_tokens':peak,'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()}
    (ROOT/'data_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
