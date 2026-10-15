"""Exclude v1-seen prompts from development; audit masks, hashes and leakage."""
import sys,re,json,gzip,hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sft_v2.prepare import ROOT,canon,digest

def main():
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer');prior=set()
 with gzip.open('/home/serda/sft-run-20261013/train.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);text=tok.decode(r['input_ids'],skip_special_tokens=False)
   prompts=re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',text,re.S)
   prior.add(digest(canon('\n'.join(prompts))))
 counts=Counter();tokens=Counter();assistant=Counter();seen={'train':set(),'validation':set()};moved=0
 handles={s:gzip.open(ROOT/f'{s}.audited.jsonl.gz','wt') for s in ['train','validation']}
 for split in ['train','validation']:
  with gzip.open(ROOT/f'{split}.jsonl.gz','rt') as f:
   for line in f:
    r=json.loads(line);dest='train' if r['source'].startswith('replay/') or r['prompt_hash'] in prior else split
    if dest!=split:moved+=1
    assert len(r['input_ids'])==len(r['labels'])<=3072
    assert all(y==-100 or y==x for x,y in zip(r['input_ids'],r['labels']))
    supervised=[i for i,l in enumerate(r['labels']) if l!=-100]
    assert supervised and r['labels'][0]==-100 and 519 in [r['labels'][i] for i in supervised]
    if dest=='validation':assert r['prompt_hash'] not in prior
    assert r['prompt_hash'] not in seen[dest];seen[dest].add(r['prompt_hash'])
    handles[dest].write(line);counts[f"{r['source']}/{dest}"]+=1;tokens[dest]+=len(r['input_ids']);assistant[dest]+=len(supervised)
 for h in handles.values():h.close()
 assert not seen['train']&seen['validation']
 for split in handles:(ROOT/f'{split}.audited.jsonl.gz').replace(ROOT/f'{split}.jsonl.gz')
 p=ROOT/'manifest.json';manifest=json.loads(p.read_text());manifest.update(counts=dict(counts),input_tokens=dict(tokens),assistant_tokens=dict(assistant),validation_prior_seen_prompts=0)
 manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.jsonl.gz')};p.write_text(json.dumps(manifest,indent=2))
 audit={'verified':True,'prior_v1_training_prompt_hashes':len(prior),'replay_or_seen_validation_rows_moved_to_training':moved,'train_rows':len(seen['train']),'validation_rows':len(seen['validation']),'validation_exact_overlap_with_v1_training':0,'train_validation_exact_overlap':0,'all_assistant_masks_and_eos_checked':True,'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 (ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
