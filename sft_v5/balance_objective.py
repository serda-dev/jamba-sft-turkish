"""Set explicit objective weights without duplicating or truncating training rows."""
import gzip,hashlib,json,shutil
from collections import Counter
from pathlib import Path
ROOT=Path('/home/serda/sft-v5-run-20261018')
SHARES={'general_tr':.25,'math_reasoning':.25,'instruction':.20,'reading':.15,'code':.10,'general_en':.05}

def component(row):
 if 'ultrachat' in row['source']:return 'general_en'
 if row['family'] in ['math','reasoning']:return 'math_reasoning'
 if row['family'] in ['instruction','grounded_instruction']:return 'instruction'
 if row['family'] in ['reading','code']:return row['family']
 return 'general_tr'

def main():
 counts=Counter()
 with gzip.open(ROOT/'train.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);counts[component(r)]+=r['n_target_tokens']
 assert set(counts)==set(SHARES)
 total=sum(counts.values());weights={k:SHARES[k]/(v/total) for k,v in counts.items()}
 with gzip.open(ROOT/'train.jsonl.gz','rt') as src,gzip.open(ROOT/'train.balanced.jsonl.gz','wt') as dst:
  for line in src:
   r=json.loads(line);r['objective_component']=component(r);r['objective_weight']=weights[component(r)];dst.write(json.dumps(r,ensure_ascii=False)+'\n')
 (ROOT/'train.balanced.jsonl.gz').replace(ROOT/'train.jsonl.gz')
 p=ROOT/'manifest.json';m=json.loads(p.read_text());m['objective']={'component_target_tokens':dict(counts),'raw_token_shares':{k:v/total for k,v in counts.items()},'weights':weights,'desired_weighted_token_shares':SHARES,'normalization':'Weighted target tokens are normalized within each optimizer batch; shares are exact over the dataset mass, not a claim of exact gradient contributions.'};m['files']['train.jsonl.gz']=hashlib.sha256((ROOT/'train.jsonl.gz').read_bytes()).hexdigest();p.write_text(json.dumps(m,indent=2))
 a=ROOT/'data_audit.json';d=json.loads(a.read_text());d['manifest_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();d['objective_weights_applied_without_text_or_label_changes']=True;a.write_text(json.dumps(d,indent=2))
 c=ROOT/'config.json';cfg=json.loads(c.read_text());cfg['loss_weighting']='Component-weighted assistant-token mean per optimizer batch';cfg['objective_shares']=SHARES;c.write_text(json.dumps(cfg,indent=2))
 for name in ['train.jsonl.gz','manifest.json','data_audit.json','config.json']:shutil.copy2(ROOT/name,ROOT/'hub_publish'/name)
 print(json.dumps(m['objective'],indent=2))

if __name__=='__main__':main()
