"""Final fixed protocol comparison; no test feedback enters training selection."""
import json,re,inspect,time,sys,hashlib
from pathlib import Path
ROOT=Path('/mnt/home_extra/sft')

def run(path,cfg):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 sys.path.insert(0,str(ROOT/'benchmark_code'))
 from scoring import score
 from numeric_score import numeric
 tok=AutoTokenizer.from_pretrained(path);tok.padding_side='left'
 model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();model.config.use_cache=True;model.config.num_logits_to_keep=1
 cases=[json.loads(l) for l in (ROOT/'benchmark_cases.jsonl').read_text().splitlines()];out=ROOT/'final_benchmark.jsonl';out.write_text('')
 stats={}
 for task in ['turkish_mmlu','belebele_tr','gsm8k_tr','ifeval_tr_subset','gsm8k_tr_cot8']:
  if (ROOT/'STOP').exists():break
  items=[r for r in cases if r['task']==task];size=16
  for offset in range(0,len(items),size):
   if (ROOT/'STOP').exists():break
   group=items[offset:offset+size];texts=[tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True) for r in group];batch=tok(texts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
   if group[0]['type']=='mcq':
    opts=[tok.encode(chr(65+j),add_special_tokens=False) for j in range(5)];assert all(len(v)==1 for v in opts)
    with torch.inference_mode():logits=model(**batch,use_cache=False).logits[:,-1].float()
    results=[]
    for i,r in enumerate(group):
     values=logits[i,[t[0] for t in opts[:r['choices']]]].tolist();p=max(range(len(values)),key=lambda j:values[j]);results.append({**r,'prediction':p,'correct':p==r['gold'],'choice_logits':values})
   else:
    limit=1024 if task=='gsm8k_tr_cot8' else 512 if task=='gsm8k_tr' else 2048
    with torch.inference_mode():outputs=model.generate(**batch,max_new_tokens=limit,do_sample=False,use_cache=True,eos_token_id=[519],pad_token_id=tok.pad_token_id)
    results=[]
    for r,ids in zip(group,outputs[:,batch['input_ids'].shape[1]:]):
     response=tok.decode(ids,skip_special_tokens=True).strip();flags=None;p=None
     if r['type']=='math':
      nums=re.findall(r'-?\d+(?:[.,]\d+)*',response.rsplit('####',1)[-1] if '####' in response else response);p=nums[-1] if nums else None;ok=p is not None and numeric(p)==numeric(r['gold'])
     else:flags=score(r['rules'],response);ok=all(flags)
     results.append({**r,'response':response,'prediction':p,'correct':ok,'checks':flags,'truncated':519 not in ids.tolist()})
   with out.open('a') as f:
    for r in results:f.write(json.dumps(r,ensure_ascii=False)+'\n')
   print('FINAL_BENCHMARK',task,offset+len(group),len(items),flush=True)
  done=[json.loads(l) for l in out.read_text().splitlines() if json.loads(l)['task']==task]
  stats[task]={'n':len(done),'correct':sum(r['correct'] for r in done),'accuracy':sum(r['correct'] for r in done)/max(len(done),1),'truncated':sum(r.get('truncated',False) for r in done)}
  (ROOT/'final_benchmark_summary.json').write_text(json.dumps(stats,indent=2))
 return stats
