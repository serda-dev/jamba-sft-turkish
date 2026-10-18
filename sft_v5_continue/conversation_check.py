"""Generate answers for the existing 15 conversation controls; no training."""
import gc,gzip,json,os,re
from pathlib import Path
ROOT=Path('/mnt/home_extra/sft')

def unique_json(text):
 def unique(pairs):
  result={}
  for key,value in pairs:
   if key in result:raise ValueError('Duplicate key')
   result[key]=value
  return result
 return json.loads(text,object_pairs_hook=unique)

def evaluate(path,tag):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer');tok.padding_side='left'
 rows=[]
 with gzip.open(ROOT/'validation.jsonl.gz','rt') as f:
  for line in f:
   row=json.loads(line)
   if row['family']!='conversation':continue
   text=tok.decode(row['input_ids'],skip_special_tokens=False)
   messages=[{'role':role,'content':content} for role,content in re.findall(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>',text,re.S)]
   rows.append((row['prompt_hash'],messages))
 assert len(rows)==15
 model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();model.config.num_logits_to_keep=1;result=[]
 for offset in range(0,len(rows),8):
  group=rows[offset:offset+8];texts=[tok.apply_chat_template(ms[:-1],tokenize=False,add_generation_prompt=True) for _,ms in group];batch=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
  with torch.inference_mode():out=model.generate(**batch,max_new_tokens=96,do_sample=False,use_cache=True,eos_token_id=[519],pad_token_id=tok.pad_token_id)
  for (key,ms),ids in zip(group,out[:,batch['input_ids'].shape[1]:]):
   response=tok.decode(ids,skip_special_tokens=True).strip()
   try:correct=unique_json(response)==unique_json(ms[-1]['content'])
   except (ValueError,TypeError):correct=False
   result.append({'prompt_hash':key,'expected':ms[-1]['content'],'response':response,'correct':correct,'truncated':519 not in ids.tolist()})
 report={'tag':tag,'n':len(result),'correct':sum(x['correct'] for x in result),'truncated':sum(x['truncated'] for x in result),'cases':result,'scope':'Existing 15 procedural project-code conversation recall controls; not broad natural-dialogue quality.'}
 (ROOT/(tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2));del model;gc.collect();torch.cuda.empty_cache();return report

def baseline_source(cfg):
 from huggingface_hub import snapshot_download
 prefix=cfg['base_subfolder']+'/'
 return Path(snapshot_download(cfg['base_repo'],revision=cfg['base_revision'],allow_patterns=[prefix+p for p in ['*.json','*.jinja','*.safetensors']]))/cfg['base_subfolder']

if __name__=='__main__':
 import torch
 os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip();torch.set_num_threads(2);cfg=json.loads((ROOT/'config.json').read_text())
 baseline=evaluate(baseline_source(cfg),'conversation_generated_v2');candidate=evaluate(ROOT/'conversation_audit_snapshot','conversation_generated_1600')
 print(json.dumps({'baseline_correct':baseline['correct'],'candidate_correct':candidate['correct'],'n':15}),flush=True)
