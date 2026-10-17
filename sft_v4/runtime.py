"""Full-weight SFT with BF16 working weights, FP32 masters, a gated 100-step pilot."""
import gc,gzip,hashlib,json,math,os,random,shutil,sys,time
from pathlib import Path
ROOT=Path('/mnt/home_extra/sft')

def write(name,value):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)
def status(stage,**kw):
 r={'stage':stage,'updated_at':time.time(),**kw};write('status.json',r)
 with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(stage,json.dumps(kw,ensure_ascii=False),flush=True)
def load_rows(name):
 with gzip.open(ROOT/(name+'.jsonl.gz'),'rt') as f:return [json.loads(l) for l in f]

def evaluate(model,tok,cases,tag,cfg):
 import torch
 from instruction_data import verify
 model.eval();model.config.num_logits_to_keep=1;tok.padding_side='left';results=[]
 for offset in range(0,len(cases),8):
  if (ROOT/'STOP').exists():break
  group=cases[offset:offset+8];texts=[tok.apply_chat_template(r['messages'][:-1],tokenize=False,add_generation_prompt=True) for r in group];batch=tok(texts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   if group[0]['family']=='reading':
    logits=model(**batch,use_cache=False).logits[:,-1].float();letters=[tok.encode(c,add_special_tokens=False) for c in 'ABCD'];assert all(len(v)==1 for v in letters)
    predicted=logits[:,[v[0] for v in letters]].argmax(-1).tolist();responses=[chr(65+j) for j in predicted];truncated=[False]*len(group)
   else:
    limit=128 if group[0]['family']=='math' else cfg['dev_max_new_tokens']
    outputs=model.generate(**batch,max_new_tokens=limit,do_sample=False,use_cache=True,eos_token_id=[519],pad_token_id=tok.pad_token_id)
    generated=outputs[:,batch['input_ids'].shape[1]:];responses=tok.batch_decode(generated,skip_special_tokens=True);truncated=[519 not in row.tolist() for row in generated]
  for row,response,cut in zip(group,responses,truncated):
   response=response.strip();flags=verify(row,response);words=response.split();grams=[tuple(words[i:i+8]) for i in range(len(words)-7)];rep=bool(grams and 1-len(set(grams))/len(grams)>.2)
   results.append({'id':row['id'],'family':row['family'],'response':response,**flags,'truncated':cut,'repetitive_8gram_gt20pct':rep})
 report={'tag':tag,'n':len(results),'correct':sum(r['correct'] for r in results),'format_correct':sum(r['format'] for r in results),'content_correct':sum(r['content'] for r in results),'truncated':sum(r['truncated'] for r in results),'repetitive':sum(r['repetitive_8gram_gt20pct'] for r in results),'cases':results}
 write(tag+'.json',report);status('DEVELOPMENT',**{k:v for k,v in report.items() if k!='cases'});model.config.num_logits_to_keep=0
 return report

def controls(model,tok,cases,tag,cfg):
 result={}
 for family in ['math','reading']:result[family]=evaluate(model,tok,[r for r in cases if r['family']==family],tag+'_'+family,cfg)
 return result

def language_scores(model,tok,tag):
 import torch
 from src.data.dataloader import ShardedMemmapPackedDataset
 from src.data.tokenize_pack import tokenization_fingerprint
 from transformers import AutoTokenizer
 # Generation mutates fast-tokenizer backend padding; validate cache identity against a fresh frozen tokenizer.
 frozen_tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
 p=ROOT/'cpt_validation/manifest.json';ds=ShardedMemmapPackedDataset(p);assert ds.manifest['cache_identity']['split']=='validation';assert ds.manifest['tokenizer_fingerprint']==tokenization_fingerprint(frozen_tok)
 model.eval();model.config.num_logits_to_keep=0;sums={n:{'nll':0.,'tokens':0} for n in ['turkish','english']}
 with torch.inference_mode():
  for i in range(len(ds)):
   if (ROOT/'STOP').exists():break
   item=ds[i];ids=item['input_ids'].unsqueeze(0).to('cuda');mask=item['attention_mask'].unsqueeze(0).to('cuda');logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
   losses=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none')
   for ident,name in ds.source_id_to_name.items():
    chosen=(item['token_source_ids'][1:].to('cuda')==ident)&mask[0,1:].bool();sums[name]['nll']+=float(losses[chosen].sum());sums[name]['tokens']+=int(chosen.sum())
   if i%64==0:status('LANGUAGE_RETENTION',tag=tag,rows=i+1,total=len(ds))
 result={'heldout_manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scores':{n:{'mean_nll':v['nll']/max(v['tokens'],1),'tokens':v['tokens']} for n,v in sums.items()},'complete':i+1==len(ds) and not (ROOT/'STOP').exists()}
 write(tag+'.json',result);return result

def retention_ok(prior,new,cfg):
 return bool(new['complete'] and prior['heldout_manifest_sha256']==new['heldout_manifest_sha256'] and all(new['scores'][n]['tokens']==prior['scores'][n]['tokens'] and new['scores'][n]['mean_nll']<=prior['scores'][n]['mean_nll']*cfg['acceptance'][n+'_nll_max_ratio_to_v2'] for n in ['turkish','english']))

def pilot_gate(baseline,new,base_controls,new_controls,language,prior,cfg):
 complete=new['n']==128 and all(new_controls[k]['n']==64 for k in ['math','reading'])
 improvement=new['correct']>=baseline['correct']+13 or (new['correct']>=122 and new['correct']>=baseline['correct'])
 skills=all(new_controls[k]['correct']>=base_controls[k]['correct']-4 for k in ['math','reading'])
 stop=new['truncated']<=baseline['truncated']+2
 gates={'complete':complete,'new_dev_improvement':improvement,'math_reading_retention':skills,'stopping':stop,'language_retention':retention_ok(prior,language,cfg)}
 return {'passed':all(gates.values()),'gates':gates,'baseline_correct':baseline['correct'],'pilot_correct':new['correct'],'baseline_controls':{k:v['correct'] for k,v in base_controls.items()},'pilot_controls':{k:v['correct'] for k,v in new_controls.items()}}

def save_model(model,tok,name,provenance,upload=True):
 from src.train.hub import upload_checkpoint
 folder=ROOT/name;model.config.use_cache=True;model.save_pretrained(folder,safe_serialization=True)
 # Preserve the frozen tokenizer files; fast generation mutates backend padding state.
 for file in (ROOT/'tokenizer').iterdir():
  if file.is_file():shutil.copy2(file,folder/file.name)
 (folder/'sft_provenance.json').write_text(json.dumps(provenance,indent=2));model.config.use_cache=False
 if upload:
  receipt=upload_checkpoint(folder,provenance['output_repo']);write('checkpoint_receipt.json',receipt);return receipt
 return None

def main():
 import torch,bitsandbytes as bnb
 from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
 from huggingface_hub import snapshot_download,HfApi
 from scripts.cpt_gpu_preflight import check_gpu_runtime
 from precision import make_master_parameters,copy_gradients_to_master,copy_master_to_working,sampled_parameters,delta
 cfg=json.loads((ROOT/'config.json').read_text());os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip();torch.set_num_threads(8);set_seed(cfg['seed'])
 manifest=json.loads((ROOT/'manifest.json').read_text())
 for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 status('PREFLIGHT',runtime=check_gpu_runtime());prefix=cfg['base_subfolder']+'/'
 source=Path(snapshot_download(cfg['base_repo'],revision=cfg['base_revision'],allow_patterns=[prefix+p for p in ['*.json','*.jinja','*.safetensors']],max_workers=4))/cfg['base_subfolder']
 receipt=json.loads((ROOT/'base_receipt.json').read_text());assert receipt['revision']==cfg['base_revision']
 for name,v in receipt['files'].items():
  p=source/name;h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  assert p.stat().st_size==v['size'] and h.hexdigest()==v['sha256']
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer');model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda');assert model.num_parameters()==3039823232 and all(p.requires_grad for p in model.parameters())
 model.config.use_cache=False;model.config.num_logits_to_keep=0
 train=load_rows('train');dev=json.loads((ROOT/'dev_cases.json').read_text());control_cases=json.loads((ROOT/'retention_cases.json').read_text());prior=json.loads((ROOT/'eval_language_v2.json').read_text())
 baseline=evaluate(model,tok,dev,'baseline_dev',cfg);base_controls=controls(model,tok,control_cases,'baseline',cfg)
 working=list(model.named_parameters());masters=make_master_parameters(working);params=[p for _,p in masters]
 opt=bnb.optim.AdamW8bit(params,lr=cfg['learning_rate'],weight_decay=.01)
 # Probe the actual longest sample and allocate optimizer states with LR=0.
 longest=max(train,key=lambda r:len(r['input_ids']));batch={k:torch.tensor([longest[k]],device='cuda') for k in ['input_ids','labels']};batch['attention_mask']=torch.ones_like(batch['input_ids'])
 torch.cuda.reset_peak_memory_stats();model.train();opt.param_groups[0]['lr']=0.
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
 assert torch.isfinite(loss);loss.backward();copy_gradients_to_master(working,masters);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
 peak=torch.cuda.max_memory_allocated()/1024**3;assert peak<.9*torch.cuda.get_device_properties(0).total_memory/1024**3
 write('memory_preflight.json',{'passed':True,'longest_tokens':len(longest['input_ids']),'peak_gib':peak,'learning_rate_zero':True,'master_dtype':'float32','working_dtype':'bfloat16'})
 del opt;model.zero_grad(set_to_none=True)
 for _,p in masters:p.grad=None
 gc.collect();torch.cuda.empty_cache();set_seed(cfg['seed']);opt=bnb.optim.AdamW8bit(params,lr=cfg['learning_rate'],weight_decay=.01)
 rng=random.Random(cfg['seed']);order=list(range(len(train)));rng.shuffle(order);cursor=0;total=math.ceil(len(train)*cfg['epochs']/cfg['gradient_accumulation']);step=0;best_score=(-1,-999);best_step=0;gate=None;stopped=False;started=time.time();losses=[];first_samples=sampled_parameters(model,masters)
 provenance={'base_repo':cfg['base_repo'],'base_revision':cfg['base_revision'],'output_repo':cfg['output_repo'],'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),'method':'Full-weight BF16 working model with FP32 optimizer masters; AdamW8bit; assistant-only loss','expected_steps':total}
 for step in range(1,total+1):
  if (ROOT/'STOP').exists():stopped=True;step-=1;break
  model.train();model.config.num_logits_to_keep=0;opt.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);micro=[]
  for _ in range(cfg['gradient_accumulation']):
   row=train[order[cursor]];cursor+=1
   if cursor==len(order):cursor=0;rng.shuffle(order)
   batch={k:torch.tensor([row[k]],device='cuda') for k in ['input_ids','labels']};batch['attention_mask']=torch.ones_like(batch['input_ids'])
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
   if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
   (loss/cfg['gradient_accumulation']).backward();micro.append(float(loss.detach()))
  copy_gradients_to_master(working,masters);norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
  lr=cfg['learning_rate']*min(1.,step/10)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-10)/max(total-10,1))))
  opt.param_groups[0]['lr']=lr;opt.step();copy_master_to_working(working,masters);losses.append(sum(micro)/len(micro))
  if step==1:write('first_parameter_updates.json',delta(first_samples));del first_samples
  if step%10==0:
   with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,'loss':losses[-1],'learning_rate':lr,'grad_norm':float(norm)})+'\n')
   status('TRAINING',step=step,max_steps=total,loss=losses[-1],learning_rate=lr,grad_norm=float(norm),peak_gib=torch.cuda.max_memory_allocated()/1024**3,elapsed_seconds=time.time()-started)
  if step==100 or step%cfg['eval_steps']==0 or step==total:
   new=evaluate(model,tok,dev,f'dev_{step}',cfg);new_controls=controls(model,tok,control_cases,f'control_{step}',cfg)
   skills=all(new_controls[k]['n']==64 and new_controls[k]['correct']>=base_controls[k]['correct']-4 for k in ['math','reading']);score=(new['correct'],-new['truncated'])
   if skills and new['n']==128 and score>best_score:save_model(model,tok,'best',{**provenance,'training_steps':step},upload=False);best_score=score;best_step=step
   if step==100:
    status('PILOT_EXPORT',step=step);save_model(model,tok,'pilot_100',{**provenance,'training_steps':step})
    language=language_scores(model,tok,'eval_language_pilot');gate=pilot_gate(baseline,new,base_controls,new_controls,language,prior,cfg);write('pilot_gate.json',gate);status('PILOT_GATE',step=step,**gate)
    if not gate['passed']:break
   if (ROOT/'STOP').exists():stopped=True;break
   if step>=300 and step%300==0:status('PERIODIC_EXPORT',step=step);save_model(model,tok,'latest',{**provenance,'training_steps':step})
 provenance.update(training_steps=step,best_step=best_step,pilot_gate=gate,stopped=stopped,training_complete=step==total,losses_first10_mean=sum(losses[:10])/max(1,len(losses[:10])),losses_last10_mean=sum(losses[-10:])/max(1,len(losses[-10:])))
 write('training_provenance.json',provenance)
 if stopped:save_model(model,tok,'stopped',provenance)
 if not best_step and not (ROOT/'pilot_100').exists():save_model(model,tok,'best',provenance,upload=False);best_step=step
 del opt,masters,params,working;model.zero_grad(set_to_none=True);del model;gc.collect();torch.cuda.empty_cache()
 if best_step:
  model=AutoModelForCausalLM.from_pretrained(ROOT/'best',torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
 else:model=AutoModelForCausalLM.from_pretrained(ROOT/'pilot_100' if (ROOT/'pilot_100').exists() else source,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
 final_receipt=save_model(model,tok,'final',provenance);write('final_receipt.json',final_receipt)
 final_lang=language_scores(model,tok,'eval_language_v4') if not stopped else {'complete':False}
 if gate and gate['passed'] and not stopped and not retention_ok(prior,final_lang,cfg) and best_step!=100:
  status('RETENTION_FALLBACK',from_step=best_step,to_step=100)
  del model;gc.collect();torch.cuda.empty_cache()
  model=AutoModelForCausalLM.from_pretrained(ROOT/'pilot_100',torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
  provenance.update(development_best_step=best_step,best_step=100,fallback_reason='Later development-selected candidate failed heldout language retention; pilot100 already passed')
  best_step=100;final_receipt=save_model(model,tok,'final',provenance);write('final_receipt.json',final_receipt)
  final_lang=json.loads((ROOT/'eval_language_pilot.json').read_text());write('eval_language_v4.json',{**final_lang,'reused_for_identical_pilot100_weights':True})
 del model;gc.collect();torch.cuda.empty_cache();report={}
 if gate and gate['passed'] and not stopped:
  status('FINAL_BENCHMARK',selected_step=best_step)
  import final_eval
  report=final_eval.run(ROOT/'final',cfg);gc.collect();torch.cuda.empty_cache()
 expected={'turkish_mmlu':900,'belebele_tr':900,'gsm8k_tr':200,'gsm8k_tr_cot8':200,'ifeval_tr_subset':120}
 complete=all(report.get(k,{}).get('n')==n for k,n in expected.items());limits=cfg['acceptance'];floors={'turkish_mmlu':'mmlu_min_correct','belebele_tr':'belebele_min_correct','gsm8k_tr':'gsm8k_min_correct','gsm8k_tr_cot8':'gsm8k_cot8_min_correct','ifeval_tr_subset':'ifeval_min_correct'}
 gates={'pilot':bool(gate and gate['passed']),'training_complete':provenance['training_complete'] and not stopped,'full_benchmark':complete,'skill_floors':complete and all(report[k]['correct']>=limits[v] for k,v in floors.items()),'stopping':complete and report['ifeval_tr_subset']['truncated']<=limits['ifeval_max_truncated'],'language_retention':retention_ok(prior,final_lang,cfg)}
 decision={'accepted':all(gates.values()),'gates':gates,'benchmark':report,'selected_step':best_step,'production_model_replaced':False,'provenance':provenance};write('acceptance.json',decision)
 card='# Turkish Jamba SFT v4 candidate\n\nPrivate candidate; final weights at `checkpoints/final`. Production v2 remains unchanged.\n\n'+json.dumps(decision,ensure_ascii=False,indent=2)+'\n\nProtocol: https://github.com/serda-dev/jamba-sft-turkish/blob/main/sft_v4/PLAN_TR.md\n'
 HfApi().upload_file(repo_id=cfg['output_repo'],path_in_repo='README.md',path_or_fileobj=card.encode());status('DONE',verified=True,accepted=decision['accepted'],gates=gates,benchmark=report)
if __name__=='__main__':
 try:main()
 except BaseException as error:status('FAILED',error=type(error).__name__);raise
