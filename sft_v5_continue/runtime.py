"""Broad single-epoch full SFT with FP32 masters and natural-math retention."""
import gc,gzip,hashlib,json,math,os,random,sys,time
from pathlib import Path
try:import v4_runtime as common
except ModuleNotFoundError:from sft_v4 import runtime as common
ROOT=common.ROOT
write=common.write;status=common.status;save_model=common.save_model

def load_rows(name):
 with gzip.open(ROOT/(name+'.jsonl.gz'),'rt') as f:return [json.loads(l) for l in f]

def family_losses(model,rows,tag):
 import torch
 model.eval();model.config.num_logits_to_keep=0;totals={}
 with torch.inference_mode():
  for row in rows:
   if (ROOT/'STOP').exists():break
   batch={k:torch.tensor([row[k]],device='cuda') for k in ['input_ids','labels']};batch['attention_mask']=torch.ones_like(batch['input_ids'])
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
   n=sum(v!=-100 for v in row['labels'][1:]);assert torch.isfinite(loss)
   k=row['family'];d=totals.setdefault(k,{'loss_sum':0.,'tokens':0,'n':0});d['loss_sum']+=float(loss)*n;d['tokens']+=n;d['n']+=1
  else:
   out={'complete':True,'scores':{k:{'mean_loss':v['loss_sum']/v['tokens'],'tokens':v['tokens'],'n':v['n']} for k,v in totals.items()}};write(tag+'.json',out);return out
 out={'complete':False,'scores':{}};write(tag+'.json',out);return out

def loss_retained(prior,new,ratio,absolute_allowance=0.):
 return new['complete'] and set(prior['scores'])==set(new['scores']) and all(new['scores'][k]['tokens']==v['tokens'] and new['scores'][k]['mean_loss']<=max(v['mean_loss']*ratio,v['mean_loss']+absolute_allowance) for k,v in prior['scores'].items())

def skill_gates(bdev,dev,bmath,math_report,bloss,loss,cfg):
 p=cfg['pilot_policy']
 return {'complete':dev['n']==bdev['n']==128 and math_report['n']==bmath['n']==143 and loss['complete'],'natural_math_retained':math_report['correct']>=bmath['correct']-p['natural_math_max_drop'],'stopping':dev['truncated']<=bdev['truncated']+p['dev_max_extra_truncation'],'family_losses_retained':loss_retained(bloss,loss,p['family_loss_max_ratio'],p.get('family_loss_absolute_allowance',0.))}

def lr_for(step,total,cfg):
 warmup=cfg['warmup_steps'];restart_ramp=min(1.,max(0,step-cfg.get('resume_step',0))/cfg.get('resume_warmup_steps',1));return restart_ramp*cfg['learning_rate']*min(1.,step/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(total-warmup,1))))

def main():
 import torch,bitsandbytes as bnb
 from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
 from huggingface_hub import snapshot_download,HfApi
 from scripts.cpt_gpu_preflight import check_gpu_runtime
 from precision import make_master_parameters,copy_gradients_to_master,copy_master_to_working,sampled_parameters,delta
 cfg=json.loads((ROOT/'config.json').read_text());os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip();torch.set_num_threads(8);set_seed(cfg['seed'])
 assert cfg['epochs']==1 and cfg['micro_batch']==1
 manifest=json.loads((ROOT/'manifest.json').read_text())
 for name,want in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want
 status('PREFLIGHT',runtime=check_gpu_runtime());prefix=cfg['base_subfolder']+'/'
 source=Path(snapshot_download(cfg['base_repo'],revision=cfg['base_revision'],allow_patterns=[prefix+p for p in ['*.json','*.jinja','*.safetensors']],max_workers=4))/cfg['base_subfolder']
 receipt=json.loads((ROOT/'base_receipt.json').read_text());assert receipt['revision']==cfg['base_revision']
 for name,want in receipt['files'].items():
  path=source/name;h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  assert path.stat().st_size==want['size'] and h.hexdigest()==want['sha256']
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
 def reload(path):
  m=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda');m.config.use_cache=False;m.config.num_logits_to_keep=0;return m
 resume_receipt=json.loads((ROOT/'resume_receipt.json').read_text());assert resume_receipt['revision']==cfg['resume_revision'];resume_prefix=resume_receipt['path']+'/'
 resume_source=Path(snapshot_download(resume_receipt['repo_id'],revision=resume_receipt['revision'],allow_patterns=[resume_prefix+p for p in ['*.json','*.jinja','*.safetensors']],max_workers=4))/resume_receipt['path']
 for name,want in resume_receipt['files'].items():
  path=resume_source/name;h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  assert path.stat().st_size==want['size'] and h.hexdigest()==want['sha256']
 model=reload(resume_source);assert model.num_parameters()==3039823232 and all(p.requires_grad for p in model.parameters())
 train=load_rows('train');validation=load_rows('validation');dev=json.loads((ROOT/'dev_cases.json').read_text());math_cases=json.loads((ROOT/'retention_cases.json').read_text())
 # Use the entire natural-math dev set and a fixed per-family sample for loss retention.
 sampled=[];rng=random.Random(cfg['seed']+1)
 for family in sorted({r['family'] for r in validation}):
  group=[r for r in validation if r['family']==family];rng.shuffle(group);sampled+=group[:64]
 write('validation_sample_identity.json',{'prompt_hashes':[r['prompt_hash'] for r in sampled],'n':len(sampled)})
 for name,want in cfg['baseline_report_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want
 baseline=json.loads((ROOT/'baseline_dev.json').read_text());bmath=json.loads((ROOT/'baseline_natural_math.json').read_text());bloss=json.loads((ROOT/'baseline_family_loss.json').read_text())
 assert baseline['n']==128 and bmath['n']==143 and bloss['complete']
 prior=json.loads((ROOT/'eval_language_current_v2.json').read_text());historic=json.loads((ROOT/'eval_language_v2.json').read_text())
 assert prior['complete'] and prior['heldout_manifest_sha256']==historic['heldout_manifest_sha256']
 assert all(prior['scores'][k]['tokens']==historic['scores'][k]['tokens'] and abs(prior['scores'][k]['mean_nll']/historic['scores'][k]['mean_nll']-1)<.005 for k in prior['scores'])
 working=list(model.named_parameters());masters=make_master_parameters(working);params=[p for _,p in masters];opt=bnb.optim.AdamW8bit(params,lr=0.,weight_decay=.01)
 longest=max(train,key=lambda r:len(r['input_ids']));batch={k:torch.tensor([longest[k]],device='cuda') for k in ['input_ids','labels']};batch['attention_mask']=torch.ones_like(batch['input_ids']);model.train();torch.cuda.reset_peak_memory_stats()
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
 assert torch.isfinite(loss);loss.backward();copy_gradients_to_master(working,masters);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();peak=torch.cuda.max_memory_allocated()/1024**3
 assert peak<.92*torch.cuda.get_device_properties(0).total_memory/1024**3
 write('memory_preflight.json',{'passed':True,'longest_tokens':len(longest['input_ids']),'peak_gib':peak,'lr_zero':True,'working':'BF16','masters':'FP32'})
 del opt,loss,batch;model.zero_grad(set_to_none=True)
 for _,p in masters:p.grad=None
 gc.collect();torch.cuda.empty_cache();set_seed(cfg['seed']);opt=bnb.optim.AdamW8bit(params,lr=cfg['learning_rate'],weight_decay=.01)
 rng=random.Random(cfg['seed']);order=list(range(len(train)));rng.shuffle(order);total=math.ceil(len(order)/cfg['gradient_accumulation']);started=time.time();start_step=cfg['resume_step'];assert start_step==100;gate=None;best_score=None;best_step=100;step=start_step;stopped=False;losses=[];samples=sampled_parameters(model,masters)
 provenance={'base_repo':cfg['base_repo'],'base_revision':cfg['base_revision'],'output_repo':cfg['output_repo'],'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),'method':'Full BF16 weights + FP32 optimizer masters; component-weighted assistant-token accumulation; AdamW8bit','objective_shares':cfg['objective_shares'],'epochs':1,'learning_rate':cfg['learning_rate'],'gradient_accumulation':cfg['gradient_accumulation'],'micro_batch':1,'expected_steps':total,'seed':cfg['seed'],'warmup_steps':cfg['warmup_steps'],'initialized_from_repo':resume_receipt['repo_id'],'initialized_from_revision':resume_receipt['revision'],'prior_completed_steps':start_step,'exact_optimizer_resume':False,'optimizer_and_fp32_masters_reinitialized':True,'resume_warmup_steps':cfg['resume_warmup_steps']}
 original_gate=json.loads((ROOT/'original_pilot_gate.json').read_text());old_dev=json.loads((ROOT/'dev_100.json').read_text());old_math=json.loads((ROOT/'natural_math_100.json').read_text());old_loss=json.loads((ROOT/'family_loss_100.json').read_text());old_lang=json.loads((ROOT/'eval_language_pilot.json').read_text())
 checks=skill_gates(baseline,old_dev,bmath,old_math,bloss,old_loss,cfg);checks['language_retention']=common.retention_ok(prior,old_lang,cfg);assert all(checks.values()),checks
 gate={'passed':True,'scope':'user-authorized continuation policy; not original pilot approval','original_pilot_passed':original_gate['passed'],'original_pilot_gate':original_gate,'continuation_checks':checks,'early_instruction_gain_requirement_removed':True,'family_loss_absolute_allowance':cfg['pilot_policy']['family_loss_absolute_allowance']}
 write('pilot_gate.json',gate);status('CONTINUATION_AUTHORIZED',resume_step=start_step,exact_optimizer_resume=False,continuation_checks=checks)
 best_score=(old_dev['correct'],-old_dev['truncated'],-sum(v['mean_loss'] for v in old_loss['scores'].values()));save_model(model,tok,'best',{**provenance,'training_steps':start_step},upload=False)
 (ROOT/'pilot_100').symlink_to(resume_source,target_is_directory=True)
 for step in range(start_step+1,total+1):
  if (ROOT/'STOP').exists():stopped=True;step-=1;break
  indexes=order[(step-1)*cfg['gradient_accumulation']:step*cfg['gradient_accumulation']];n_targets=sum(train[i]['n_target_tokens']*train[i]['objective_weight'] for i in indexes);model.train();model.config.num_logits_to_keep=0;opt.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);sumloss=0.
  for i in indexes:
   row=train[i];batch={k:torch.tensor([row[k]],device='cuda') for k in ['input_ids','labels']};batch['attention_mask']=torch.ones_like(batch['input_ids'])
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
   assert torch.isfinite(loss);weight=row['n_target_tokens']*row['objective_weight']/n_targets;(loss*weight).backward();sumloss+=float(loss.detach())*weight
  copy_gradients_to_master(working,masters);norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);lr=lr_for(step,total,cfg);opt.param_groups[0]['lr']=lr;opt.step();copy_master_to_working(working,masters);losses.append(sumloss)
  if step==start_step+1:write('first_parameter_updates.json',delta(samples));del samples
  if step%10==0:
   elapsed=time.time()-started;metric={'step':step,'max_steps':total,'loss':sumloss,'learning_rate':lr,'grad_norm':float(norm),'elapsed_seconds':elapsed,'estimated_remaining_seconds':elapsed/(step-start_step)*(total-step),'executed_steps_this_attempt':step-start_step,'peak_gib':torch.cuda.max_memory_allocated()/1024**3}
   with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
   status('TRAINING',**metric)
  if step%cfg['eval_steps']==0 or step==total:
   new=common.evaluate(model,tok,dev,f'dev_{step}',cfg);new_math=common.evaluate(model,tok,math_cases,f'natural_math_{step}',cfg);new_loss=family_losses(model,sampled,f'family_loss_{step}');checks=skill_gates(baseline,new,bmath,new_math,bloss,new_loss,cfg)
   if all(checks.values()):
    score=(new['correct'],-new['truncated'],-sum(v['mean_loss'] for v in new_loss['scores'].values()))
    if best_score is None or score>best_score:save_model(model,tok,'best',{**provenance,'training_steps':step},upload=False);best_score=score;best_step=step
   if (ROOT/'STOP').exists():stopped=True;break
   if step%cfg['eval_steps']==0:save_model(model,tok,'latest',{**provenance,'training_steps':step})
 stopped=stopped or (ROOT/'STOP').exists();complete=step==total and not stopped;provenance.update(training_steps=step,optimizer_steps_this_attempt=step-start_step,training_complete=complete,best_step=best_step,pilot_gate=gate,stopped=stopped,loss_first10=sum(losses[:10])/max(1,len(losses[:10])),loss_last10=sum(losses[-10:])/max(1,len(losses[-10:])))
 write('training_provenance.json',provenance)
 if stopped:save_model(model,tok,'stopped',provenance)
 if not best_step and not (ROOT/'pilot_100').exists():save_model(model,tok,'best',{**provenance,'training_steps':step},upload=False);best_step=step
 del opt,params,masters,working;
 if 'loss' in locals():del loss
 if 'batch' in locals():del batch
 model.zero_grad(set_to_none=True);del model;gc.collect();torch.cuda.empty_cache()
 selected=ROOT/'best' if (ROOT/'best').exists() else ROOT/'pilot_100';assert selected.exists();model=reload(selected);final_lang=common.language_scores(model,tok,'eval_language_selected') if not stopped else {'complete':False}
 if gate and gate['passed'] and not stopped and best_step!=100 and not common.retention_ok(prior,final_lang,cfg):
  write('eval_language_rejected_development_best.json',{'selected_step':best_step,**final_lang});del model;gc.collect();torch.cuda.empty_cache();model=reload(ROOT/'pilot_100');best_step=100;provenance['fallback_reason']='Development best failed language retention';final_lang=json.loads((ROOT/'eval_language_pilot.json').read_text())
 provenance['selected_step']=best_step or cfg['pilot_steps'];write('selected_model_provenance.json',provenance);write('eval_language_v5.json',final_lang);receipt=save_model(model,tok,'final',provenance);write('final_receipt.json',receipt)
 del model;gc.collect();torch.cuda.empty_cache();report={};sealed={}
 if complete and gate and gate['passed'] and common.retention_ok(prior,final_lang,cfg):
  # Sealed test is opened only after weight selection; both V2 and candidate run identically.
  sealed_cases=json.loads((ROOT/'sealed_math_cases.json').read_text());model=reload(source);ref=common.evaluate(model,tok,sealed_cases,'sealed_math_v2',cfg);del model;gc.collect();torch.cuda.empty_cache();model=reload(ROOT/'final');candidate=common.evaluate(model,tok,sealed_cases,'sealed_math_v5',cfg);del model;gc.collect();torch.cuda.empty_cache();sealed={'v2_correct':ref['correct'],'v5_correct':candidate['correct'],'n':128}
  import final_eval
  report=final_eval.run(ROOT/'final',cfg)
 limits=cfg['acceptance'];expected={'turkish_mmlu':900,'belebele_tr':900,'gsm8k_tr':200,'gsm8k_tr_cot8':200,'ifeval_tr_subset':120};full=all(report.get(k,{}).get('n')==n for k,n in expected.items());floors={'turkish_mmlu':'mmlu_min_correct','belebele_tr':'belebele_min_correct','gsm8k_tr':'gsm8k_min_correct','gsm8k_tr_cot8':'gsm8k_cot8_min_correct','ifeval_tr_subset':'ifeval_min_correct'}
 gates={'continuation_allowed':bool(gate and gate['passed']),'training_complete':complete,'full_regression_benchmark':full,'skill_floors':full and all(report[k]['correct']>=limits[v] for k,v in floors.items()),'stopping':full and report['ifeval_tr_subset']['truncated']<=limits['ifeval_max_truncated'],'language_retention':common.retention_ok(prior,final_lang,cfg),'sealed_math_retention':bool(sealed and sealed['v5_correct']>=sealed['v2_correct']-4)}
 decision={'accepted':all(gates.values()),'gates':gates,'benchmark':report,'sealed_math':sealed,'provenance':provenance,'production_model_replaced':False};write('acceptance.json',decision)
 HfApi().upload_file(repo_id=cfg['output_repo'],path_in_repo='README.md',path_or_fileobj=('# Turkish SFT v5 private candidate\n\nProduction V2 unchanged. Final weights: checkpoints/final.\n\n```json\n'+json.dumps(decision,indent=2)+'\n```\n').encode());status('DONE',accepted=decision['accepted'],verified=True,gates=gates,benchmark=report)

if __name__=='__main__':
 try:main()
 except BaseException as error:status('FAILED',error=type(error).__name__);raise
