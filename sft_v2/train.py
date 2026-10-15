"""Paired starting-point pilot, full-weight SFT and independent final comparison."""
import os,sys,json,time,gzip,math,gc,re,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path('/mnt/home_extra/sft')
sys.path.insert(0,str(ROOT))

def status(stage,**kw):
 data={'stage':stage,'updated_at':time.time(),**kw};p=ROOT/'status.tmp';p.write_text(json.dumps(data,indent=2));p.replace(ROOT/'status.json');print(stage,json.dumps(kw),flush=True)

def main():
 import torch
 from probe_score import check
 from datasets import Dataset
 from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,set_seed
 from huggingface_hub import snapshot_download
 from src.train.hub import upload_checkpoint
 from scripts.cpt_gpu_preflight import check_gpu_runtime
 os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip();torch.set_num_threads(8)
 cfg=json.loads((ROOT/'config.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text());set_seed(cfg['seed'])
 for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 status('PREFLIGHT',runtime=check_gpu_runtime())
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer');probes=json.loads((ROOT/'probes.json').read_text())
 def read(name,cap=None):
  with gzip.open(ROOT/f'{name}.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
  if cap:rows=sorted(rows,key=lambda r:r['prompt_hash'])[:cap]
  return Dataset.from_list([{'input_ids':r['input_ids'],'labels':r['labels']} for r in rows]).shuffle(seed=cfg['seed'])
 pilot=read('pilot');val=read('validation',640)
 def collate(rows):
  length=math.ceil(max(len(r['input_ids']) for r in rows)/8)*8
  return {'input_ids':torch.tensor([r['input_ids']+[tok.pad_token_id]*(length-len(r['input_ids'])) for r in rows]),'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(length-len(r['input_ids'])) for r in rows]),'labels':torch.tensor([r['labels']+[-100]*(length-len(r['labels'])) for r in rows])}
 def load(path):
  model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
  assert model.num_parameters()==3039823232
  model.config.use_cache=False;model.config.num_logits_to_keep=0
  model.generation_config.eos_token_id=[519];model.generation_config.pad_token_id=tok.pad_token_id
  return model
 def probe(model,name):
  model.eval();tok.padding_side='left';responses=[]
  with torch.inference_mode():
   for offset in range(0,len(probes),8):
    group=probes[offset:offset+8];text=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True) for r in group]
    batch=tok(text,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
    result=model.generate(**batch,max_new_tokens=256,do_sample=False,use_cache=True,eos_token_id=[519],pad_token_id=tok.pad_token_id)
    for r,ids in zip(group,result[:,batch['input_ids'].shape[1]:]):
     reply=tok.decode(ids,skip_special_tokens=True).strip();ok=check(r,reply)
     responses.append({**r,'response':reply,'correct':ok,'truncated':519 not in ids.tolist()})
  tok.padding_side='right';families={f:sum(r['correct'] for r in responses if r['family']==f)/sum(r['family']==f for r in responses) for f in sorted({r['family'] for r in responses})}
  out={'name':name,'macro_accuracy':sum(families.values())/len(families),'family_accuracy':families,'constraint_probe':True,'note':'Procedural diagnostic; prescribed content is exact, open prose uses structural constraints. Not a factual or broad capability benchmark.','cases':responses}
  (ROOT/f'probe_{name}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('PROBE',name,families,flush=True);return out
 def args(directory,pilot_mode=False):
  return TrainingArguments(output_dir=str(directory),num_train_epochs=cfg['epochs'],max_steps=cfg['pilot_steps'] if pilot_mode else -1,per_device_train_batch_size=cfg['micro_batch'],gradient_accumulation_steps=cfg['gradient_accumulation'],per_device_eval_batch_size=2,learning_rate=cfg['learning_rate'],weight_decay=.01,warmup_ratio=.04,lr_scheduler_type='cosine',optim='adamw_bnb_8bit',bf16=True,gradient_checkpointing=False,logging_steps=10,logging_nan_inf_filter=False,eval_strategy='no' if pilot_mode else 'steps',eval_steps=cfg['eval_save_steps'],save_strategy='no' if pilot_mode else 'steps',save_steps=cfg['eval_save_steps'],save_total_limit=2,load_best_model_at_end=not pilot_mode,metric_for_best_model='eval_loss',greater_is_better=False,group_by_length=True,prediction_loss_only=True,report_to='none',seed=cfg['seed'],dataloader_num_workers=2)
 class Monitor(TrainerCallback):
  def __init__(self,name):self.name=name
  def on_log(self,args,state,control,logs=None,**kw):
   if logs:
    with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps({'branch':self.name,'step':state.global_step,'time':time.time(),**logs})+'\n')
    if any(not math.isfinite(float(v)) for k,v in logs.items() if k in ['loss','eval_loss']):raise ValueError('Nonfinite loss')
    status('PILOT_TRAINING' if self.name!='full' else 'TRAINING',branch=self.name,step=state.global_step,max_steps=state.max_steps,epoch=state.epoch,metrics=logs,peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
  def on_step_end(self,args,state,control,**kw):
   if (ROOT/'STOP').exists():control.should_training_stop=True;control.should_save=True
   return control
  def on_save(self,args,state,control,**kw):
   p=Path(args.output_dir)/f'checkpoint-{state.global_step}';r=upload_checkpoint(p,cfg['output_repo']);(ROOT/'checkpoint_receipt.json').write_text(json.dumps(r,indent=2))
 def trainer_for(model,data,directory,name,pilot_mode=False):
  trainer=Trainer(model=model,args=args(directory,pilot_mode),train_dataset=data,eval_dataset=val,data_collator=collate,processing_class=tok,callbacks=[Monitor(name)])
  trainer.model_accepts_loss_kwargs=False;return trainer
 outcomes=[]
 for candidate in cfg['candidates']:
  name=candidate['name'];status('PILOT_LOADING',branch=name)
  folder=candidate['subfolder'];prefix=folder+'/' if folder else ''
  cache=snapshot_download(candidate['repo'],revision=candidate['revision'],allow_patterns=[prefix+p for p in ['*.json','*.jinja','*.safetensors']],max_workers=4)
  model=load(Path(cache)/folder);baseline=probe(model,name+'_before')
  trainer=trainer_for(model,pilot,ROOT/('pilot_train_'+name),name,True);before=trainer.evaluate()['eval_loss'];trainer.train();after=trainer.evaluate()['eval_loss']
  model=trainer.accelerator.unwrap_model(model,keep_fp32_wrapper=False);result=probe(model,name+'_after')
  dest=ROOT/('pilot_'+name);model.save_pretrained(dest,safe_serialization=True);tok.save_pretrained(dest)
  outcome={'candidate':candidate,'loss_before':before,'loss_after':after,'probe_before':baseline['macro_accuracy'],'probe_after':result['macro_accuracy'],'path':str(dest)};outcomes.append(outcome)
  (ROOT/'pilot_results.json').write_text(json.dumps(outcomes,indent=2));del trainer,model;gc.collect();torch.cuda.empty_cache()
  if (ROOT/'STOP').exists():raise RuntimeError('Budget stop before full training')
 # Loss on the same balanced development set is primary; procedural scores guard major regression.
 eligible=[o for o in outcomes if o['loss_after']<o['loss_before'] and o['probe_after']>=o['probe_before']-.05]
 if not eligible:status('PILOT_REJECTED',outcomes=outcomes);return
 winner=min(eligible,key=lambda r:r['loss_after']);(ROOT/'selection.json').write_text(json.dumps({'winner':winner,'criterion':'Lowest same-development assistant loss among pilots with improving loss and <=5pp procedural macro regression; no external benchmark used for selection.'},indent=2))
 status('FULL_LOADING',selected=winner['candidate']['name']);model=load(winner['path']);train=read('train');trainer=trainer_for(model,train,ROOT/'checkpoints','full')
 baseline_loss=trainer.evaluate()['eval_loss'];status('TRAINING',samples=len(train),validation=len(val),selected=winner['candidate']['name'],baseline_loss=baseline_loss)
 trainer.train();final_loss=trainer.evaluate()['eval_loss'];state=trainer.state.to_dict() if hasattr(trainer.state,'to_dict') else vars(trainer.state)
 (ROOT/'trainer_state_final.json').write_text(json.dumps(state,indent=2,default=str))
 model=trainer.accelerator.unwrap_model(model,keep_fp32_wrapper=False);final_probe=probe(model,'final')
 final=ROOT/'final';model.config.use_cache=True;model.save_pretrained(final,safe_serialization=True);tok.save_pretrained(final)
 provenance={'selected_start':winner,'training_steps':trainer.state.global_step,'best_checkpoint':trainer.state.best_model_checkpoint,'assistant_validation_loss':final_loss,'source_manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()};(final/'sft_provenance.json').write_text(json.dumps(provenance,indent=2))
 receipt=upload_checkpoint(final,cfg['output_repo']);(ROOT/'final_receipt.json').write_text(json.dumps(receipt,indent=2))
 status('FINAL_BENCHMARK',verified=receipt['verified'],baseline_loss=baseline_loss,final_loss=final_loss)
 del trainer,model,train;gc.collect();torch.cuda.empty_cache()
 # Frozen public tests are final reporting only, never checkpoint selection.
 import final_eval
 report=final_eval.run(final,cfg)
 gc.collect();torch.cuda.empty_cache()
 import subprocess
 language_ok=False
 if not (ROOT/'STOP').exists():
  with (ROOT/'language-eval.log').open('w') as log:
   subprocess.run(['python','-u','-m','scripts.cpt_evaluate','--model',str(final),'--cache',str(ROOT/'cpt_validation/manifest.json'),'--output',str(ROOT/'eval_language_v2.json')],cwd='/workspace/mamba-cpt-tr',stdout=log,stderr=subprocess.STDOUT,check=True)
  prior=json.loads((ROOT/'eval_language_v1.json').read_text());new=json.loads((ROOT/'eval_language_v2.json').read_text())
  language_ok=all(new['scores'][k]['mean_nll']<=prior['scores'][k]['mean_nll']*ratio for k,ratio in [('turkish',1.05),('english',1.10)])
 expected={'turkish_mmlu':900,'belebele_tr':900,'gsm8k_tr':200,'gsm8k_tr_cot8':200,'ifeval_tr_subset':120}
 complete=all(report.get(k,{}).get('n')==n for k,n in expected.items())
 floors={'turkish_mmlu':206,'belebele_tr':422,'gsm8k_tr':39,'gsm8k_tr_cot8':19,'ifeval_tr_subset':34}
 regression_ok=complete and all(report[k]['correct']>=floor for k,floor in floors.items())
 improved=complete and (report['ifeval_tr_subset']['correct']>=44 or report['gsm8k_tr']['correct']>=55 or report['gsm8k_tr_cot8']['correct']>=35)
 accepted=final_loss<baseline_loss and language_ok and regression_ok and improved
 from huggingface_hub import HfApi
 card='# Jamba2-3B-Turkish SFT v2 candidate\n\nPrivate experimental checkpoint; published v1 is preserved.\n\nFinal weights: `checkpoints/final`.\n\nAcceptance: '+str(accepted)+' (development loss, retention NLL and frozen-task regression gates).\n\n```json\n'+json.dumps({'selected_start':winner['candidate'],'final_benchmark':report,'language_ok':language_ok,'regression_ok':regression_ok,'improved':improved},indent=2)+'\n```\n\nFull protocol: https://github.com/serda-dev/jamba-sft-turkish/tree/main/sft_v2\n'
 HfApi().upload_file(repo_id=cfg['output_repo'],path_in_repo='README.md',path_or_fileobj=card.encode(),commit_message='Report final v2 candidate and frozen evaluation')
 status('STOPPED_EXPORTED' if (ROOT/'STOP').exists() else 'DONE',verified=True,accepted=accepted,final_loss=final_loss,benchmark=report,selected=winner['candidate']['name'])
if __name__=='__main__':
 try:main()
 except BaseException as error:status('FAILED',error=type(error).__name__);raise
