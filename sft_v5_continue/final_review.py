"""Finalize completed training with direct conversation checks and all original tests."""
import gc,hashlib,json,os,time
from pathlib import Path
import v4_runtime as common
from conversation_check import evaluate as conversation_eval,baseline_source
from review_policy import eligible
from runtime import family_losses,load_rows
ROOT=common.ROOT

def load(name):return json.loads((ROOT/name).read_text())
def main():
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip();torch.set_num_threads(8);cfg=load('config.json');common.status('FINAL_REVIEW_STARTING')
 pause=load('final_review_pause.json');assert pause['optimizer_training_complete']
 raw=load('training_provenance.json');expected=raw['expected_steps'];assert raw['training_steps']==expected==1997
 stopped_provenance=json.loads((ROOT/'stopped/sft_provenance.json').read_text());assert stopped_provenance['training_steps']==1997
 # Preserve the original runner's stop/receipt; the pause happened after all updates.
 for name in ['acceptance.json','final_receipt.json']:
  if (ROOT/name).exists():common.write('original_runner_'+name,load(name))
 stop=ROOT/'STOP';assert stop.exists() and stop.stat().st_mtime<=pause['requested_at']+5,'A newer stop request must not be overridden';stop.unlink()
 policy=load('final_review_policy.json');assert policy['conversation_measure']=='generated_exact_json' and policy['frozen_test_feedback_used'] is False
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer');validation=load_rows('validation');identity=set(load('validation_sample_identity.json')['prompt_hashes']);sampled=[r for r in validation if r['prompt_hash'] in identity];assert len(sampled)==len(identity)
 baseline_dev=load('baseline_dev.json');baseline_math=load('baseline_natural_math.json');baseline_loss=load('baseline_family_loss.json');prior=load('eval_language_current_v2.json');ref_conversation=load('conversation_generated_v2.json');assert ref_conversation['n']==15
 dev_cases=load('dev_cases.json');math_cases=load('retention_cases.json')
 def model_at(path):
  m=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();m.config.use_cache=False;m.config.num_logits_to_keep=0;return m
 candidates=[]
 for step,path in [(1997,ROOT/'stopped'),(1600,ROOT/'conversation_audit_snapshot'),(100,ROOT/'pilot_100')]:
  saved=json.loads((path/'sft_provenance.json').read_text());assert saved['training_steps']==step
  dev=load(f'dev_{step}.json')
  if step==1997:
   model=model_at(path);math_report=common.evaluate(model,tok,math_cases,'review_math_1997',cfg);loss=family_losses(model,sampled,'review_family_loss_1997');del model;gc.collect();torch.cuda.empty_cache()
  else:math_report=load(f'natural_math_{step}.json');loss=load(f'family_loss_{step}.json')
  tag=f'conversation_generated_{step}';conversation=load(tag+'.json') if (ROOT/(tag+'.json')).exists() else conversation_eval(path,tag)
  gates=eligible(baseline_dev,dev,baseline_math,math_report,baseline_loss,loss,ref_conversation,conversation,cfg)
  item={'step':step,'path':str(path),'dev_correct':dev['correct'],'dev_truncated':dev['truncated'],'math_correct':math_report['correct'],'conversation_correct':conversation['correct'],'gates':gates,'eligible':all(gates.values()),'nonconversation_loss_sum':sum(v['mean_loss'] for k,v in loss['scores'].items() if k!='conversation')};candidates.append(item);common.write('review_candidates.json',candidates)
 valid=sorted([c for c in candidates if c['eligible']],key=lambda c:(c['dev_correct'],-c['dev_truncated'],-c['nonconversation_loss_sum']),reverse=True);assert valid,'No non-regressing development candidate'
 selected=None;model=None;language=None
 for candidate in valid:
  model=model_at(candidate['path']);language=common.language_scores(model,tok,f"review_language_{candidate['step']}")
  if common.retention_ok(prior,language,cfg):selected=candidate;break
  del model;gc.collect();torch.cuda.empty_cache()
 assert selected is not None,'No candidate passed heldout language retention'
 provenance={**raw,'training_complete':True,'stopped':False,'optimizer_updates_completed':1997,'original_runner_stopped_for_final_review':raw['stopped'],'original_runner_best_step':raw['best_step'],'best_step':selected['step'],'optimizer_training_interrupted':False,'selected_step':selected['step'],'selection_policy':policy,'development_selection':selected,'full_optimization_verified_by':'original training_steps==1997, stopped model provenance==1997, and completed dev_1997'}
 common.write('review_training_provenance.json',provenance);receipt=common.save_model(model,tok,'final',provenance);common.write('final_receipt.json',receipt);common.write('eval_language_final_review.json',language);del model;gc.collect();torch.cuda.empty_cache()
 # Open the sealed set only after final weight selection. Do not select using these results.
 sealed_cases=load('sealed_math_cases.json');model=model_at(baseline_source(cfg));ref=common.evaluate(model,tok,sealed_cases,'sealed_math_v2',cfg);del model;gc.collect();torch.cuda.empty_cache();model=model_at(ROOT/'final');candidate=common.evaluate(model,tok,sealed_cases,'sealed_math_v5',cfg);del model;gc.collect();torch.cuda.empty_cache()
 import final_eval
 report=final_eval.run(ROOT/'final',cfg);limits=cfg['acceptance'];expected_counts={'turkish_mmlu':900,'belebele_tr':900,'gsm8k_tr':200,'gsm8k_tr_cot8':200,'ifeval_tr_subset':120};complete=all(report.get(k,{}).get('n')==n for k,n in expected_counts.items());floors={'turkish_mmlu':'mmlu_min_correct','belebele_tr':'belebele_min_correct','gsm8k_tr':'gsm8k_min_correct','gsm8k_tr_cot8':'gsm8k_cot8_min_correct','ifeval_tr_subset':'ifeval_min_correct'}
 gates={'training_complete':True,'development_nonregression':selected['eligible'],'language_retention':common.retention_ok(prior,language,cfg),'full_regression_benchmark':complete,'skill_floors':complete and all(report[k]['correct']>=limits[v] for k,v in floors.items()),'stopping':complete and report['ifeval_tr_subset']['truncated']<=limits['ifeval_max_truncated'],'sealed_math_retention':ref['n']==candidate['n']==128 and candidate['correct']>=ref['correct']-4}
 decision={'accepted':all(gates.values()),'gates':gates,'benchmark':report,'sealed_math':{'v2_correct':ref['correct'],'v5_correct':candidate['correct'],'n':candidate['n']},'provenance':provenance,'production_model_replaced':False};common.write('acceptance.json',decision)
 from huggingface_hub import HfApi
 card='# Turkish SFT v5 continuation candidate\n\nAll 1997 optimizer updates completed. Final weights at checkpoints/final; selected step '+str(selected['step'])+'. Production V2 unchanged.\n\n```json\n'+json.dumps(decision,indent=2)+'\n```\n';HfApi().upload_file(repo_id=cfg['output_repo'],path_in_repo='README.md',path_or_fileobj=card.encode());common.status('DONE',accepted=decision['accepted'],verified=True,gates=gates,benchmark=report,selected_step=selected['step'],final_review_complete=True)

if __name__=='__main__':
 try:main()
 except BaseException as error:common.status('FAILED',error=type(error).__name__,phase='final_review');raise
