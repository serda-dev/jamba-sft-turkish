"""Full-weight Jamba SFT using assistant-only targets and verified exports."""
import gzip
import json
import math
import os
import time
from pathlib import Path

ROOT=Path('/mnt/home_extra/sft')


def status(stage,**extra):
    path=ROOT/'status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'stage':stage,'updated_at':time.time(),**extra},indent=2));tmp.replace(path)
    print(stage,json.dumps(extra),flush=True)


def main():
    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download,HfApi
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,set_seed
    from src.train.hub import upload_checkpoint
    from scripts.cpt_gpu_preflight import check_gpu_runtime
    os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip()
    cfg=json.loads((ROOT/'config.json').read_text());set_seed(cfg['seed'])
    import hashlib
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    status('PREFLIGHT',result=check_gpu_runtime())
    cache=snapshot_download(cfg['base_repo'],revision=cfg['base_revision'],
        allow_patterns=[cfg['base_subfolder']+'/'+p for p in ['*.json','*.jinja','*.safetensors']],max_workers=4)
    base=Path(cache)/cfg['base_subfolder']
    tok=AutoTokenizer.from_pretrained(base);tok.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model.config.use_cache=False;model.config.num_logits_to_keep=0
    model.generation_config.eos_token_id=list(set([tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')]))
    model.generation_config.pad_token_id=tok.pad_token_id
    assert model.num_parameters()==3039823232
    def read(name):
        with gzip.open(ROOT/f'{name}.jsonl.gz','rt') as handle:rows=[json.loads(line) for line in handle]
        return Dataset.from_list([{'input_ids':r['input_ids'],'labels':r['labels']} for r in rows])
    train=read('train');validation=read('validation')
    def collate(rows):
        length=math.ceil(max(len(r['input_ids']) for r in rows)/8)*8
        return {'input_ids':torch.tensor([r['input_ids']+[tok.pad_token_id]*(length-len(r['input_ids'])) for r in rows]),
            'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(length-len(r['input_ids'])) for r in rows]),
            'labels':torch.tensor([r['labels']+[-100]*(length-len(r['labels'])) for r in rows])}
    # Independent templates use distinct numeric ranges; no test targets enter training.
    cases=json.loads((ROOT/'eval_cases.json').read_text())
    def diagnose(label):
        model.eval();rows=[]
        with torch.inference_mode():
            for messages in cases:
                prompt=tok.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
                batch=tok(prompt,add_special_tokens=False,return_tensors='pt').to('cuda')
                out=model.generate(**batch,max_new_tokens=128,do_sample=False,use_cache=True,
                    eos_token_id=model.generation_config.eos_token_id,pad_token_id=tok.pad_token_id)
                reply=tok.decode(out[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                expected=messages[-1]['content'];correct=reply==expected
                try:correct=json.loads(reply)==json.loads(expected)
                except ValueError:pass
                rows.append({'messages':messages,'response':reply,'correct':correct})
        report={'label':label,'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows}
        (ROOT/f'eval_{label}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print('DIAGNOSTIC',label,report['correct'],report['total'],flush=True)
        return report
    baseline=diagnose('cpt_baseline')
    class Monitor(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if not logs:return
            with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':state.global_step,'time':time.time(),**logs})+'\n')
            status('TRAINING',step=state.global_step,max_steps=state.max_steps,epoch=state.epoch,metrics=logs)
            if any(not math.isfinite(float(v)) for k,v in logs.items() if k in ['loss','eval_loss']):
                raise ValueError('Nonfinite SFT loss')
        def on_step_end(self,args,state,control,**kwargs):
            if (ROOT/'STOP').exists():control.should_training_stop=True;control.should_save=True
            return control
        def on_save(self,args,state,control,**kwargs):
            checkpoint=ROOT/'checkpoints'/f'checkpoint-{state.global_step}'
            receipt=upload_checkpoint(checkpoint,cfg['output_repo'])
            (ROOT/'checkpoint_receipt.json').write_text(json.dumps(receipt,indent=2))
    args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),num_train_epochs=cfg['epochs'],
        per_device_train_batch_size=cfg['micro_batch'],gradient_accumulation_steps=cfg['gradient_accumulation'],
        per_device_eval_batch_size=2,learning_rate=cfg['learning_rate'],weight_decay=.01,warmup_ratio=.03,
        lr_scheduler_type='cosine',optim='adamw_bnb_8bit',bf16=True,gradient_checkpointing=False,
        logging_steps=10,logging_nan_inf_filter=False,eval_strategy='steps',eval_steps=cfg['eval_save_steps'],save_strategy='steps',
        save_steps=cfg['eval_save_steps'],save_total_limit=2,load_best_model_at_end=True,
        metric_for_best_model='eval_loss',greater_is_better=False,group_by_length=True,
        prediction_loss_only=True,report_to='none',seed=cfg['seed'],dataloader_num_workers=2)
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=validation,data_collator=collate,
                    processing_class=tok,callbacks=[Monitor()])
    trainer.model_accepts_loss_kwargs=False
    baseline_loss=trainer.evaluate()['eval_loss']
    (ROOT/'validation_baseline.json').write_text(json.dumps({'eval_loss':baseline_loss}))
    status('TRAINING',baseline_loss=baseline_loss,samples=len(train))
    trainer.train()
    final_loss=trainer.evaluate()['eval_loss'];diagnostic=diagnose('sft_final')
    final=ROOT/'final';trainer.save_model(str(final));tok.save_pretrained(final)
    receipt=upload_checkpoint(final,cfg['output_repo']);(ROOT/'final_receipt.json').write_text(json.dumps(receipt,indent=2))
    accepted=final_loss<baseline_loss and diagnostic['correct']>=baseline['correct']
    status('FINAL_LANGUAGE_EVALUATION',step=trainer.state.global_step)
    import gc
    import subprocess
    del trainer,model
    gc.collect();torch.cuda.empty_cache()
    with (ROOT/'language-eval.log').open('w') as log:
        subprocess.run(['python','-u','-m','scripts.cpt_evaluate','--model',str(final),
            '--cache',str(ROOT/'cpt_validation/manifest.json'),'--output',str(ROOT/'eval_language_sft_final.json')],
            cwd='/workspace/mamba-cpt-tr',stdout=log,stderr=subprocess.STDOUT,check=True)
    old=json.loads((ROOT/'eval_language_cpt_baseline.json').read_text())
    new=json.loads((ROOT/'eval_language_sft_final.json').read_text())
    language_checks={name:new['scores'][name]['mean_nll']<=old['scores'][name]['mean_nll']*tolerance
                     for name,tolerance in [('turkish',1.05),('english',1.10)]}
    accepted=accepted and all(language_checks.values())
    status('STOPPED_EXPORTED' if (ROOT/'STOP').exists() else 'DONE',verified=receipt['verified'],
           accepted=accepted,baseline_loss=baseline_loss,final_loss=final_loss,
           diagnostic_before=baseline['correct'],diagnostic_after=diagnostic['correct'],
           language_checks=language_checks)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('FAILED',error=type(error).__name__);raise
