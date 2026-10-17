"""Separate memory preflight, full-weight SFT and resumable final reporting."""
import argparse
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/home_extra/sft')


def write(name,value):
    path=ROOT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def status(stage,**fields):
    row={'stage':stage,'updated_at':time.time(),**fields};write('status.json',row)
    with (ROOT/'events.jsonl').open('a') as out:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(stage,json.dumps(fields,ensure_ascii=False),flush=True)


def acceptance(cfg,report,prior,new,provenance,stopped):
    expected={'turkish_mmlu':900,'belebele_tr':900,'gsm8k_tr':200,'gsm8k_tr_cot8':200,'ifeval_tr_subset':120}
    limits=cfg['acceptance'];complete=all(report.get(k,{}).get('n')==n for k,n in expected.items())
    floors={'turkish_mmlu':'mmlu_min_correct','belebele_tr':'belebele_min_correct','gsm8k_tr':'gsm8k_min_correct','gsm8k_tr_cot8':'gsm8k_cot8_min_correct','ifeval_tr_subset':'ifeval_min_correct'}
    skills=complete and all(report[k]['correct']>=limits[v] for k,v in floors.items())
    stopping=complete and report['ifeval_tr_subset']['truncated']<=limits['ifeval_max_truncated']
    retention=bool(prior and new and prior['heldout_manifest_sha256']==new['heldout_manifest_sha256'])
    if retention:
        retention=all(new['scores'][k]['mean_nll']<=prior['scores'][k]['mean_nll']*limits[k+'_nll_max_ratio_to_v2'] for k in ['turkish','english'])
    training=not stopped and provenance['training_steps']==provenance['expected_steps'] and provenance['final_loss']<provenance['baseline_loss']
    gates={'complete_benchmark':complete,'skills':skills,'stopping':stopping,'retention':retention,'training':training}
    return {'accepted':all(gates.values()),'gates':gates}


def rows(name):
    with gzip.open(ROOT/f'{name}.jsonl.gz','rt') as stream:return [json.loads(l) for l in stream]


def load(path):
    import torch
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    assert model.num_parameters()==3039823232 and all(p.requires_grad for p in model.parameters())
    model.config.use_cache=False;model.config.num_logits_to_keep=0
    return model


def preflight(cfg):
    import torch
    import bitsandbytes as bnb
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from scripts.cpt_gpu_preflight import check_gpu_runtime
    status('PREFLIGHT',runtime=check_gpu_runtime())
    prefix=cfg['base_subfolder']+'/'
    source=Path(snapshot_download(cfg['base_repo'],revision=cfg['base_revision'],allow_patterns=[prefix+p for p in ['*.json','*.jinja','*.safetensors']],max_workers=4))/cfg['base_subfolder']
    receipt=json.loads((ROOT/'base_receipt.json').read_text())
    assert receipt['repo_id']==cfg['base_repo'] and receipt['revision']==cfg['base_revision']
    for name,info in receipt['files'].items():
        p=source/name;h=hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        assert p.stat().st_size==info['size'] and h.hexdigest()==info['sha256']
    write('base_path.json',{'path':str(source),'verified':True,'revision':cfg['base_revision']})
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer');longest=max(rows('train'),key=lambda r:len(r['input_ids']))
    model=load(source).train();n=cfg['max_length'];actual=len(longest['input_ids']);batch=cfg['micro_batch']
    inputs={'input_ids':torch.tensor([longest['input_ids']+[tok.pad_token_id]*(n-actual)]*batch,device='cuda'),'labels':torch.tensor([longest['labels']+[-100]*(n-actual)]*batch,device='cuda'),'attention_mask':torch.tensor([[1]*actual+[0]*(n-actual)]*batch,device='cuda')}
    opt=bnb.optim.AdamW8bit(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    torch.cuda.reset_peak_memory_stats();start=time.time();losses=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg['gradient_accumulation']):
            loss=model(**inputs).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite memory preflight loss')
            (loss/cfg['gradient_accumulation']).backward();losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        status('MEMORY_PREFLIGHT',step=step+1,peak_gib=torch.cuda.max_memory_allocated()/1024**3)
    peak=torch.cuda.max_memory_allocated();total=torch.cuda.get_device_properties(0).total_memory
    model.eval();model.config.num_logits_to_keep=1
    message=[{'role':'user','content':'Yalnızca {"hazir": true} JSON nesnesini yaz.'}]
    encoded=tok(tok.apply_chat_template(message,tokenize=False,add_generation_prompt=True),add_special_tokens=False,return_tensors='pt').to('cuda')
    with torch.inference_mode():answer=model.generate(**encoded,max_new_tokens=64,do_sample=False,use_cache=True,eos_token_id=[519],pad_token_id=tok.pad_token_id)
    result={'passed':peak<.85*total,'optimizer_steps':2,'padded_length':n,'actual_length':actual,'peak_allocated_gib':peak/1024**3,'total_gib':total/1024**3,'duration_seconds':time.time()-start,'finite_loss':all(math.isfinite(v) for v in losses),'settings':{k:cfg[k] for k in ['micro_batch','gradient_accumulation','max_length']},'generation_smoke_completed':True,'generation_smoke_tokens':int(answer.shape[1]-encoded['input_ids'].shape[1]),'note':'Scratch weights discarded. Training reloads original authenticated v2 weights.'}
    write('memory_preflight.json',result)
    if not result['passed']:raise RuntimeError('Insufficient VRAM headroom')
    status('PREFLIGHT_PASSED',**result)


def train(cfg):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer,Trainer,TrainingArguments,TrainerCallback
    from src.train.hub import upload_checkpoint
    memory=json.loads((ROOT/'memory_preflight.json').read_text());assert memory['passed'] and memory['optimizer_steps']==2
    assert memory['settings']=={k:cfg[k] for k in ['micro_batch','gradient_accumulation','max_length']}
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer');model=load(json.loads((ROOT/'base_path.json').read_text())['path'])
    training=rows('train');validation=rows('validation')
    def collate(batch):
        n=math.ceil(max(len(r['input_ids']) for r in batch)/8)*8
        return {'input_ids':torch.tensor([r['input_ids']+[tok.pad_token_id]*(n-len(r['input_ids'])) for r in batch]),'labels':torch.tensor([r['labels']+[-100]*(n-len(r['labels'])) for r in batch]),'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(n-len(r['input_ids'])) for r in batch])}
    started=time.time()
    class Monitor(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kw):
            if logs:
                if any(not math.isfinite(float(v)) for k,v in logs.items() if k in ['loss','eval_loss']):raise ValueError('Nonfinite training loss')
                with (ROOT/'metrics.jsonl').open('a') as out:out.write(json.dumps({'time':time.time(),'step':state.global_step,**logs})+'\n')
                status('TRAINING',step=state.global_step,max_steps=state.max_steps,epoch=state.epoch,metrics=logs,peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,elapsed_seconds=time.time()-started)
        def on_step_end(self,args,state,control,**kw):
            if state.global_step==state.max_steps:control.should_evaluate=True;control.should_save=True
            if (ROOT/'STOP').exists():control.should_training_stop=True;control.should_save=True
            return control
        def on_save(self,args,state,control,**kw):
            receipt=upload_checkpoint(Path(args.output_dir)/f'checkpoint-{state.global_step}',cfg['output_repo']);write('checkpoint_receipt.json',receipt)
    args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),num_train_epochs=cfg['epochs'],per_device_train_batch_size=cfg['micro_batch'],gradient_accumulation_steps=cfg['gradient_accumulation'],per_device_eval_batch_size=cfg['eval_micro_batch'],learning_rate=cfg['learning_rate'],weight_decay=.01,warmup_ratio=cfg['warmup_ratio'],lr_scheduler_type='cosine',optim=cfg['optimizer'],bf16=True,gradient_checkpointing=False,logging_steps=10,logging_nan_inf_filter=False,eval_strategy='steps',eval_steps=cfg['eval_save_steps'],save_strategy='steps',save_steps=cfg['eval_save_steps'],save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,group_by_length=True,prediction_loss_only=True,report_to='none',seed=cfg['seed'],dataloader_num_workers=2)
    trainer=Trainer(model=model,args=args,train_dataset=Dataset.from_list([{'input_ids':r['input_ids'],'labels':r['labels']} for r in training]),eval_dataset=Dataset.from_list([{'input_ids':r['input_ids'],'labels':r['labels']} for r in validation]),data_collator=collate,processing_class=tok,callbacks=[Monitor()]);trainer.model_accepts_loss_kwargs=False
    baseline=trainer.evaluate()['eval_loss'];write('baseline_development.json',{'loss':baseline});trainer.train();final_loss=trainer.evaluate()['eval_loss']
    trainer.state.save_to_json(str(ROOT/'trainer_state_final.json'))
    final=ROOT/'final';model=trainer.accelerator.unwrap_model(model,keep_fp32_wrapper=False);model.config.use_cache=True;model.save_pretrained(final,safe_serialization=True);tok.save_pretrained(final)
    provenance={'base_repo':cfg['base_repo'],'base_revision':cfg['base_revision'],'training_steps':trainer.state.global_step,'expected_steps':trainer.state.max_steps,'best_checkpoint':trainer.state.best_model_checkpoint,'baseline_loss':baseline,'final_loss':final_loss,'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()}
    (final/'sft_provenance.json').write_text(json.dumps(provenance,indent=2));write('final_receipt.json',upload_checkpoint(final,cfg['output_repo']))
    status('FINAL_EXPORTED',verified=True,**provenance)
    del trainer,model,training,validation;gc.collect();torch.cuda.empty_cache()
    finalize(cfg)


def finalize(cfg):
    import torch
    import final_eval
    from huggingface_hub import HfApi
    final=ROOT/'final';provenance=json.loads((final/'sft_provenance.json').read_text())
    if not (ROOT/'STOP').exists():
        from transformers import AutoTokenizer
        from instruction_data import verify
        status('DEVELOPMENT_GENERATION')
        tok=AutoTokenizer.from_pretrained(final);tok.padding_side='left';model=load(final).eval();model.config.num_logits_to_keep=1
        selected=[];families={}
        for row in json.loads((ROOT/'probes.json').read_text()):
            if families.get(row['family'],0)<2:selected.append(row);families[row['family']]=families.get(row['family'],0)+1
        results=[]
        for offset in range(0,len(selected),4):
            group=selected[offset:offset+4];texts=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True) for r in group]
            encoded=tok(texts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():out=model.generate(**encoded,max_new_tokens=384,do_sample=False,use_cache=True,eos_token_id=[519],pad_token_id=tok.pad_token_id)
            for row,ids in zip(group,out[:,encoded['input_ids'].shape[1]:]):
                response=tok.decode(ids,skip_special_tokens=True).strip();results.append({**row,'response':response,'correct':verify(row,response),'truncated':519 not in ids.tolist()})
        write('development_generation.json',{'n':len(results),'correct':sum(r['correct'] for r in results),'cases':results,'note':'Same procedural task templates as training. Development diagnostic only.'})
        del model;gc.collect();torch.cuda.empty_cache()
    status('FINAL_BENCHMARK',training_steps=provenance['training_steps'])
    report=final_eval.run(final,cfg);gc.collect();torch.cuda.empty_cache();new=None
    prior=json.loads((ROOT/'eval_language_v2.json').read_text())
    if not (ROOT/'STOP').exists():
        status('RETENTION_EVALUATION')
        with (ROOT/'language-eval.log').open('w') as out:subprocess.run(['python','-u','-m','scripts.cpt_evaluate','--model',str(final),'--cache',str(ROOT/'cpt_validation/manifest.json'),'--output',str(ROOT/'eval_language_v3.json')],cwd='/workspace/mamba-cpt-tr',stdout=out,stderr=subprocess.STDOUT,check=True)
        new=json.loads((ROOT/'eval_language_v3.json').read_text())
    decision=acceptance(cfg,report,prior,new,provenance,(ROOT/'STOP').exists());write('acceptance.json',{**decision,'benchmark':report,'provenance':provenance})
    card='# Jamba2-3B Turkish SFT v3 candidate\n\nPrivate candidate. Final model: `checkpoints/final`.\n\n'+json.dumps({'acceptance':decision,'benchmark':report},indent=2)+'\n\nProtocol: https://github.com/serda-dev/jamba-sft-turkish/blob/main/sft_v3/PLAN_TR.md\n'
    HfApi().upload_file(repo_id=cfg['output_repo'],path_in_repo='README.md',path_or_fileobj=card.encode())
    status('STOPPED_EXPORTED' if (ROOT/'STOP').exists() else 'DONE',verified=True,**decision,benchmark=report)


def main():
    import torch
    from transformers import set_seed
    cfg=json.loads((ROOT/'config.json').read_text());os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip();torch.set_num_threads(8);set_seed(cfg['seed'])
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['preflight','train','finalize'],required=True);args=parser.parse_args()
    {'preflight':preflight,'train':train,'finalize':finalize}[args.mode](cfg)


if __name__=='__main__':
    try:main()
    except BaseException as error:status('FAILED',error=type(error).__name__);raise
