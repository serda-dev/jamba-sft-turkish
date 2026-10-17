"""Disposable three-arm learning and batch/cache diagnostic on one GPU."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

ROOT=Path('/mnt/home_extra/sft')


def write(name,value):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)


def status(stage,**fields):
    row={'stage':stage,'updated_at':time.time(),**fields};write('status.json',row)
    with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(stage,json.dumps(fields,ensure_ascii=False),flush=True)


def generate(model,tok,cases,cfg,tag,batch_size=8,cache=True,system=False,eos_ids=None):
    import torch
    from contracts import check
    model.eval();model.config.num_logits_to_keep=1;tok.padding_side='left';results=[];logits=[];eos_ids=eos_ids or [519]
    for offset in range(0,len(cases),batch_size):
        if (ROOT/'STOP').exists():break
        group=cases[offset:offset+batch_size];texts=[]
        for r in group:
            messages=r['messages'][:-1]
            if system:messages=[{'role':'system','content':'Sen yardımsever bir Türkçe asistansın. Kullanıcının istediği koşullara uy.'}]+messages
            texts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
        batch=tok(texts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model.generate(**batch,max_new_tokens=cfg['max_new_tokens'],do_sample=False,use_cache=cache,eos_token_id=eos_ids,pad_token_id=tok.pad_token_id,return_dict_in_generate=True,output_scores=True)
        for i,(r,ids) in enumerate(zip(group,out.sequences[:,batch['input_ids'].shape[1]:])):
            values=ids.tolist();ends=[values.index(e) for e in eos_ids if e in values];raw=values[:min(ends)+1] if ends else values
            response=tok.decode(raw,skip_special_tokens=True).strip();score=out.scores[0][i].float().cpu();top=torch.topk(score,2)
            results.append({'id':r['id'],'family':r['family'],'response':response,'token_ids':raw,'correct':check(r['contracts'],response),'eos_519':519 in raw,'eos_2_count':raw.count(2),'truncated':not any(e in raw for e in eos_ids),'first_token':int(top.indices[0]),'first_token_margin':float(top.values[0]-top.values[1])});logits.append(score)
        del out
    report={'tag':tag,'n':len(results),'correct':sum(r['correct'] for r in results),'accuracy':sum(r['correct'] for r in results)/max(len(results),1),'truncated':sum(r['truncated'] for r in results),'cache':cache,'batch_size':batch_size,'system':system,'eos_token_ids':eos_ids,'cases':results}
    write(tag+'.json',report);status('GENERATION',tag=tag,n=report['n'],correct=report['correct'],truncated=report['truncated'])
    return report,logits


def sampled_parameters(model):
    import torch
    result={}
    for name,p in model.named_parameters():
        if p.ndim<2:continue
        idx=torch.linspace(0,p.numel()-1,2048,device=p.device).long()
        result[name]=(p,idx,p.detach().flatten()[idx].float().clone())
        if len(result)==8:break
    return result


def delta(samples):
    out={}
    for name,(p,idx,before) in samples.items():
        diff=p.detach().flatten()[idx].float()-before
        out[name]={'storage_dtype':str(p.dtype),'sample_count':len(idx),'changed_fraction':float((diff!=0).float().mean()),'mean_abs_change':float(diff.abs().mean()),'max_abs_change':float(diff.abs().max())}
    return out


def train_arm(source,tok,encoded,cfg,arm):
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM,set_seed
    set_seed(cfg['seed']);model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=getattr(torch,arm['storage_dtype']),attn_implementation='sdpa').to('cuda').train()
    assert model.num_parameters()==3039823232 and all(p.requires_grad for p in model.parameters())
    model.config.use_cache=False;model.config.num_logits_to_keep=0
    opt=bnb.optim.AdamW8bit(model.parameters(),lr=arm['learning_rate'],weight_decay=.01)
    rng=random.Random(cfg['seed']);order=list(range(len(encoded)));rng.shuffle(order);cursor=0;losses=[];samples=sampled_parameters(model);first=None
    torch.cuda.reset_peak_memory_stats();started=time.time();steps=0
    for step in range(cfg['optimizer_steps']):
        if (ROOT/'STOP').exists():break
        opt.zero_grad(set_to_none=True);micro_losses=[]
        for _ in range(cfg['gradient_accumulation']):
            row=encoded[order[cursor]];cursor+=1
            if cursor==len(order):cursor=0;rng.shuffle(order)
            batch={k:torch.tensor([row[k]],device='cuda') for k in ['input_ids','labels']};batch['attention_mask']=torch.ones_like(batch['input_ids'])
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite diagnostic loss')
            (loss/cfg['gradient_accumulation']).backward();micro_losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if step==0:
            grads={name:{'dtype':str(p.grad.dtype),'nonzero_fraction':float((p.grad.detach().flatten()[idx]!=0).float().mean())} for name,(p,idx,_) in samples.items() if p.grad is not None}
        opt.step();steps=step+1
        if step==0:first={'gradients':grads,'parameter_updates':delta(samples)}
        losses.append(sum(micro_losses)/len(micro_losses))
        if steps%10==0:status('TRAINING',arm=arm['name'],step=steps,max_steps=cfg['optimizer_steps'],loss=losses[-1],grad_norm=float(norm),peak_gib=torch.cuda.max_memory_allocated()/1024**3)
    report={'arm':arm,'steps':steps,'duration_seconds':time.time()-started,'first_optimizer_step':first,'parameter_updates_final':delta(samples),'losses':losses,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
    write(arm['name']+'_training.json',report)
    del opt,samples;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    return model,report


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from huggingface_hub import snapshot_download
    from scripts.cpt_gpu_preflight import check_gpu_runtime
    cfg=json.loads((ROOT/'config.json').read_text());cases=json.loads((ROOT/'cases.json').read_text());encoded=json.loads((ROOT/'encoded.json').read_text())
    os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip();torch.set_num_threads(8);set_seed(cfg['seed'])
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    status('PREFLIGHT',runtime=check_gpu_runtime())
    prefix=cfg['base_subfolder']+'/'
    source=Path(snapshot_download(cfg['base_repo'],revision=cfg['base_revision'],allow_patterns=[prefix+p for p in ['*.json','*.jinja','*.safetensors']],max_workers=4))/cfg['base_subfolder']
    receipt=json.loads((ROOT/'base_receipt.json').read_text())
    for name,v in receipt['files'].items():
        p=source/name;h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        assert p.stat().st_size==v['size'] and h.hexdigest()==v['sha256']
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
    model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();model.config.num_logits_to_keep=1
    variants=[('b1_cache',1,True,False,[519]),('b16_cache',16,True,False,[519]),('b1_no_cache',1,False,False,[519]),('b16_no_cache',16,False,False,[519]),('b16_system',16,True,True,[519]),('b16_eos_both',16,True,False,[2,519])]
    reference=None;path_results={}
    for tag,size,cache,system,eos_ids in variants:
        if (ROOT/'STOP').exists():break
        report,logits=generate(model,tok,cases['dev'][:16],cfg,tag,size,cache,system,eos_ids)
        if reference is None:reference=(report,logits)
        if not system and len(logits)==16:
            report['comparison_to_b1_cache']={'first_token_equal':sum(a['first_token']==b['first_token'] for a,b in zip(reference[0]['cases'],report['cases'])),'entire_response_tokens_equal':sum(a['token_ids']==b['token_ids'] for a,b in zip(reference[0]['cases'],report['cases'])),'max_abs_first_logits_difference':max(float((a-b).abs().max()) for a,b in zip(reference[1],logits))}
            write(tag+'.json',report)
        path_results[tag]={k:v for k,v in report.items() if k!='cases'}
    if not (ROOT/'STOP').exists():
        base_train,_=generate(model,tok,cases['train'],cfg,'baseline_train')
        base_dev,_=generate(model,tok,cases['dev'],cfg,'baseline_dev')
    else:base_train=base_dev={'n':0,'accuracy':0}
    del model,reference;gc.collect();torch.cuda.empty_cache();arms=[]
    for arm in cfg['arms']:
        if (ROOT/'STOP').exists():break
        status('LOADING_ARM',arm=arm['name'])
        try:
            model,training=train_arm(source,tok,encoded,cfg,arm)
            train_report,_=generate(model,tok,cases['train'],cfg,arm['name']+'_train')
            dev_report,_=generate(model,tok,cases['dev'],cfg,arm['name']+'_dev')
            result={'arm':arm,'steps':training['steps'],'train_accuracy':train_report['accuracy'],'train_n':train_report['n'],'dev_accuracy':dev_report['accuracy'],'dev_n':dev_report['n'],'dev_truncated':dev_report['truncated'],'train_gate':train_report['n']==32 and train_report['accuracy']>=cfg['train_success_threshold']}
            arms.append(result);write('arm_results.json',arms)
            del model;gc.collect();torch.cuda.empty_cache()
        except Exception as error:
            write(arm['name']+'_error.json',{'error':type(error).__name__,'message':str(error)[:1000]});raise
    complete=len(arms)==len(cfg['arms']) and all(a['steps']==cfg['optimizer_steps'] and a['dev_n']==128 and a['train_n']==32 for a in arms)
    result={'complete':complete,'base_repo':cfg['base_repo'],'base_revision':cfg['base_revision'],'path_comparison':path_results,'baseline_train_accuracy':base_train['accuracy'],'baseline_dev_accuracy':base_dev['accuracy'],'arms':arms,'weights_saved':False,'note':'Scratch diagnostic; no production candidate. Task templates share across train/dev. Changes in storage regime may also change intermediate numerical precision. Full token and gradient/update evidence is in arm reports.'}
    write('pilot_result.json',result);status('DONE',complete=complete,arms=arms,weights_saved=False)


if __name__=='__main__':
    try:main()
    except BaseException as error:status('FAILED',error=type(error).__name__);raise
