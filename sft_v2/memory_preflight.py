"""Two real AdamW8bit steps at maximum padded length, from preserved pilot weights."""
import gc
import gzip
import json
import math
import os
import time
from pathlib import Path
from pilot_recovery import verified_pilots,eligible_pilots,digest

ROOT=Path('/mnt/home_extra/sft')


def main():
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    cfg=json.loads((ROOT/'config.json').read_text());set_seed(cfg['seed']);torch.set_num_threads(8)
    outcomes=verified_pilots(ROOT,cfg);winner=min(eligible_pilots(outcomes),key=lambda o:o['loss_after'])
    longest=None
    with gzip.open(ROOT/'train.jsonl.gz','rt') as handle:
        for line in handle:
            row=json.loads(line)
            if longest is None or len(row['input_ids'])>len(longest['input_ids']):longest=row
    assert longest and len(longest['input_ids'])<=cfg['max_length']
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
    model=AutoModelForCausalLM.from_pretrained(winner['path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').train()
    model.config.use_cache=False;model.config.num_logits_to_keep=0
    assert model.num_parameters()==3039823232 and all(p.requires_grad for p in model.parameters())
    n=cfg['max_length'];actual=len(longest['input_ids']);batch=cfg['micro_batch']
    inputs={'input_ids':torch.tensor([longest['input_ids']+[tok.pad_token_id]*(n-actual)]*batch,device='cuda'),
            'labels':torch.tensor([longest['labels']+[-100]*(n-actual)]*batch,device='cuda'),
            'attention_mask':torch.tensor([[1]*actual+[0]*(n-actual)]*batch,device='cuda')}
    optimizer=bnb.optim.AdamW8bit(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    torch.cuda.reset_peak_memory_stats();start=time.time();losses=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg['gradient_accumulation']):
            loss=model(**inputs).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite preflight loss')
            (loss/cfg['gradient_accumulation']).backward()
            losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        print('MEMORY_STEP',step+1,'loss',losses[-1],'peak_GiB',torch.cuda.max_memory_allocated()/1024**3,flush=True)
    total=torch.cuda.get_device_properties(0).total_memory
    peak=torch.cuda.max_memory_allocated()
    report={'passed':peak<=.85*total,'settings':{k:cfg[k] for k in ['micro_batch','gradient_accumulation','max_length']},
            'optimizer_steps':2,'longest_actual_length':actual,'padded_length':n,'gpu':torch.cuda.get_device_name(),
            'peak_allocated_gib':peak/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
            'total_gib':total/1024**3,'duration_seconds':time.time()-start,'finite_loss':all(math.isfinite(v) for v in losses),
            'pilot_reuse_sha256':digest(ROOT/'pilot_reuse.json'),
            'note':'Isolated scratch process; mutated probe weights are discarded. Full training reloads preserved pilot weights.'}
    (ROOT/'memory_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('Less than 15% VRAM headroom; do not start full training')


if __name__=='__main__':main()
