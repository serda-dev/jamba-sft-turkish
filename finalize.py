"""Evaluate and export a completed SFT checkpoint without replaying training."""
import gc
import gzip
import json
import os
import subprocess
from pathlib import Path
from train import ROOT,status


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from src.train.hub import upload_checkpoint
    os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip()
    cfg=json.loads((ROOT/'config.json').read_text());set_seed(cfg['seed'])
    latest=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))[-1]
    state=json.loads((latest/'trainer_state.json').read_text())
    assert state['global_step']==3944 and state['epoch']==2.0
    best=Path(state['best_model_checkpoint'])
    assert best.parent==ROOT/'checkpoints' and best.is_dir()
    status('FINALIZING_COMPLETED_TRAINING',checkpoint=str(best),completed_step=state['global_step'])
    # Fresh inference model has no Trainer/Accelerate forward wrappers.
    model=AutoModelForCausalLM.from_pretrained(best,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    tok=AutoTokenizer.from_pretrained(best);tok.padding_side='right'
    model.config.use_cache=False;model.config.num_logits_to_keep=0
    model.generation_config.eos_token_id=list(set([tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')]))
    model.generation_config.pad_token_id=tok.pad_token_id
    with gzip.open(ROOT/'validation.jsonl.gz','rt') as handle:data=[json.loads(line) for line in handle]
    losses=[]
    with torch.inference_mode():
        for offset in range(0,len(data),2):
            batch=data[offset:offset+2];length=max(len(r['input_ids']) for r in batch)
            args={'input_ids':torch.tensor([r['input_ids']+[tok.pad_token_id]*(length-len(r['input_ids'])) for r in batch],device='cuda'),
                  'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(length-len(r['input_ids'])) for r in batch],device='cuda'),
                  'labels':torch.tensor([r['labels']+[-100]*(length-len(r['labels'])) for r in batch],device='cuda')}
            loss=float(model(**args,use_cache=False).loss)
            losses.extend([loss]*len(batch))
    final_loss=sum(losses)/len(losses)
    rows=[]
    cases=json.loads((ROOT/'eval_cases.json').read_text())
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
    diagnostic={'label':'sft_final','correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows}
    (ROOT/'eval_sft_final.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2))
    final=ROOT/'final';model.save_pretrained(final,safe_serialization=True);tok.save_pretrained(final)
    (final/'sft_provenance.json').write_text(json.dumps({'base_repo':cfg['base_repo'],'base_revision':cfg['base_revision'],
        'best_checkpoint':str(best),'completed_training_step':3944,'epochs':2,'validation_loss':final_loss},indent=2))
    receipt=upload_checkpoint(final,cfg['output_repo']);(ROOT/'final_receipt.json').write_text(json.dumps(receipt,indent=2))
    del model;gc.collect();torch.cuda.empty_cache()
    status('FINAL_LANGUAGE_EVALUATION',verified=True,final_loss=final_loss,diagnostic_correct=diagnostic['correct'])
    with (ROOT/'language-eval.log').open('w') as log:
        subprocess.run(['python','-u','-m','scripts.cpt_evaluate','--model',str(final),
            '--cache',str(ROOT/'cpt_validation/manifest.json'),'--output',str(ROOT/'eval_language_sft_final.json')],
            cwd='/workspace/mamba-cpt-tr',stdout=log,stderr=subprocess.STDOUT,check=True)
    old=json.loads((ROOT/'eval_language_cpt_baseline.json').read_text());new=json.loads((ROOT/'eval_language_sft_final.json').read_text())
    baseline_loss=json.loads((ROOT/'validation_baseline.json').read_text())['eval_loss']
    baseline=json.loads((ROOT/'eval_cpt_baseline.json').read_text())
    checks={name:new['scores'][name]['mean_nll']<=old['scores'][name]['mean_nll']*tolerance
            for name,tolerance in [('turkish',1.05),('english',1.10)]}
    accepted=final_loss<baseline_loss and diagnostic['correct']>=baseline['correct'] and all(checks.values())
    status('DONE',verified=True,accepted=accepted,step=3944,epoch=2.0,baseline_loss=baseline_loss,final_loss=final_loss,
        diagnostic_before=baseline['correct'],diagnostic_after=diagnostic['correct'],language_checks=checks,
        best_checkpoint=str(best),finalized_without_retraining=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:status('FAILED',error=type(error).__name__);raise
