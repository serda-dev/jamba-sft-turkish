"""Common BF16 Transformers quality and speed protocol; no training."""
import gc
import hashlib
import inspect
import json
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT=Path('/mnt/home_extra/benchmark')


def status(stage,**fields):
    data={'stage':stage,'time':time.time(),**fields};temp=ROOT/'status.tmp'
    temp.write_text(json.dumps(data,indent=2));temp.replace(ROOT/'status.json')
    print(json.dumps(data),flush=True)


def upload():
    from huggingface_hub import HfApi
    api=HfApi();repo='serda-dev/Jamba2-3B-Turkish-SFT-20261013'
    files=[p for pattern in ['*.json','*.jsonl','*.log'] for p in ROOT.glob(pattern) if p.name!='upload_receipt.json']
    folder=ROOT/'export';folder.mkdir(exist_ok=True)
    for p in files:(folder/p.name).write_bytes(p.read_bytes())
    prefix='benchmarks/20261014'
    commit=api.upload_folder(repo_id=repo,folder_path=str(folder),path_in_repo=prefix,commit_message='Turkish benchmark progress and raw responses')
    remote={s.rfilename:s for s in api.model_info(repo,revision=commit.oid,files_metadata=True).siblings}
    hashes={}
    for p in folder.iterdir():
        data=p.read_bytes();entry=remote[prefix+'/'+p.name]
        digest=hashlib.sha256(data).hexdigest()
        wanted=digest if entry.lfs else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        actual=(entry.lfs.sha256 if hasattr(entry.lfs,'sha256') else entry.lfs['sha256']) if entry.lfs else entry.blob_id
        assert entry.size==len(data) and wanted==actual
        hashes[p.name]=digest
    receipt={'verified':True,'repo':repo,'revision':commit.oid,'path':prefix,'sha256':hashes}
    (ROOT/'upload_receipt.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def summarize(manifest):
    results={}
    for config in manifest['models']:
        path=ROOT/f"responses_{config['id']}.jsonl"
        if not path.exists():continue
        tasks=defaultdict(list)
        for line in path.read_text().splitlines():
            row=json.loads(line);tasks[row['task']].append(row)
        result={}
        for task,rows in tasks.items():
            correct=sum(r['correct'] for r in rows);n=len(rows);p=correct/n;z=1.96
            center=(p+z*z/(2*n))/(1+z*z/n)
            margin=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
            result[task]={'correct':correct,'n':n,'accuracy':p,'wilson95':[center-margin,center+margin],
                          'truncated_outputs':sum(r.get('truncated',False) for r in rows)}
            if task=='ifeval_tr_subset':
                flags=[f for r in rows for f in r['checks']]
                result[task]['instruction_accuracy']=sum(flags)/len(flags)
            if task=='turkish_mmlu':
                result[task]['subjects']={s:{'n':sum(r['subject']==s for r in rows),'correct':sum(r['correct'] for r in rows if r['subject']==s)} for s in sorted({r['subject'] for r in rows})}
        speed=ROOT/f"speed_{config['id']}.json"
        if speed.exists():result['speed']=json.loads(speed.read_text())
        results[config['id']]=result
    (ROOT/'results.json').write_text(json.dumps({'manifest':manifest,'results':results},ensure_ascii=False,indent=2))


def main():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from transformers.generation.streamers import BaseStreamer
    from scoring import score
    from numeric_score import numeric
    from scripts.cpt_gpu_preflight import check_gpu_runtime
    os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip()
    manifest=json.loads((ROOT/'manifest.json').read_text());assert hashlib.sha256((ROOT/'cases.jsonl').read_bytes()).hexdigest()==manifest['case_sha256']
    cases=[json.loads(line) for line in (ROOT/'cases.jsonl').read_text().splitlines()]
    set_seed(manifest['seed']);torch.set_num_threads(8)
    status('PREFLIGHT',runtime=check_gpu_runtime())
    for config in manifest['models']:
        if (ROOT/'STOP').exists():status('STOPPED');upload();return
        ident=config['id'];status('LOADING',model=ident)
        folder=config.get('subfolder','')
        allowed=[(folder+'/' if folder else '')+p for p in ['*.json','*.jinja','*.safetensors','*.model','*.txt']]
        cache=snapshot_download(config['repo'],revision=config['revision'],allow_patterns=allowed,max_workers=4)
        path=Path(cache)/folder
        tok=AutoTokenizer.from_pretrained(path);tok.padding_side='left'
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        model.config.use_cache=True
        parameter_count=model.num_parameters()
        def encode(prompts):
            texts=[tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in prompts]
            return tok(texts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
        eos=model.generation_config.eos_token_id or tok.eos_token_id
        class Timer(BaseStreamer):
            def __init__(self):self.calls=0;self.first=None
            def put(self,value):
                self.calls+=1
                if self.calls==2:torch.cuda.synchronize();self.first=time.perf_counter()
            def end(self):pass
        speed_path=ROOT/f'speed_{ident}.json'
        if not speed_path.exists():
            measures=[]
            for input_length in [128,512,2048]:
                batch=encode([('Türkçe dilinin tarihini ve özelliklerini ayrıntılı anlat. ' * 600)])
                batch={k:v[:,-input_length:] for k,v in batch.items()}
                with torch.inference_mode():model.generate(**batch,max_new_tokens=16,min_new_tokens=16,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=eos)
                runs=[]
                for repeat in range(3):
                    timer=Timer();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    with torch.inference_mode():out=model.generate(**batch,max_new_tokens=128,min_new_tokens=128,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=eos,streamer=timer)
                    torch.cuda.synchronize();end=time.perf_counter();text=tok.decode(out[0,input_length:],skip_special_tokens=True)
                    runs.append({'ttft_ms':1000*(timer.first-start),'decode_tokens_s':127/(end-timer.first),
                      'end_to_end_tokens_s':128/(end-start),'characters_s':len(text)/(end-start),
                      'total_s':end-start,'peak_vram_gib':torch.cuda.max_memory_allocated()/1024**3})
                measures.append({'input_tokens':input_length,'output_tokens':128,'runs':runs,
                    'median':{k:statistics.median(r[k] for r in runs) for k in runs[0]}})
            batch=encode(['Türkçe olarak bir bilim kurgu hikayesinin ilk paragrafını yaz.']*4)
            with torch.inference_mode():model.generate(**batch,max_new_tokens=16,min_new_tokens=16,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=eos)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():model.generate(**batch,max_new_tokens=128,min_new_tokens=128,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=eos)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            speed={'parameters':parameter_count,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
                   'dtype':'BF16','attention':'sdpa','batch1':measures,'batch4_aggregate_tokens_s':512/elapsed}
            speed_path.write_text(json.dumps(speed,indent=2));summarize(manifest);upload()
        output=ROOT/f'responses_{ident}.jsonl';seen=set()
        if output.exists():seen={json.loads(line)['id'] for line in output.read_text().splitlines()}
        for task in ['turkish_mmlu','belebele_tr','gsm8k_tr','ifeval_tr_subset']:
            items=[r for r in cases if r['task']==task and r['id'] not in seen]
            if not items:continue
            status('EVALUATING',model=ident,task=task,remaining=len(items))
            batch_size=8 if items[0]['type']=='mcq' else 4
            for offset in range(0,len(items),batch_size):
                if (ROOT/'STOP').exists():status('STOPPED');summarize(manifest);upload();return
                group=items[offset:offset+batch_size];batch=encode([r['prompt'] for r in group])
                if group[0]['type']=='mcq':
                    options=[tok.encode(chr(65+i),add_special_tokens=False) for i in range(5)]
                    assert all(len(x)==1 for x in options)
                    kwargs={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
                    with torch.inference_mode():logits=model(**batch,use_cache=False,**kwargs).logits[:,-1].float()
                    for index,row in enumerate(group):
                        values=logits[index,[x[0] for x in options[:row['choices']]]].tolist()
                        prediction=max(range(len(values)),key=lambda i:values[i])
                        result={**row,'prediction':prediction,'choice_logits':values,'correct':prediction==row['gold']}
                        with output.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                else:
                    limit=512 if group[0]['type']=='math' else 2048
                    with torch.inference_mode():out=model.generate(**batch,max_new_tokens=limit,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=eos)
                    for index,row in enumerate(group):
                        ids=out[index,batch['input_ids'].shape[1]:];response=tok.decode(ids,skip_special_tokens=True).strip()
                        stop_ids=eos if isinstance(eos,list) else [eos]
                        truncated=len(ids)==limit and not any(int(t) in stop_ids for t in ids)
                        if row['type']=='math':
                            answer=response.rsplit('####',1)[-1] if '####' in response else response
                            nums=re.findall(r'-?\d+(?:[.,]\d+)*',answer)
                            prediction=nums[-1].replace(',','') if nums else None
                            correct=numeric(prediction)==numeric(row['gold']) if prediction is not None else False
                            flags=None
                        else:
                            flags=score(row['rules'],response);correct=all(flags);prediction=None
                        result={**row,'response':response,'prediction':prediction,'correct':correct,'checks':flags,'truncated':truncated}
                        with output.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                if (offset//batch_size)%10==0:status('EVALUATING',model=ident,task=task,completed=offset+len(group),total=len(items))
            summarize(manifest);upload()
        del model;gc.collect();torch.cuda.empty_cache()
    summarize(manifest);status('DONE');receipt=upload();status('DONE',verified=receipt['verified'])


if __name__=='__main__':
    try:main()
    except BaseException as error:status('FAILED',error=type(error).__name__);raise
