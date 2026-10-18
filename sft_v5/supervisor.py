"""One bounded SFT v5 rental; verified exports before automatic deletion."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT=Path('/home/serda/sft-v5-run-20261018');PROJECT=Path('/home/serda/jamba-sft-turkish');STATE=ROOT/'supervisor.json';LABEL='linguai-sft-v5-20261018'
sys.path.insert(0,str(PROJECT))
import supervisor as previous
previous.ROOT=ROOT;previous.STATE=STATE
import scripts.cpt_vps_supervisor as transport
transport.ROOT=ROOT
from scripts.cpt_vps_supervisor import vast,ssh,copy_to,IMAGE,SSH_KEY


def save(state,**fields):
    state.update(fields,updated_at=time.time());p=STATE.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(STATE)
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),json.dumps(fields),flush=True)


def journal(event,details):
    stamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    with (ROOT/'OPERATIONS_TR.md').open('a') as out:out.write(f'\n### {stamp} — {event}\n\n```json\n'+json.dumps(details,ensure_ascii=False,indent=2)+'\n```\n')


def budget(cfg,state,credit,now):
    spent=max(0,state['initial_credit']-credit);elapsed=now-state['started_at']
    hard=spent>=cfg['max_run_cost_usd'] or elapsed>=cfg['max_hours']*3600 or credit<=cfg['hard_stop_credit']
    soft=spent>=cfg['max_run_cost_usd']-1 or elapsed>=(cfg['max_hours']*3600-1200) or credit<=cfg['stop_credit']
    return hard,soft,spent


def stop(state,stage,reason):
    item=next((i for i in vast('show','instances') if i['id']==state.get('instance_id')),None)
    if item:
        assert item.get('label')==LABEL
        vast('stop','instance',item['id'])
    save(state,stage=stage,reason=reason);journal(stage,{'reason':reason,'instance_id':state.get('instance_id')})


def remote(state,command,timeout=60):
    for attempt in range(3):
        try:return ssh(state,command,timeout=timeout)
        except subprocess.CalledProcessError as exc:
            if exc.returncode!=255 or attempt==2:raise
        except subprocess.TimeoutExpired:
            if attempt==2:raise
        time.sleep(3)


def main():
    from huggingface_hub import HfApi
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get('stage')=='COMPLETE':return
    cfg=json.loads((ROOT/'config.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text());audit=json.loads((ROOT/'data_audit.json').read_text())
    assert audit['verified'] and audit['manifest_sha256']==hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    api=HfApi();api.create_repo(cfg['output_repo'],private=True,exist_ok=True);assert api.model_info(cfg['output_repo']).private
    if not state.get('prepared_backup'):
        commit=api.upload_folder(repo_id=cfg['output_repo'],folder_path=str(ROOT/'hub_publish'),path_in_repo='prepared',commit_message='Audited broad v5 replay, weighted objective and FP32 master training before rental')
        stored={s.rfilename:s for s in api.model_info(cfg['output_repo'],revision=commit.oid,files_metadata=True).siblings}
        for path in (ROOT/'hub_publish').iterdir():
            if not path.is_file():continue
            data=path.read_bytes();item=stored['prepared/'+path.name];assert item.size==len(data)
            assert (item.lfs.sha256==hashlib.sha256(data).hexdigest()) if item.lfs else (item.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest())
        save(state,stage='PREPARED',prepared_backup=commit.oid)
    if not state.get('instance_id'):
        existing=[i for i in vast('show','instances') if i.get('label')==LABEL]
        if existing:save(state,instance_id=existing[0]['id'])
        else:
            if state.get('rent_requested'):raise RuntimeError('Uncertain prior rental; inspect before retrying')
            credit=vast('show','user')['credit'];assert credit>=cfg['max_run_cost_usd']+cfg['hard_stop_credit']
            offers=vast('search','offers','gpu_name=A100_SXM4 gpu_ram>=79 num_gpus=1 rentable=true rented=false reliability>0.98 verified=true cpu_ram>=32 cpu_cores_effective>=8','--storage',70,'--limit',20,'--order','dph_total')
            good=[o for o in offers if o['dph_total']<=cfg['rental_ceiling_usd_per_hour'] and o['end_date']>time.time()+(cfg['max_hours']+1)*3600 and o['dph_total']*cfg['max_hours']+30*o.get('inet_up_cost',1)+15*o.get('inet_down_cost',1)<cfg['max_run_cost_usd']]
            if not good:raise RuntimeError('No A100 within compute and transfer budget')
            offer=min(good,key=lambda o:o['dph_total']*cfg['max_hours']+30*o['inet_up_cost']+15*o['inet_down_cost'])
            save(state,stage='RENTING',rent_requested=True,initial_credit=cfg.get('budget_initial_credit',credit),started_at=cfg.get('budget_started_at',time.time()),hourly_rate=offer['dph_total'],offer_id=offer['id'])
            journal('GPU kiralama isteği',{'offer_id':offer['id'],'hourly_rate':offer['dph_total'],'credit':credit,'cost_cap':cfg['max_run_cost_usd']})
            result=vast('create','instance',offer['id'],'--disk',70,'--image',IMAGE,'--ssh','--direct','--label',LABEL,'--env','-e AUTO_INSTALL=0 -e AUTO_TRAIN=0 -e HF_HOME=/mnt/home_extra/hf-cache','--onstart-cmd','bash /workspace/mamba-cpt-tr/scripts/vast_onstart.sh','--cancel-unavail')
            save(state,stage='BOOTING',instance_id=result['new_contract']);journal('GPU kiralandı',{'instance_id':state['instance_id']})
    vast('attach','ssh',state['instance_id'],Path(SSH_KEY+'.pub').read_text().strip())
    for attempt in range(75):
        item=next(i for i in vast('show','instances') if i['id']==state['instance_id']);assert item['label']==LABEL
        host,port=transport.select_connection(state,item);save(state,ssh_host=host,ssh_port=port)
        try:ssh(state,'true',timeout=12);break
        except (subprocess.SubprocessError,OSError):time.sleep(8)
    else:raise RuntimeError('GPU SSH timeout')
    if not state.get('launched'):
        ssh(state,'mkdir -p /mnt/home_extra/sft')
        with tarfile.open(ROOT/'payload.tar.gz','w:gz') as tar:
            for name in ['train.jsonl.gz','validation.jsonl.gz','train_messages.jsonl.gz','validation_messages.jsonl.gz','manifest.json','config.json','dev_cases.json','retention_cases.json','tokenizer','base_receipt.json','sealed_math_cases.json','code_manifest.json']:tar.add(ROOT/name,arcname=name)
            for name in cfg.get('reuse_baseline_reports',[]):tar.add(ROOT/name,arcname=name)
            tar.add(PROJECT/'sft_v4/runtime.py',arcname='v4_runtime.py');tar.add(PROJECT/'sft_v4/data.py',arcname='grounded_data.py');tar.add(PROJECT/'sft_v5/runtime.py',arcname='runtime.py');tar.add(PROJECT/'sft_v5/data.py',arcname='instruction_data.py');tar.add(PROJECT/'sft_v2/final_eval.py',arcname='final_eval.py')
            tar.add(PROJECT/'research/instruction_following_20261017/pilot_runtime.py',arcname='precision.py')
            tar.add(PROJECT/'benchmarks',arcname='benchmark_code',filter=lambda i:None if '__pycache__' in i.name else i)
            tar.add(Path('/home/serda/benchmark-run-20261014/cases.jsonl'),arcname='benchmark_cases.jsonl')
            tar.add(Path('/home/serda/cpt-run-20261011/cache/token_cache/phase_1_validation'),arcname='cpt_validation')
            tar.add(Path('/home/serda/sft-v2-run-20261015/eval_language_v2.json'),arcname='eval_language_v2.json')
        copy_to(state,ROOT/'payload.tar.gz','/mnt/home_extra/sft/payload.tar.gz')
        for directory in ['src','eval','customtokenizer']:copy_to(state,Path('/home/serda/cpt-production')/directory,'/workspace/mamba-cpt-tr/',recursive=True)
        for name in ['cpt_gpu_preflight.py','cpt_evaluate.py']:copy_to(state,Path('/home/serda/cpt-production/scripts')/name,'/workspace/mamba-cpt-tr/scripts/'+name)
        ssh(state,'umask 077; cat > /mnt/home_extra/sft/hf_token',data=Path('/home/serda/.cache/huggingface/token').read_bytes())
        ssh(state,"cd /mnt/home_extra/sft && tar xzf payload.tar.gz && python -m pip install accelerate==1.10.1 langdetect==1.0.9 immutabledict nltk absl-py==2.3.1 > setup.log 2>&1 && PYTHONPATH=/workspace/mamba-cpt-tr:benchmark_code python -c 'import scoring,numeric_score; from scripts import cpt_evaluate'",timeout=240)
        verify_code="import hashlib,json;from pathlib import Path;r=Path('/mnt/home_extra/sft');want=json.loads((r/'code_manifest.json').read_text());assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==v for p,v in want.items());print(json.dumps({'verified_code_files':len(want)}))"
        verified=ssh(state,'python -c '+shlex.quote(verify_code));journal('GPU kod dosyaları hash doğrulandı',json.loads(verified))
        ssh(state,"cd /mnt/home_extra/sft && python -c \"import json,time;from pathlib import Path;Path('status.json').write_text(json.dumps({'stage':'STARTING','updated_at':time.time()}))\" && (nohup env PYTHONPATH=/workspace/mamba-cpt-tr python -u runtime.py > training.log 2>&1 < /dev/null & echo $! > job.pid)")
        save(state,stage='RUNNING',launched=True,launched_at=time.time());journal('Bellek kontrolü ve eğitim işi başlatıldı',{'instance_id':state['instance_id'],'epochs':cfg['epochs'],'learning_rate':cfg['learning_rate']})
    errors=0;last_event=None
    while True:
        try:
            credit=vast('show','user')['credit'];hard,soft,spent=budget(cfg,state,credit,time.time());save(state,credit=credit,spent=spent)
            if hard:stop(state,'BUDGET_STOPPED_RETAINED','Cost or time limit');return
            if soft and not state.get('stop_requested'):
                remote(state,'touch /mnt/home_extra/sft/STOP');save(state,stop_requested=True);journal('Güvenli export istendi',{'spent':spent,'credit':credit})
            result=json.loads(remote(state,'cat /mnt/home_extra/sft/status.json'));save(state,stage='RUNNING',remote_status=result)
            event=(result['stage'],result.get('step'))
            if event!=last_event:journal('GPU ilerleme',result);last_event=event
            terminal=result['stage'] in ['DONE','STOPPED_EXPORTED','FAILED']
            if terminal or time.time()-state.get('last_backup',0)>300:
                logs=previous.backup(state);save(state,last_backup=time.time(),log_receipt=logs)
                api.upload_file(repo_id=cfg['output_repo'],path_in_repo='run/OPERATIONS_TR.md',path_or_fileobj=str(ROOT/'OPERATIONS_TR.md'))
            if result['stage'] in ['DONE','STOPPED_EXPORTED']:
                receipt=previous.verify_final();assert logs['verified']
                vast('destroy','instance',state['instance_id'],'--yes');assert not any(i['id']==state['instance_id'] for i in vast('show','instances'))
                save(state,stage='COMPLETE',instance_deleted=True,final_receipt=receipt);journal('Model/kayıt doğrulandı ve GPU silindi',{'instance_id':state['instance_id'],'accepted':result.get('accepted'),'credit':credit,'spent':spent,'final_revision':receipt['revision']})
                try:api.upload_file(repo_id=cfg['output_repo'],path_in_repo='run/OPERATIONS_TR.md',path_or_fileobj=str(ROOT/'OPERATIONS_TR.md'))
                except Exception as error:save(state,journal_upload_error=type(error).__name__)
                return
            if result['stage']=='FAILED':stop(state,'FAILED_RETAINED',result.get('error','Unknown failure'));return
            health=json.loads(remote(state,"python -c \"import os,time,json;from pathlib import Path;r=Path('/mnt/home_extra/sft');os.kill(int((r/'job.pid').read_text()),0);print(json.dumps({'age':time.time()-max(p.stat().st_mtime for p in [r/'training.log',r/'language-eval.log'] if p.exists())}))\""))
            if health['age']>1800:stop(state,'STALLED_RETAINED','No log progress for 30 minutes');return
            errors=0;save(state,last_error=None,consecutive_errors=0)
        except (subprocess.SubprocessError,OSError,ValueError,RuntimeError,KeyError,AssertionError) as error:
            errors+=1;save(state,last_error=type(error).__name__,consecutive_errors=errors)
            if errors>=3:stop(state,'MONITOR_FAILED_RETAINED','Three monitor failures');return
        time.sleep(60)


def watchdog():
    if not STATE.exists():return
    state=json.loads(STATE.read_text())
    if state.get('stage')=='COMPLETE':return
    active=subprocess.run(['systemctl','--user','is-active','--quiet','linguai-sft-v5-supervisor.service']).returncode==0
    items=vast('show','instances')
    if not state.get('instance_id'):
        if not state.get('rent_requested') or active:return
        own=[i for i in items if i.get('label')==LABEL]
        if len(own)!=1:return
        state['instance_id']=own[0]['id']
    item=next((i for i in items if i['id']==state['instance_id']),None)
    if not item or item.get('label')!=LABEL or item.get('intended_status')=='stopped':return
    cfg=json.loads((ROOT/'config.json').read_text());credit=vast('show','user')['credit'];hard,_,_=budget(cfg,state,credit,time.time())
    if hard or not active:stop(state,'WATCHDOG_STOPPED_RETAINED','Hard budget or main monitor inactive')


if __name__=='__main__':
    try:
        if '--watchdog' in sys.argv:watchdog()
        else:main()
    except BaseException as error:
        state=json.loads(STATE.read_text()) if STATE.exists() else {}
        if state.get('instance_id'):stop(state,'FAILED_RETAINED',type(error).__name__)
        journal('Hata',{'error':type(error).__name__});raise
