"""Rent one diagnostic GPU within $2, preserve evidence and delete after backup."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import sft_v3.supervisor as shared
ROOT=Path('/home/serda/instruction-pilot-20261017');STATE=ROOT/'supervisor.json';LABEL='linguai-instruction-diagnostic-20261017'
shared.ROOT=ROOT;shared.STATE=STATE;shared.LABEL=LABEL
shared.previous.ROOT=ROOT;shared.previous.STATE=STATE;shared.transport.ROOT=ROOT
from sft_v3.supervisor import vast,ssh,copy_to,IMAGE,SSH_KEY,save,journal,remote,stop,budget
SOURCE=Path(__file__).resolve().parent
SERVICE='linguai-instruction-pilot.service'


def watchdog():
    if not STATE.exists():return
    state=json.loads(STATE.read_text())
    if state.get('stage')=='COMPLETE':return
    active=subprocess.run(['systemctl','--user','is-active','--quiet',SERVICE]).returncode==0
    items=vast('show','instances')
    if not state.get('instance_id'):
        if not state.get('rent_requested') or active:return
        candidates=[i for i in items if i.get('label')==LABEL]
        if len(candidates)!=1:return
        state['instance_id']=candidates[0]['id']
    item=next((i for i in items if i['id']==state['instance_id']),None)
    if not item or item['label']!=LABEL or item.get('intended_status')=='stopped':return
    cfg=json.loads((ROOT/'config.json').read_text());credit=vast('show','user')['credit']
    hard,_,_=budget(cfg,state,credit,time.time())
    if hard or not active:stop(state,'WATCHDOG_STOPPED_RETAINED','Diagnostic cap or main monitor inactive')


def main():
    from huggingface_hub import HfApi
    import hashlib
    cfg=json.loads((ROOT/'config.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text())
    assert manifest['train']==32 and manifest['dev']==128 and manifest['all_targets_verified']
    for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get('stage')=='COMPLETE':return
    api=HfApi();api.create_repo(cfg['output_repo'],private=True,exist_ok=True);assert api.model_info(cfg['output_repo']).private
    if not state.get('prepared_backup'):
        c=api.upload_folder(repo_id=cfg['output_repo'],folder_path=str(ROOT),path_in_repo='prepared',allow_patterns=['cases.json','encoded.json','base_receipt.json','manifest.json','config.json','tokenizer/*'])
        api.upload_folder(repo_id=cfg['output_repo'],folder_path=str(SOURCE),path_in_repo='source',allow_patterns=['*.py','*.md','*.json'])
        api.upload_file(repo_id=cfg['output_repo'],path_in_repo='README.md',path_or_fileobj=b'# Scratch instruction diagnostic\n\nPrivate reports only. No production model is trained or published here. All three scratch weight sets are discarded.\n')
        save(state,stage='PREPARED',prepared_backup=c.oid)
    if not state.get('instance_id'):
        existing=[i for i in vast('show','instances') if i.get('label')==LABEL]
        if existing:save(state,instance_id=existing[0]['id'])
        else:
            if state.get('rent_requested'):raise RuntimeError('Inspect uncertain prior rental')
            credit=vast('show','user')['credit'];assert credit>=cfg['hard_stop_credit']+cfg['max_run_cost_usd']
            offers=vast('search','offers','gpu_name=A100_SXM4 gpu_ram>=79 num_gpus=1 rentable=true rented=false reliability>0.98 verified=true','--storage',70,'--limit',20,'--order','dph_total')
            good=[o for o in offers if o['dph_total']<=cfg['rental_ceiling_usd_per_hour'] and o['end_date']>time.time()+3*3600 and o['dph_total']+10*o.get('inet_down_cost',1)+o.get('inet_up_cost',1)<1.6]
            if not good:raise RuntimeError('No qualifying diagnostic GPU')
            offer=min(good,key=lambda o:o['dph_total']+10*o['inet_down_cost']+o['inet_up_cost'])
            save(state,stage='RENTING',rent_requested=True,initial_credit=credit,started_at=time.time(),hourly_rate=offer['dph_total'],offer_id=offer['id'])
            journal('Tanı GPU kiralama isteği',{'offer_id':offer['id'],'hourly_rate':offer['dph_total'],'credit':credit,'cost_cap_usd':2,'max_hours':1})
            result=vast('create','instance',offer['id'],'--disk',70,'--image',IMAGE,'--ssh','--direct','--label',LABEL,'--env','-e AUTO_INSTALL=0 -e AUTO_TRAIN=0 -e HF_HOME=/mnt/home_extra/hf-cache','--onstart-cmd','bash /workspace/mamba-cpt-tr/scripts/vast_onstart.sh','--cancel-unavail')
            save(state,stage='BOOTING',instance_id=result['new_contract']);journal('GPU kiralandı',{'instance_id':state['instance_id']})
    vast('attach','ssh',state['instance_id'],Path(SSH_KEY+'.pub').read_text().strip())
    for attempt in range(60):
        item=next(i for i in vast('show','instances') if i['id']==state['instance_id']);assert item['label']==LABEL
        host,port=shared.transport.select_connection(state,item);save(state,ssh_host=host,ssh_port=port)
        try:ssh(state,'true',timeout=12);break
        except (subprocess.SubprocessError,OSError):time.sleep(8)
    else:raise RuntimeError('Diagnostic GPU SSH timeout')
    if not state.get('launched'):
        ssh(state,'mkdir -p /mnt/home_extra/sft')
        with tarfile.open(ROOT/'payload.tar.gz','w:gz') as tar:
            for name in ['cases.json','encoded.json','base_receipt.json','manifest.json','config.json','tokenizer']:tar.add(ROOT/name,arcname=name)
            tar.add(SOURCE/'pilot_runtime.py',arcname='pilot_runtime.py');tar.add(SOURCE/'contracts.py',arcname='contracts.py')
        copy_to(state,ROOT/'payload.tar.gz','/mnt/home_extra/sft/payload.tar.gz')
        copy_to(state,Path('/home/serda/cpt-production/scripts/cpt_gpu_preflight.py'),'/workspace/mamba-cpt-tr/scripts/cpt_gpu_preflight.py')
        ssh(state,'umask 077; cat > /mnt/home_extra/sft/hf_token',data=Path('/home/serda/.cache/huggingface/token').read_bytes())
        ssh(state,'cd /mnt/home_extra/sft && tar xzf payload.tar.gz && python -m pip install accelerate==1.10.1 > setup.log 2>&1 && python -c "import bitsandbytes,transformers"',timeout=180)
        # Write a status before launch so an import delay is not a monitoring error.
        ssh(state,"cd /mnt/home_extra/sft && python -c \"import json,time;from pathlib import Path;Path('status.json').write_text(json.dumps({'stage':'STARTING','updated_at':time.time()}))\" && (nohup env PYTHONPATH=/workspace/mamba-cpt-tr python -u pilot_runtime.py > training.log 2>&1 < /dev/null & echo $! > job.pid)")
        save(state,stage='RUNNING',launched=True,launched_at=time.time());journal('Tanı işi başlatıldı',{'arms':cfg['arms'],'train':32,'dev':128})
    errors=0;last=None
    while True:
        try:
            credit=vast('show','user')['credit'];hard,soft,spent=budget(cfg,state,credit,time.time());save(state,credit=credit,spent=spent)
            if hard:stop(state,'BUDGET_STOPPED_RETAINED','Diagnostic hard budget or deadline');return
            if soft and not state.get('stop_requested'):
                remote(state,'touch /mnt/home_extra/sft/STOP');save(state,stop_requested=True)
            result=json.loads(remote(state,'cat /mnt/home_extra/sft/status.json'));save(state,stage='RUNNING',remote_status=result)
            event=(result['stage'],result.get('arm'),result.get('step'),result.get('tag'))
            if event!=last:journal('Tanı ilerleme',result);last=event
            terminal=result['stage'] in ['DONE','FAILED']
            if terminal or time.time()-state.get('last_backup',0)>180:
                receipt=shared.previous.backup(state);save(state,last_backup=time.time(),log_receipt=receipt)
                api.upload_file(repo_id=cfg['output_repo'],path_in_repo='run/OPERATIONS_TR.md',path_or_fileobj=str(ROOT/'OPERATIONS_TR.md'))
            if terminal:
                assert receipt['verified']
                if result['stage']=='DONE':assert (ROOT/'pilot_result.json').is_file()
                vast('destroy','instance',state['instance_id'],'--yes');assert not any(i['id']==state['instance_id'] for i in vast('show','instances'))
                save(state,stage='COMPLETE',instance_deleted=True,diagnostic_status=result['stage'],diagnostic_complete=result.get('complete',False));journal('Tanı kayıtları doğrulandı ve GPU silindi',{'instance_id':state['instance_id'],'diagnostic_status':result['stage'],'complete':result.get('complete'),'spent':spent,'credit':credit})
                try:api.upload_file(repo_id=cfg['output_repo'],path_in_repo='run/OPERATIONS_TR.md',path_or_fileobj=str(ROOT/'OPERATIONS_TR.md'))
                except Exception as error:save(state,journal_upload_error=type(error).__name__)
                return
            health=json.loads(remote(state,"python -c \"import os,time,json;from pathlib import Path;r=Path('/mnt/home_extra/sft');os.kill(int((r/'job.pid').read_text()),0);print(json.dumps({'age':time.time()-(r/'training.log').stat().st_mtime}))\""))
            if health['age']>900:stop(state,'STALLED_RETAINED','No diagnostic log progress for 15 minutes');return
            errors=0;save(state,last_error=None,consecutive_errors=0)
        except (subprocess.SubprocessError,OSError,ValueError,RuntimeError,KeyError,AssertionError) as error:
            errors+=1;save(state,last_error=type(error).__name__,consecutive_errors=errors)
            if errors>=3:stop(state,'MONITOR_FAILED_RETAINED','Three monitor failures');return
        time.sleep(45)


if __name__=='__main__':
    if '--watchdog' in sys.argv:watchdog()
    else:
        try:main()
        except BaseException as error:
            state=json.loads(STATE.read_text()) if STATE.exists() else {}
            if state.get('instance_id'):stop(state,'FAILED_RETAINED',type(error).__name__)
            journal('Tanı hatası',{'error':type(error).__name__});raise
