"""Prepare before rental; monitor credit; verify model and logs before deletion."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT=Path('/home/serda/sft-run-20261013')
PROJECT=Path('/home/serda/jamba-sft-turkish')
os.environ['LINGUAI_CPT_RUN_ROOT']=str(ROOT)
sys.path.insert(0,'/home/serda/cpt-production')
from scripts.cpt_vps_supervisor import vast,ssh,copy_to,IMAGE,SSH_KEY

LABEL='linguai-sft-20261013'
STATE=ROOT/'supervisor.json'


def save(state,**fields):
    state.update(fields,updated_at=time.time())
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)
    print(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),state.get('stage'),flush=True)


def backup(state):
    from huggingface_hub import HfApi
    code="""import tarfile
from pathlib import Path
r=Path('/mnt/home_extra/sft')
with tarfile.open(r/'logs.tar.gz','w:gz') as t:
 for pattern in ['*.log','*.json','*.jsonl']:
  for p in r.glob(pattern):
   if p.is_file() and not p.is_symlink():t.add(p,arcname=p.name)
"""
    ssh(state,'python -c '+shlex.quote(code),timeout=60)
    options=['-i',SSH_KEY,'-P',str(state['ssh_port']),'-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','StrictHostKeyChecking=accept-new','-o',f'UserKnownHostsFile={ROOT}/known_hosts']
    subprocess.run(['scp',*options,f'root@{state["ssh_host"]}:/mnt/home_extra/sft/logs.tar.gz',str(ROOT/'logs.tar.gz')],
        capture_output=True,check=True,timeout=120)
    # Keep VPS preparation/monitoring evidence alongside GPU logs, without keys.
    archive=ROOT/'logs.tar.gz'
    with tarfile.open(archive) as old:
        members={m.name:old.extractfile(m).read() for m in old.getmembers()
                 if m.isfile() and Path(m.name).name==m.name}
    import io
    temporary=ROOT/'logs.tmp'
    with tarfile.open(temporary,'w:gz') as tar:
        for name,data in members.items():
            item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
        for name in ['prepare.log','supervisor.log','source-research.json','sample-review.json']:
            if (ROOT/name).is_file():tar.add(ROOT/name,arcname='vps-'+name)
    temporary.replace(archive)
    cfg=json.loads((ROOT/'config.json').read_text());api=HfApi()
    archive=ROOT/'logs.tar.gz';data=archive.read_bytes();path='run/logs.tar.gz'
    commit=api.upload_file(repo_id=cfg['output_repo'],path_or_fileobj=str(archive),path_in_repo=path)
    item=next(f for f in api.model_info(cfg['output_repo'],revision=commit.oid,files_metadata=True).siblings if f.rfilename==path)
    if item.lfs:
        actual=item.lfs.sha256 if hasattr(item.lfs,'sha256') else item.lfs['sha256'];expected=hashlib.sha256(data).hexdigest()
    else:actual=item.blob_id;expected=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert item.size==len(data) and actual==expected
    receipt={'repo':cfg['output_repo'],'revision':commit.oid,'verified':True,'sha256':hashlib.sha256(data).hexdigest()}
    (ROOT/'log_receipt.json').write_text(json.dumps(receipt,indent=2))
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            if member.isfile() and Path(member.name).name==member.name:
                (ROOT/member.name).write_bytes(t.extractfile(member).read())
    return receipt


def verify_final():
    from huggingface_hub import HfApi
    r=json.loads((ROOT/'final_receipt.json').read_text())
    assert r['verified']
    remote={s.rfilename:s for s in HfApi().model_info(r['repo_id'],revision=r['revision'],files_metadata=True).siblings}
    assert any(name.endswith('.safetensors') for name in r['files'])
    for name,wanted in r['files'].items():
        item=remote[r['path']+'/'+name]
        actual=(item.lfs.sha256 if hasattr(item.lfs,'sha256') else item.lfs['sha256']) if item.lfs else item.blob_id
        assert item.size==wanted['size'] and actual==wanted['sha256' if item.lfs else 'git_sha1']
    return r


def main():
    ROOT.mkdir(exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'stage':'PREPARED'}
    if state.get('stage')=='COMPLETE':return
    # Preparation is validated by the operator before enabling this service.
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for file,expected in manifest['files'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==expected
    from huggingface_hub import HfApi
    cfg=json.loads((ROOT/'config.json').read_text());HfApi().create_repo(cfg['output_repo'],private=True,exist_ok=True)
    if not state.get('instance_id'):
        instances=vast('show','instances')
        existing=[i for i in instances if i.get('label')==LABEL]
        if existing:save(state,instance_id=existing[0]['id'])
        else:
            if state.get('rent_requested'):raise RuntimeError('Uncertain earlier rent; inspect account before retry')
            credit=vast('show','user')['credit'];assert credit>=30
            offers=vast('search','offers','gpu_name=A100_SXM4 gpu_ram>=79 num_gpus=1 rentable=true rented=false reliability>0.98 verified=true',
                        '--storage',120,'--limit',20,'--order','dph_total')
            choices=[o for o in offers if o['dph_total']<=1.25 and o['end_date']>time.time()+30*3600
                     and o.get('inet_up_cost',1)<=.01 and o.get('inet_down_cost',1)<=.01]
            if not choices:raise RuntimeError('No qualifying A100 within budget')
            offer=min(choices,key=lambda x:x['dph_total'])
            save(state,stage='RENTING',rent_requested=True,initial_credit=credit,hourly_rate=offer['dph_total'],started_at=time.time())
            rented=vast('create','instance',offer['id'],'--disk',120,'--image',IMAGE,'--ssh','--direct','--label',LABEL,
                '--env','-e AUTO_INSTALL=0 -e AUTO_TRAIN=0 -e HF_HOME=/mnt/home_extra/hf-cache -e TRANSFORMERS_CACHE=/mnt/home_extra/hf-cache/hub',
                '--onstart-cmd','bash /workspace/mamba-cpt-tr/scripts/vast_onstart.sh','--cancel-unavail')
            save(state,instance_id=rented['new_contract'],stage='BOOTING')
    vast('attach','ssh',state['instance_id'],Path(SSH_KEY+'.pub').read_text().strip())
    for _ in range(120):
        instance=next((i for i in vast('show','instances') if i['id']==state['instance_id']),None)
        if not instance:raise RuntimeError('Run instance no longer exists')
        assert instance['label']==LABEL
        ports=(instance.get('ports') or {}).get('22/tcp',[])
        state.update(ssh_host=instance.get('public_ipaddr') if ports else instance['ssh_host'],
                     ssh_port=int(ports[0]['HostPort']) if ports else instance['ssh_port'])
        try:ssh(state,'true',timeout=20);break
        except (subprocess.SubprocessError,OSError):time.sleep(10)
    else:raise RuntimeError('GPU SSH never became ready')
    save(state)
    if not state.get('launched'):
        ssh(state,'mkdir -p /mnt/home_extra/sft',timeout=30)
        with tarfile.open(ROOT/'payload.tar.gz','w:gz') as tar:
            for name in ['train.jsonl.gz','validation.jsonl.gz','tokenizer','config.json','manifest.json','eval_cases.json']:
                tar.add(ROOT/name,arcname=name)
            tar.add(PROJECT/'train.py',arcname='train.py')
            tar.add(Path('/home/serda/cpt-run-20261011/cache/token_cache/phase_1_validation'),arcname='cpt_validation')
            tar.add(Path('/home/serda/cpt-run-20261011/log-backup/files/gpu/eval_final.json'),arcname='eval_language_cpt_baseline.json')
        copy_to(state,ROOT/'payload.tar.gz','/mnt/home_extra/sft/payload.tar.gz')
        for directory in ['src','eval','customtokenizer']:
            copy_to(state,Path('/home/serda/cpt-production')/directory,'/workspace/mamba-cpt-tr/',recursive=True)
        # The token is transmitted on stdin, never included in an argv or log archive.
        ssh(state,'umask 077; cat > /mnt/home_extra/sft/hf_token',
            data=Path('/home/serda/.cache/huggingface/token').read_bytes())
        copy_to(state,Path('/home/serda/cpt-production/scripts/cpt_gpu_preflight.py'),'/workspace/mamba-cpt-tr/scripts/cpt_gpu_preflight.py')
        copy_to(state,Path('/home/serda/cpt-production/scripts/cpt_evaluate.py'),'/workspace/mamba-cpt-tr/scripts/cpt_evaluate.py')
        ssh(state,'cd /mnt/home_extra/sft && tar xzf payload.tar.gz && python -m pip install accelerate==1.10.1 > setup.log 2>&1',timeout=180)
        command='cd /mnt/home_extra/sft && (nohup env PYTHONPATH=/workspace/mamba-cpt-tr python -u train.py > training.log 2>&1 < /dev/null & echo $! > job.pid)'
        ssh(state,command,timeout=30)
        save(state,launched=True,stage='RUNNING')
    while True:
        try:
            credit=vast('show','user')['credit'];save(state,credit=credit)
            if credit<=12 or time.time()-state['started_at']>20*3600:
                ssh(state,'touch /mnt/home_extra/sft/STOP');save(state,stop_requested=True)
            remote=json.loads(ssh(state,'cat /mnt/home_extra/sft/status.json',timeout=45))
            save(state,remote_status=remote)
            log_receipt=backup(state)
            if remote['stage'] in ['DONE','STOPPED_EXPORTED']:
                receipt=verify_final()
                assert log_receipt['verified']
                vast('destroy','instance',state['instance_id'])
                save(state,stage='COMPLETE',final_receipt=receipt,log_receipt=log_receipt);return
            if remote['stage']=='FAILED':
                vast('stop','instance',state['instance_id']);save(state,stage='FAILED_RETAINED');return
            code="""import json,os,time
from pathlib import Path
r=Path('/mnt/home_extra/sft')
try:os.kill(int((r/'job.pid').read_text()),0);alive=True
except (ProcessLookupError,FileNotFoundError):alive=False
log_time=max(p.stat().st_mtime for p in [r/'training.log',r/'finalize.log'] if p.exists())
print(json.dumps({'alive':alive,'log_age':time.time()-log_time}))
"""
            health=json.loads(ssh(state,'python -c '+shlex.quote(code)))
            if not health['alive'] or health['log_age']>1800:
                vast('stop','instance',state['instance_id']);save(state,stage='FAILED_RETAINED',health=health);return
        except (subprocess.SubprocessError,OSError,ValueError,RuntimeError,KeyError,AssertionError) as error:
            save(state,error=type(error).__name__)
            if state.get('credit',99)<=7:
                vast('stop','instance',state['instance_id']);save(state,stage='EXPORT_BLOCKED_RETAINED');return
        time.sleep(600)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        state=json.loads(STATE.read_text()) if STATE.exists() else {}
        if state.get('instance_id') and state.get('stage')!='COMPLETE':
            instance=next((i for i in vast('show','instances') if i['id']==state['instance_id']),None)
            if instance and instance.get('label')==LABEL:vast('stop','instance',state['instance_id'])
            save(state,stage='FAILED_RETAINED',error=type(error).__name__)
        raise
