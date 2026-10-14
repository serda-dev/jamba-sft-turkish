"""One bounded evaluation rental, verified output preservation and cleanup."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT=Path('/home/serda/benchmark-run-20261014')
PROJECT=Path('/home/serda/jamba-sft-turkish/benchmarks')
STATE=ROOT/'supervisor.json';LABEL='linguai-benchmark-20261014'
os.environ['LINGUAI_CPT_RUN_ROOT']=str(ROOT)
sys.path.insert(0,'/home/serda/cpt-production')
from scripts.cpt_vps_supervisor import vast,ssh,copy_to,IMAGE,SSH_KEY


def save(state,**fields):
    state.update(fields,time=time.time());tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATE)
    print(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),state.get('stage'),flush=True)


def collect(state):
    code="""import base64,json
from pathlib import Path
r=Path('/mnt/home_extra/benchmark')
live=[p for pattern in ['*.json','*.jsonl','*.log'] for p in r.glob(pattern) if p.is_file()]
snapshot=[p for p in (r/'export').glob('*') if p.is_file()]
assert sum(p.stat().st_size for p in live+snapshot)<128*1024**2
print(json.dumps({'live':{p.name:base64.b64encode(p.read_bytes()).decode() for p in live},
 'snapshot':{p.name:base64.b64encode(p.read_bytes()).decode() for p in snapshot}}))
"""
    import base64
    data=json.loads(ssh(state,'python -c '+shlex.quote(code),timeout=90))
    (ROOT/'export').mkdir(exist_ok=True)
    for kind,files in data.items():
        destination=ROOT/'export' if kind=='snapshot' else ROOT
        for name,text in files.items():
            assert Path(name).name==name
            (destination/name).write_bytes(base64.b64decode(text,validate=True))


def verify_export():
    from huggingface_hub import HfApi
    r=json.loads((ROOT/'upload_receipt.json').read_text());assert r['verified']
    items={s.rfilename:s for s in HfApi().model_info(r['repo'],revision=r['revision'],files_metadata=True).siblings}
    for name,digest in r['sha256'].items():
        file=ROOT/'export'/name;data=file.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
        item=items[r['path']+'/'+name]
        actual=(item.lfs.sha256 if hasattr(item.lfs,'sha256') else item.lfs['sha256']) if item.lfs else item.blob_id
        wanted=digest if item.lfs else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert actual==wanted and item.size==len(data)
    return r


def main():
    state=json.loads(STATE.read_text()) if STATE.exists() else {'stage':'PREPARED'}
    if state.get('stage') in {'COMPLETE','STOPPED_EXPORTED'}:return
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert hashlib.sha256((ROOT/'cases.jsonl').read_bytes()).hexdigest()==manifest['case_sha256']
    if not state.get('instance_id'):
        existing=[i for i in vast('show','instances') if i.get('label')==LABEL]
        if existing:save(state,instance_id=existing[0]['id'])
        else:
            if state.get('rent_requested'):raise RuntimeError('Uncertain prior rent; inspect account before retry')
            credit=vast('show','user')['credit'];assert credit>=20
            offers=vast('search','offers','gpu_name=RTX_4090 gpu_ram>=23 gpu_ram<=26 num_gpus=1 rentable=true rented=false reliability>0.98 verified=true',
                '--storage',100,'--limit',100,'--order','dph_total')
            offers+=vast('search','offers','gpu_name=A100_SXM4 gpu_ram>=39 num_gpus=1 rentable=true rented=false reliability>0.98 verified=true',
                '--storage',100,'--limit',30,'--order','dph_total')
            choices=[o for o in offers if o['dph_total']<=.85 and o['cpu_ram']>=32000 and o['cpu_cores_effective']>=8
                     and o['id']!=53491151 and o.get('geolocation','').split(',')[-1].strip() in {'TR','US','CA','GB','DE','ES','FR','NL','PL','FI','SE','RO','IT'}
                     and o['end_date']>time.time()+24*3600 and o.get('inet_down_cost',1)<=.01 and o.get('inet_up_cost',1)<=.01]
            if not choices:raise RuntimeError('No qualified inference GPU within price limit')
            offer=min(choices,key=lambda o:o['dph_total'])
            save(state,stage='RENTING',rent_requested=True,initial_credit=credit,started_at=time.time(),hourly_rate=offer['dph_total'],offer_id=offer['id'])
            rented=vast('create','instance',offer['id'],'--disk',100,'--image',IMAGE,'--ssh','--direct','--label',LABEL,
                '--env','-e AUTO_INSTALL=0 -e AUTO_TRAIN=0 -e HF_HOME=/mnt/home_extra/hf-cache -e TRANSFORMERS_CACHE=/mnt/home_extra/hf-cache/hub',
                '--onstart-cmd','bash /workspace/mamba-cpt-tr/scripts/vast_onstart.sh','--cancel-unavail')
            save(state,instance_id=rented['new_contract'],stage='BOOTING')
    vast('attach','ssh',state['instance_id'],Path(SSH_KEY+'.pub').read_text().strip())
    for _ in range(120):
        item=next((i for i in vast('show','instances') if i['id']==state['instance_id']),None)
        if not item:raise RuntimeError('Evaluation instance missing')
        assert item['label']==LABEL
        ports=(item.get('ports') or {}).get('22/tcp',[])
        state.update(ssh_host=item['public_ipaddr'] if ports else item['ssh_host'],ssh_port=int(ports[0]['HostPort']) if ports else item['ssh_port'])
        try:ssh(state,'true',timeout=20);break
        except (subprocess.SubprocessError,OSError):time.sleep(10)
    else:raise RuntimeError('GPU did not become reachable')
    save(state)
    if not state.get('launched'):
        ssh(state,'mkdir -p /mnt/home_extra/benchmark')
        with tarfile.open(ROOT/'payload.tar.gz','w:gz') as tar:
            for name in ['cases.jsonl','manifest.json']:tar.add(ROOT/name,arcname=name)
            for name in ['run.py','scoring.py','numeric_score.py','instruction_following_eval']:tar.add(PROJECT/name,arcname=name)
        copy_to(state,ROOT/'payload.tar.gz','/mnt/home_extra/benchmark/payload.tar.gz')
        copy_to(state,Path('/home/serda/cpt-production/scripts/cpt_gpu_preflight.py'),'/workspace/mamba-cpt-tr/scripts/cpt_gpu_preflight.py')
        ssh(state,'umask 077; cat > /mnt/home_extra/benchmark/hf_token',data=Path('/home/serda/.cache/huggingface/token').read_bytes())
        ssh(state,'cd /mnt/home_extra/benchmark && tar xzf payload.tar.gz && python -m pip install absl-py langdetect immutabledict nltk > setup.log 2>&1',timeout=180)
        ssh(state,'cd /mnt/home_extra/benchmark && (nohup env PYTHONPATH=/workspace/mamba-cpt-tr python -u run.py > benchmark.log 2>&1 < /dev/null & echo $! > job.pid)')
        save(state,stage='RUNNING',launched=True)
    while True:
        try:
            credit=vast('show','user')['credit'];save(state,credit=credit)
            if credit<=20 or state['initial_credit']-credit>=10 or time.time()-state['started_at']>16*3600:
                ssh(state,'touch /mnt/home_extra/benchmark/STOP');save(state,stop_requested=True)
            remote=json.loads(ssh(state,'cat /mnt/home_extra/benchmark/status.json'));save(state,remote_status=remote)
            collect(state)
            if remote['stage'] in ['DONE','STOPPED']:
                if remote['stage']=='DONE' and not remote.get('verified'):
                    time.sleep(5);continue
                receipt=verify_export()
                if remote['stage']=='DONE':
                    results=json.loads((ROOT/'results.json').read_text())['results']
                    from collections import Counter
                    expected=Counter(json.loads(line)['task'] for line in (ROOT/'cases.jsonl').read_text().splitlines())
                    assert len(results)==len(manifest['models'])
                    for model in results.values():
                        assert all(model[task]['n']==n for task,n in expected.items())
                vast('destroy','instance',state['instance_id'],'--yes')
                save(state,stage='COMPLETE' if remote['stage']=='DONE' else 'STOPPED_EXPORTED',export_receipt=receipt);return
            if remote['stage']=='FAILED':vast('stop','instance',state['instance_id']);save(state,stage='FAILED_RETAINED');return
            code="""import os,time,json
from pathlib import Path
r=Path('/mnt/home_extra/benchmark')
try:os.kill(int((r/'job.pid').read_text()),0);alive=True
except ProcessLookupError:alive=False
print(json.dumps({'alive':alive,'log_age':time.time()-(r/'benchmark.log').stat().st_mtime}))
"""
            health=json.loads(ssh(state,'python -c '+shlex.quote(code)))
            if not health['alive'] or health['log_age']>1800:
                vast('stop','instance',state['instance_id']);save(state,stage='FAILED_RETAINED',health=health);return
        except (OSError,subprocess.SubprocessError,ValueError,RuntimeError,AssertionError,KeyError) as error:
            save(state,error=type(error).__name__)
            if state.get('credit',99)<=18:
                vast('stop','instance',state['instance_id']);save(state,stage='EXPORT_BLOCKED_RETAINED');return
        time.sleep(300)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        state=json.loads(STATE.read_text()) if STATE.exists() else {}
        if state.get('instance_id') and state.get('stage')!='COMPLETE':
            instance=next((i for i in vast('show','instances') if i['id']==state['instance_id']),None)
            if instance and instance.get('label')==LABEL:vast('stop','instance',state['instance_id'])
            save(state,stage='FAILED_RETAINED',error=type(error).__name__)
        raise
