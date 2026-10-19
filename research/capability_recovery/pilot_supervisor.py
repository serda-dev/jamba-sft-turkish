"""One authorized $2/one-hour diagnostic; independent watchdog, backup and deletion."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

from .common import digest, write_json

ROOT=Path('/home/serda/capability-pilot-20261019')
PROJECT=Path('/home/serda/jamba-sft-turkish')
LABEL='linguai-capability-pilot-20261019'
UNIT='linguai-capability-pilot'
sys.path.insert(0,'/home/serda/cpt-production')
from scripts import cpt_vps_supervisor as transport
transport.ROOT=ROOT
vast=transport.vast


def rent(*args):
    key=transport.KEY.read_text().strip()
    result=subprocess.run(['/home/serda/.local/bin/vastai','--api-key',key,'--raw',*map(str,args)],
                          capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError('Rental CLI failed; see sanitized response')
    reply=json.loads(result.stdout or result.stderr)
    write_json(ROOT/'rent_response.json',{'returncode':result.returncode,
               'response':{k:reply[k] for k in ['success','new_contract','error','status_code','msg'] if k in reply}})
    if not reply.get('success') or not reply.get('new_contract'):raise RuntimeError('Rental rejected')
    return reply


def save(**updates):
    with (ROOT/'state.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((ROOT/'supervisor.json').read_text()) if (ROOT/'supervisor.json').exists() else {}
        state.update(updates,updated_at=time.time())
        write_json(ROOT/'supervisor.json',state)
    return state


def journal(event, details):
    with (ROOT/'OPERATIONS_TR.md').open('a') as f:
        f.write('\n### '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+' — '+event+'\n\n```json\n'+json.dumps(details,ensure_ascii=False,indent=2)+'\n```\n')


def limits(state,credit,now=None):
    now=time.time() if now is None else now
    elapsed=now-state['started_at']
    observed=max(0,state['initial_credit']-credit)
    conservative=max(observed,elapsed/3600*state['hourly_rate']+.30)
    return {'elapsed_seconds':elapsed,'observed_spent':observed,
            'conservative_spent':conservative,'hard':elapsed>=3300 or conservative>=1.80}


def verify_hub_receipt(receipt):
    from huggingface_hub import HfApi
    assert receipt['verified']
    items={s.rfilename:s for s in HfApi().model_info(receipt['repo_id'],revision=receipt['revision'],files_metadata=True).siblings}
    for name,meta in receipt['files'].items():
        actual=items[receipt['path']+'/'+name]
        actual_hash=actual.lfs.sha256 if actual.lfs else actual.blob_id
        if actual.size!=meta['size'] or actual_hash!=meta['sha256' if actual.lfs else 'git_sha1']:
            raise ValueError('Independent checkpoint backup verification failed')


def backup_logs(state):
    from huggingface_hub import HfApi
    cfg=json.loads((ROOT/'config.json').read_text())
    command="cd /mnt/home_extra/capability && python - <<'PY'\nfrom pathlib import Path\nimport tarfile\nr=Path('.')\nwith tarfile.open('logs.tar.gz','w:gz') as t:\n for p in r.iterdir():\n  if p.is_file() and p.suffix in ['.json','.log']:t.add(p,arcname=p.name)\nPY"
    transport.ssh(state,command,timeout=30)
    path=ROOT/'logs.tar.gz'
    with path.open('wb') as out:
        subprocess.run(['ssh',*transport.connection(state),'root@'+state['ssh_host'],
                        'cat /mnt/home_extra/capability/logs.tar.gz'],stdout=out,stderr=subprocess.PIPE,
                       check=True,timeout=90)
    remote_hash=transport.ssh(state,'sha256sum /mnt/home_extra/capability/logs.tar.gz',timeout=20).split()[0]
    if digest(path)!=remote_hash:
        raise ValueError('Log transfer hash mismatch')
    destination=ROOT/'results'
    destination.mkdir(exist_ok=True)
    with tarfile.open(path,'r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile() or Path(member.name).name!=member.name:
                raise ValueError('Unexpected log archive member')
            (destination/member.name).write_bytes(archive.extractfile(member).read())
    for name in ['precision_existing_receipt.json','precision_fp32_micro_receipt.json']:
        if (destination/name).exists():verify_hub_receipt(json.loads((destination/name).read_text()))
    api=HfApi()
    commit=api.upload_file(repo_id=cfg['output_repo'],path_in_repo='run/logs.tar.gz',path_or_fileobj=str(path))
    item=next(s for s in api.model_info(cfg['output_repo'],revision=commit.oid,files_metadata=True).siblings if s.rfilename=='run/logs.tar.gz')
    expected=remote_hash if item.lfs else __import__('hashlib').sha1(b'blob '+str(path.stat().st_size).encode()+b'\0'+path.read_bytes()).hexdigest()
    if (item.lfs.sha256 if item.lfs else item.blob_id)!=expected or item.size!=path.stat().st_size:
        raise ValueError('Hub log receipt mismatch')
    receipt={'repo':cfg['output_repo'],'revision':commit.oid,'sha256':remote_hash,'verified':True}
    write_json(ROOT/'log_receipt.json',receipt)
    return receipt


def retire(state,reason,skip_backup=False):
    item=next((i for i in vast('show','instances') if i['id']==state.get('instance_id')),None)
    if not item:
        save(stage='COMPLETE',instance_deleted=True,reason=reason)
        return
    if item.get('label')!=LABEL:raise ValueError('Refusing to touch a different instance')
    try:
        receipt=backup_logs(state) if state.get('launched') and not skip_backup else None
        save(log_receipt=receipt)
    except Exception as error:
        save(backup_error=type(error).__name__)
        journal('Kapanışta log yedekleme hatası',{'error':type(error).__name__})
    # Source checkpoints are immutable. Disposable diagnostic weights, when produced,
    # are backed up per arm; hard spending limits take precedence over retrying forever.
    vast('destroy','instance',state['instance_id'])
    if any(i['id']==state['instance_id'] for i in vast('show','instances')):
        raise RuntimeError('GPU deletion not verified')
    credit=vast('show','user')['credit']
    save(stage='COMPLETE',instance_deleted=True,reason=reason,final_credit=credit,
         observed_spent=max(0,state['initial_credit']-credit))
    journal('Pilot GPU silindi',{'instance_id':state['instance_id'],'reason':reason,'credit':credit})


def watchdog():
    if not (ROOT/'supervisor.json').exists():return
    state=json.loads((ROOT/'supervisor.json').read_text())
    if state.get('instance_deleted') or not state.get('started_at'):return
    if not state.get('instance_id'):
        own=[i for i in vast('show','instances') if i.get('label')==LABEL]
        if not own:return
        if len(own)!=1:raise RuntimeError('Ambiguous pilot instances')
        state=save(instance_id=own[0]['id'])
    credit=vast('show','user')['credit']
    active=subprocess.run(['systemctl','--user','is-active','--quiet',UNIT+'.service']).returncode==0
    if limits(state,credit)['hard'] or not active:
        retire(state,'WATCHDOG_LIMIT_OR_MONITOR_INACTIVE',skip_backup=True)


def main():
    from huggingface_hub import HfApi
    state=save()
    if state.get('instance_deleted'):return
    cfg=json.loads((ROOT/'config.json').read_text())
    assert cfg['max_cost_usd']==2 and cfg['max_seconds']==3600 and cfg['user_authorized']
    api=HfApi()
    api.create_repo(cfg['output_repo'],private=True,exist_ok=True)
    assert api.model_info(cfg['output_repo']).private
    if not state.get('instance_id'):
        if state.get('rent_requested'):raise RuntimeError('Prior rental uncertain; inspect by label')
        offer=json.loads((ROOT/'selected_offer.json').read_text())
        assert offer['gpu_name']=='A100 SXM4' and offer['dph_total']<=1.35
        assert offer['dph_total']+25*(offer['inet_up_cost']+offer['inet_down_cost'])<1.80
        credit=vast('show','user')['credit']
        assert credit>=3
        state=save(stage='RENTING',rent_requested=True,initial_credit=credit,started_at=time.time(),
                   hourly_rate=offer['dph_total'],offer_id=offer['id'])
        result=rent('create','instance',offer['id'],'--disk',35,'--image',transport.IMAGE,
                    '--ssh','--direct','--label',LABEL,'--env',
                    '-e AUTO_INSTALL=0 -e AUTO_TRAIN=0 -e HF_HOME=/mnt/home_extra/hf-cache',
                    '--onstart-cmd','bash /workspace/mamba-cpt-tr/scripts/vast_onstart.sh','--cancel-unavail')
        state=save(stage='BOOTING',instance_id=result['new_contract'])
        journal('Yetkili pilot kiralandı',{'instance_id':state['instance_id'],'hourly_rate':offer['dph_total'],'max_cost_usd':2,'max_seconds':3600})
    vast('attach','ssh',state['instance_id'],Path(transport.SSH_KEY+'.pub').read_text().strip())
    boot_start=time.time()
    while time.time()-boot_start<600:
        if limits(state,vast('show','user')['credit'])['hard']:raise RuntimeError('Budget/time limit')
        item=next(i for i in vast('show','instances') if i['id']==state['instance_id'])
        host,port=transport.select_connection(state,item)
        state=save(ssh_host=host,ssh_port=port)
        try:
            transport.ssh(state,'true',timeout=12)
            break
        except (subprocess.SubprocessError,OSError):time.sleep(8)
    else:raise RuntimeError('Boot timeout')
    transport.ssh(state,'mkdir -p /mnt/home_extra/capability')
    transport.copy_to(state,ROOT/'payload.tar.gz','/mnt/home_extra/capability/payload.tar.gz')
    transport.ssh(state,'umask 077; cat > /mnt/home_extra/capability/hf_token',data=Path('/home/serda/.cache/huggingface/token').read_bytes())
    transport.ssh(state,"cd /mnt/home_extra/capability && tar xzf payload.tar.gz && chmod 600 hf_token && (nohup env PYTHONPATH=/mnt/home_extra/capability:/workspace/mamba-cpt-tr python -u -m research.capability_recovery.pilot_job > job.log 2>&1 < /dev/null & echo $! > job.pid)")
    state=save(stage='RUNNING',launched=True)
    while True:
        credit=vast('show','user')['credit']
        usage=limits(state,credit)
        save(credit=credit,usage=usage)
        if usage['hard']:
            retire(state,'BUDGET_OR_TIME_LIMIT')
            return
        try:
            report=json.loads(transport.ssh(state,'cat /mnt/home_extra/capability/status.json',timeout=20))
            save(remote_status=report)
            if report['stage'] in ['DONE','FAILED']:
                retire(state,report['stage'])
                return
        except (subprocess.SubprocessError,ValueError):
            if time.time()-state['updated_at']>300:raise
        time.sleep(20)


if __name__=='__main__':
    if '--watchdog' in sys.argv:
        watchdog()
    else:
        try:main()
        except BaseException as error:
            state=save(error_type=type(error).__name__)
            journal('Pilot denetleyici hatası',{'error_type':type(error).__name__})
            if state.get('instance_id'):retire(state,'SUPERVISOR_ERROR')
            raise
