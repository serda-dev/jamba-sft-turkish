"""One labelled A100, durable exports, cost guard, no idle paid preparation."""
import os,sys,time,json,hashlib,tarfile,shlex,subprocess
from pathlib import Path
ROOT=Path('/home/serda/sft-v2-run-20261015');PROJECT=Path('/home/serda/jamba-sft-turkish');STATE=ROOT/'supervisor.json';LABEL='linguai-sft-v2-20261015'
sys.path.insert(0,str(PROJECT))
# Reuse the already exercised export/hash implementation, with this run's root.
import supervisor as previous
previous.ROOT=ROOT;previous.STATE=STATE
import scripts.cpt_vps_supervisor as transport
transport.ROOT=ROOT
from scripts.cpt_vps_supervisor import vast,ssh,copy_to,IMAGE,SSH_KEY

def save(state,**fields):
 state.update(fields,updated_at=time.time());p=STATE.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(STATE);print(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),state.get('stage'),fields.get('remote_status',{}).get('stage',''),flush=True)
def stop(state,stage,reason):
 instance=next((i for i in vast('show','instances') if i['id']==state.get('instance_id')),None)
 if instance and instance.get('label')==LABEL:vast('stop','instance',instance['id'])
 save(state,stage=stage,reason=reason)

def retry_connection(action):
 # Only reads and idempotent log snapshots use retries; job launch is not retried.
 for attempt in range(3):
  try:return action()
  except subprocess.CalledProcessError as exc:
   if exc.returncode!=255 or attempt==2:raise
  except (subprocess.TimeoutExpired,ConnectionError):
   if attempt==2:raise
  time.sleep(3)

def main():
 state=json.loads(STATE.read_text()) if STATE.exists() else {'stage':'WAITING_FOR_DATA'}
 if state.get('stage')=='COMPLETE':return
 while not (ROOT/'data_audit.json').exists():
  save(state,stage='WAITING_FOR_DATA');time.sleep(20)
 cfg=json.loads((ROOT/'config.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text())
 audit=json.loads((ROOT/'data_audit.json').read_text());assert audit['verified'] and audit['manifest_sha256']==hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
 for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 from huggingface_hub import HfApi
 api=HfApi();api.create_repo(cfg['output_repo'],private=True,exist_ok=True)
 if not state.get('prepared_backup'):
  commit=api.upload_folder(repo_id=cfg['output_repo'],folder_path=str(ROOT),path_in_repo='prepared',allow_patterns=['*.json','*.jsonl.gz','tokenizer/*'],ignore_patterns=['supervisor.json','initial_credit.json'],commit_message='Pinned SFT v2 preparation and independent development probes')
  save(state,prepared_backup=commit.oid)
 if not state.get('instance_id'):
  existing=[i for i in vast('show','instances') if i.get('label')==LABEL]
  if existing:save(state,instance_id=existing[0]['id'])
  else:
   if state.get('rent_requested'):raise RuntimeError('Inspect uncertain previous rental before retry')
   credit=vast('show','user')['credit'];assert credit>=25
   offers=vast('search','offers','gpu_name=A100_SXM4 gpu_ram>=79 num_gpus=1 rentable=true rented=false reliability>0.98 verified=true','--storage',120,'--limit',30,'--order','dph_total')
   good=[o for o in offers if o['dph_total']<=1.35 and o['end_date']>time.time()+24*3600 and o.get('inet_up_cost',1)<=.005 and o.get('inet_down_cost',1)<=.005]
   if not good:raise RuntimeError('No qualifying A100 under $1.35/hour')
   offer=min(good,key=lambda o:o['dph_total']);save(state,stage='RENTING',rent_requested=True,initial_credit=credit,hourly_rate=offer['dph_total'],started_at=time.time(),offer_id=offer['id'])
   result=vast('create','instance',offer['id'],'--disk',120,'--image',IMAGE,'--ssh','--direct','--label',LABEL,'--env','-e AUTO_INSTALL=0 -e AUTO_TRAIN=0 -e HF_HOME=/mnt/home_extra/hf-cache','--onstart-cmd','bash /workspace/mamba-cpt-tr/scripts/vast_onstart.sh','--cancel-unavail')
   save(state,instance_id=result['new_contract'],stage='BOOTING')
 vast('attach','ssh',state['instance_id'],Path(SSH_KEY+'.pub').read_text().strip())
 for attempt in range(75):
  instance=next((i for i in vast('show','instances') if i['id']==state['instance_id']),None)
  if not instance:raise RuntimeError('GPU instance disappeared')
  assert instance['label']==LABEL
  host,port=transport.select_connection(state,instance);state.update(ssh_host=host,ssh_port=port)
  try:ssh(state,'true',timeout=12);break
  except (subprocess.SubprocessError,OSError):save(state,stage='BOOTING',boot_attempt=attempt);time.sleep(8)
 else:raise RuntimeError('GPU SSH boot timeout')
 save(state,stage='RUNNING' if state.get('launched') else 'CONNECTED')
 if not state.get('launched'):
  ssh(state,'mkdir -p /mnt/home_extra/sft')
  with tarfile.open(ROOT/'payload.tar.gz','w:gz') as tar:
   for name in ['train.jsonl.gz','validation.jsonl.gz','pilot.jsonl.gz','probes.json','manifest.json','config.json','tokenizer','error_analysis.json']:
    tar.add(ROOT/name,arcname=name)
   for name in ['train.py','final_eval.py','probe_score.py','pilot_recovery.py','memory_preflight.py','finalize_only.py']:tar.add(PROJECT/'sft_v2'/name,arcname=name)
   tar.add(PROJECT/'benchmarks',arcname='benchmark_code',filter=lambda info:None if '__pycache__' in info.name else info)
   tar.add(Path('/home/serda/benchmark-run-20261014/cases.jsonl'),arcname='benchmark_cases.jsonl')
   tar.add(Path('/home/serda/cpt-run-20261011/cache/token_cache/phase_1_validation'),arcname='cpt_validation')
   tar.add(Path('/home/serda/sft-run-20261013/eval_language_sft_final.json'),arcname='eval_language_v1.json')
  copy_to(state,ROOT/'payload.tar.gz','/mnt/home_extra/sft/payload.tar.gz')
  for directory in ['src','eval','customtokenizer']:copy_to(state,Path('/home/serda/cpt-production')/directory,'/workspace/mamba-cpt-tr/',recursive=True)
  copy_to(state,Path('/home/serda/cpt-production/scripts/cpt_gpu_preflight.py'),'/workspace/mamba-cpt-tr/scripts/cpt_gpu_preflight.py')
  copy_to(state,Path('/home/serda/cpt-production/scripts/cpt_evaluate.py'),'/workspace/mamba-cpt-tr/scripts/cpt_evaluate.py')
  ssh(state,'umask 077; cat > /mnt/home_extra/sft/hf_token',data=Path('/home/serda/.cache/huggingface/token').read_bytes())
  ssh(state,'cd /mnt/home_extra/sft && tar xzf payload.tar.gz && python -m pip install accelerate==1.10.1 langdetect==1.0.9 immutabledict nltk absl-py==2.3.1 > setup.log 2>&1',timeout=240)
  ssh(state,'cd /mnt/home_extra/sft && (nohup env PYTHONPATH=/workspace/mamba-cpt-tr python -u train.py > training.log 2>&1 < /dev/null & echo $! > job.pid)')
  save(state,stage='RUNNING',launched=True,launched_at=time.time())
 errors=0
 while True:
  try:
   credit=vast('show','user')['credit'];save(state,credit=credit)
   if credit<=cfg['hard_stop_credit']:stop(state,'BUDGET_STOPPED_RETAINED','Hard credit floor');return
   if credit<=cfg['stop_credit'] or time.time()-state['started_at']>cfg['max_hours']*3600:ssh(state,'touch /mnt/home_extra/sft/STOP');save(state,stop_requested=True)
   remote=json.loads(retry_connection(lambda:ssh(state,'cat /mnt/home_extra/sft/status.json')));save(state,remote_status=remote)
   logs=retry_connection(lambda:previous.backup(state))
   if remote['stage'] in ['DONE','STOPPED_EXPORTED']:
    receipt=previous.verify_final();assert logs['verified']
    # Training has exported a fully verified candidate; the published v1 is retained.
    vast('destroy','instance',state['instance_id'],'--yes')
    assert not any(i['id']==state['instance_id'] for i in vast('show','instances'))
    save(state,stage='COMPLETE',final_receipt=receipt,log_receipt=logs,instance_deleted=True);return
   if remote['stage'] in ['FAILED','PILOT_REJECTED']:
    stop(state,'FAILED_RETAINED',remote['stage']);return
   health=json.loads(retry_connection(lambda:ssh(state,"python -c \"import json,time,os;from pathlib import Path;r=Path('/mnt/home_extra/sft');p=int((r/'job.pid').read_text());os.kill(p,0);logs=[r/n for n in ['training.log','final-recovery.log'] if (r/n).exists()];print(json.dumps({'log_age':time.time()-max(p.stat().st_mtime for p in logs)}))\"")))
   if health['log_age']>1800:stop(state,'STALLED_RETAINED','No progress for 30 minutes');return
   errors=0;save(state,last_error=None,consecutive_errors=0)
  except (subprocess.SubprocessError,OSError,ValueError,RuntimeError,KeyError,AssertionError) as e:
   errors+=1;save(state,last_error=type(e).__name__,consecutive_errors=errors)
   if errors>=3:stop(state,'MONITOR_FAILED_RETAINED','Three consecutive monitor errors');return
  time.sleep(300)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  state=json.loads(STATE.read_text()) if STATE.exists() else {}
  if state.get('instance_id'):stop(state,'FAILED_RETAINED',type(error).__name__)
  raise
