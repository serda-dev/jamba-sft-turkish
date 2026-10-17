"""Independent timer stops this rental if the main monitor dies or hits its cap."""
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sft_v3.supervisor import ROOT,STATE,LABEL,vast,budget,stop


def main():
    if not STATE.exists():return
    state=json.loads(STATE.read_text())
    if state.get('stage')=='COMPLETE':return
    active=subprocess.run(['systemctl','--user','is-active','--quiet','linguai-sft-v3-supervisor.service']).returncode==0
    if not state.get('instance_id'):
        # Recover a rental accepted just before a CLI/process disconnect.
        if not state.get('rent_requested') or active:return
        matches=[i for i in vast('show','instances') if i.get('label')==LABEL]
        if len(matches)!=1:return
        state['instance_id']=matches[0]['id']
    items=vast('show','instances');item=next((i for i in items if i['id']==state['instance_id']),None)
    if not item or item['label']!=LABEL or item.get('intended_status')=='stopped':return
    cfg=json.loads((ROOT/'config.json').read_text());credit=vast('show','user')['credit']
    hard,_,_=budget(cfg,state,credit,time.time())
    if hard or not active:stop(state,'WATCHDOG_STOPPED_RETAINED','Budget/time cap or main monitor inactive')


if __name__=='__main__':main()
