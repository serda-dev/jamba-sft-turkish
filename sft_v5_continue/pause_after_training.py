"""Preserve the raw last model by pausing finalization only after update 1997."""
import json,time
from pathlib import Path
from review_policy import pause_ready
ROOT=Path('/mnt/home_extra/sft')
if (ROOT/'final_review_pause.json').exists():raise SystemExit(0)
while True:
 p=ROOT/'dev_1997.json'
 if p.exists():
  try:ready=pause_ready(json.loads(p.read_text()))
  except (OSError,ValueError):ready=False
  if ready:
   if (ROOT/'STOP').exists():break
   marker={'optimizer_training_complete':True,'step':1997,'requested_at':time.time(),'reason':'Preserve raw final model for development selection using direct conversation accuracy; no optimization steps interrupted.'}
   (ROOT/'final_review_pause.json').write_text(json.dumps(marker,indent=2));(ROOT/'STOP').touch();break
 time.sleep(2)
