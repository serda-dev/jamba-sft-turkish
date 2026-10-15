import json,re,ast
from pathlib import Path
from collections import Counter,defaultdict
root=Path('/home/serda/benchmark-run-20261014');out=Path('/home/serda/sft-v2-run-20261015');report={}
for ident in ['jamba_sft','jamba_cpt','jamba_original']:
 rows=[json.loads(l) for l in (root/f'responses_{ident}.jsonl').read_text().splitlines()];tasks={}
 for task in sorted(set(r['task'] for r in rows)):
  rr=[r for r in rows if r['task']==task];s={'n':len(rr),'correct':sum(r['correct'] for r in rr),'truncated':sum(r.get('truncated',False) for r in rr)}
  if rr[0]['type']=='mcq':s.update(predicted_choices=dict(Counter(chr(65+r['prediction']) for r in rr)),gold_choices=dict(Counter(chr(65+r['gold']) for r in rr)))
  else:
   repeated=0;echo=0
   for r in rr:
    words=r['response'].split();grams=[tuple(words[i:i+8]) for i in range(len(words)-7)]
    repeated+=bool(grams and 1-len(set(grams))/len(grams)>.2)
    echo+=r['response'].startswith('Soru:')
   s.update(repetitive_8gram_fraction_gt_20pct=repeated,starts_with_question_label=echo)
  if task=='ifeval_tr_subset':
   checks=defaultdict(lambda:[0,0])
   for r in rr:
    ids=ast.literal_eval(r['rules']['instruction_id_list'])
    for rule,ok in zip(ids,r['checks']):checks[rule][0]+=ok;checks[rule][1]+=1
   s['rule_success_and_total']=dict(checks)
  tasks[task]=s
 report[ident]=tasks
(out/'error_analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['jamba_sft'],indent=2))
