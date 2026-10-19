"""Bounded diagnostic subsets from existing audited training artifacts, not new evaluation."""
import argparse
from collections import Counter, defaultdict
import json
from itertools import zip_longest
from pathlib import Path

from .common import rows, digest, write_json
from .sampling import balanced_schedule, TARGETS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--encoded',required=True)
    p.add_argument('--messages',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():
        raise SystemExit('New output directory required')
    a.output_dir.mkdir(parents=True,mode=0o700)
    pools=defaultdict(list)
    counts=Counter()
    metadata=[]
    selected_groups=set()
    precision=[]
    english_kinds=set()
    for encoded, message in zip_longest(rows(a.encoded),rows(a.messages)):
        if encoded is None or message is None:
            raise ValueError('Encoded/message row counts differ')
        if encoded['source'] != message['source'] or encoded['family'] != message['family']:
            raise ValueError('Encoded/message order mismatch')
        original=encoded['family']
        family='instruction' if original=='grounded_instruction' else original
        if family in TARGETS:
            metadata.append({'family':family,'target_tokens':sum(y!=-100 for y in encoded['labels'][1:])})
        else:
            counts['unclassified_'+family]+=1
        ms=message['messages']
        source=encoded['source']
        if len(precision)<128:
            precision.append(encoded)
        diagnostic=None
        if original=='math' and 'malhajar/' in source:
            diagnostic='math'
        elif original=='reasoning' and 'openthoughts_science' in source:
            diagnostic='reasoning'
        elif original=='instruction':
            diagnostic='instruction'
        elif original=='reading':
            diagnostic='reading'
        elif original=='code':
            diagnostic='code'
        elif original=='general' and sum(m['role']=='assistant' for m in ms)>=2 and 'ultrachat' not in source:
            diagnostic='dialogue'
        if diagnostic and len(pools[diagnostic])<3 and encoded['group'] not in selected_groups:
            pools[diagnostic].append({**encoded,'family':diagnostic})
            selected_groups.add(encoded['group'])
        english_kind = ('general' if 'ultrachat' in source else
                        'math' if 'openai/gsm8k' in source else
                        'reading' if original=='reading' and 'smoltalk' in source else None)
        if english_kind and english_kind not in english_kinds and encoded['group'] not in selected_groups:
            pools['english_retention'].append({**encoded,'family':'english_retention'})
            selected_groups.add(encoded['group'])
            english_kinds.add(english_kind)
    for name,data in [('precision',precision),('interference',[r for k in sorted(pools) for r in pools[k]])]:
        path=a.output_dir/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
        path.chmod(0o600)
    _,schedule=balanced_schedule(metadata,3_000_000,max_reuse=2)
    report={'encoded_sha256':digest(a.encoded),'messages_sha256':digest(a.messages),
            'files':{p.name:digest(p) for p in a.output_dir.glob('*.jsonl')},
            'gradient_family_counts':{k:len(v) for k,v in pools.items()},
            'gradient_scope':'Existing seen training examples; distinct stored groups, not proof of translation/source semantic disjointness. Dialogue selection is a multi-turn proxy requiring manual naturalness review.',
            'english_gradient_scope':{'included_task_families':sorted(english_kinds),'English_code_covered':False},
            'unclassified_rows':dict(counts),'v6_sample_exposure_dry_run':schedule,
            'ready_for_v6_training':False,
            'blockers':['No separately audited grounded_qa family','General reasoning is not verified mathematics',
                        'Language/source labels need independent review','Insufficient natural-family diversity at target token budget']}
    write_json(a.output_dir/'manifest.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
