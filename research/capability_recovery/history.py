"""Reanalyse saved historical responses without selecting models or doing inference."""
import json
from pathlib import Path
from .common import digest, rows, write_json
from .evaluation import paired_difference, repetition, wilson


def main():
    report = {'status':'SAVED_HISTORICAL_OUTPUTS_ONLY','models':{},'paired_v2_v5':{}}
    data = {}
    for name, root in [('v2','sft-v2-run-20261015'),('v4b','sft-v4-run-20261018'),
                       ('v5','sft-v5-cont-run-20261018')]:
        path = Path('/home/serda')/root/'final_benchmark.jsonl'
        data[name] = list(rows(path))
        summaries = {}
        for family in sorted({r['task'] for r in data[name]}):
            group=[r for r in data[name] if r['task']==family]
            correct=sum(r['correct'] for r in group)
            summaries[family]={'n':len(group),'correct':correct,'ci95':wilson(correct,len(group)),
                               'truncated':sum(r.get('truncated',False) for r in group),
                               'repetitive_8gram_gt_20pct':sum(repetition(r.get('response',''))>.2 for r in group)}
        report['models'][name]={'response_file_sha256':digest(path),'summary':summaries}
    for family in report['models']['v2']['summary']:
        a,b=[[{**r,'group':r['id']} for r in data[k] if r['task']==family] for k in ['v2','v5']]
        amap={r['id']:r for r in a}
        if any(amap[r['id']]['prompt']!=r['prompt'] or amap[r['id']].get('gold')!=r.get('gold') for r in b):
            raise ValueError('Historical paired questions differ')
        report['paired_v2_v5'][family]=paired_difference(a,b)
    write_json('research/capability_recovery/results/historical_statistics.json',report)


if __name__=='__main__':
    main()
