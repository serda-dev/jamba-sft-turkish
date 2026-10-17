"""Reproduce the data, independent mask and contract audit without a GPU."""
import ast
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/'benchmarks')]
from sft_v3.data import verify
from instruction_following_eval.instructions_registry import INSTRUCTION_DICT
ROOT=Path('/home/serda/sft-v3-run-20261017')
OUT=Path(__file__).parent


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
    groups=defaultdict(lambda:{'rows':0,'assistant_tokens':0,'system_rows':0})
    errors=Counter();lengths=defaultdict(list);examples={};paragraphs=Counter()
    with gzip.open(ROOT/'train.jsonl.gz','rt') as encoded,gzip.open(ROOT/'train_messages.jsonl.gz','rt') as messages:
        for left,right in zip(encoded,messages):
            e=json.loads(left);r=json.loads(right)
            group='new_turkish' if r['source'].startswith('linguai/') else 'replay' if r['source'].startswith('replay/') else 'external_english'
            target=[v for v in e['labels'] if v!=-100]
            groups[group]['rows']+=1;groups[group]['assistant_tokens']+=len(target)
            groups[group]['system_rows']+=any(m['role']=='system' for m in r['messages'])
            expected=''.join(m['content']+'<|im_end|>\n' for m in r['messages'] if m['role']=='assistant')
            errors['assistant_decode_mismatch']+=tok.decode(target,skip_special_tokens=False)!=expected
            errors['eos_count_mismatch']+=target.count(519)!=sum(m['role']=='assistant' for m in r['messages'])
            errors['role_marker_labeled']+=518 in target;errors['padding_labeled']+=0 in target
            lengths[group].append(len(r['messages'][-1]['content'].split()))
            if group=='new_turkish':
                examples.setdefault(r['family'],r)
                if r['family'] in ['paragraphs','bounded_explanation']:
                    n=next(c['n'] for c in r['checks'] if c['kind']=='paragraphs')
                    checker=INSTRUCTION_DICT['length_constraints:number_paragraphs']('length_constraints:number_paragraphs');checker.build_description(num_paragraphs=n)
                    paragraphs['blank_line_targets']+=1
                    paragraphs['passes_star_contract']+=checker.check_following(r['messages'][-1]['content'])
    total=sum(g['assistant_tokens'] for g in groups.values())
    for name,g in groups.items():
        g['assistant_token_share_percent']=100*g['assistant_tokens']/total
        g['median_answer_words']=sorted(lengths[name])[len(lengths[name])//2]
    by_rule={};repetition={}
    for name,folder in [('v2','/home/serda/sft-v2-run-20261015'),('v3',str(ROOT))]:
        rr=[json.loads(l) for l in (Path(folder)/'final_benchmark.jsonl').read_text().splitlines() if json.loads(l)['task']=='ifeval_tr_subset']
        tally=defaultdict(lambda:[0,0]);rep=0
        for r in rr:
            ids=r['rules']['instruction_id_list'];ids=ast.literal_eval(ids) if isinstance(ids,str) else ids
            for ident,ok in zip(ids,r['checks']):tally[ident][0]+=ok;tally[ident][1]+=1
            words=r['response'].split();grams=[tuple(words[i:i+8]) for i in range(len(words)-7)]
            rep+=bool(grams and 1-len(set(grams))/len(grams)>.2)
        by_rule[name]=dict(tally);repetition[name]=rep
    r=examples['frequency'];n=next(c['n'] for c in r['checks'] if c['kind']=='frequency');kw=next(c['value'] for c in r['checks'] if c['kind']=='frequency');code=next(c['values'][0] for c in r['checks'] if c['kind']=='contains')
    bad_lines=verify(r,code+' '+' '.join([kw]*n))
    r=examples['two_answers'];code=next(c['value'] for c in r['checks'] if c['kind']=='frequency')
    duplicate=verify(r,code+'\n******\n'+code)
    result={'rows':sum(g['rows'] for g in groups.values()),'source_mix':dict(groups),'independent_label_checks':dict(errors),'paragraph_contract_coverage':dict(paragraphs),'legacy_checker_counterexamples':{'wrong_line_count_accepted':bad_lines,'duplicate_answers_accepted':duplicate},'rule_scores':by_rule,'ifeval_repetitive_gt20pct_8grams':repetition,'note':'Target-token shares are not exact gradient weights. Paragraph targets match their own blank-line prompts; star-check failures demonstrate a coverage mismatch, not mislabeled targets.'}
    (OUT/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rule_scores'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
