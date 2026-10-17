"""CPU preparation only; no GPU rental or model training."""
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from prepare import encode, normalize
from sft_v2.prepare import canon, digest
from sft_v3.data import generated, verify, external_check

ROOT=Path('/home/serda/sft-v3-run-20261017')
OLD=Path('/home/serda/sft-v2-run-20261015')
SEED=20261017
REPO='allenai/tulu-3-sft-personas-instruction-following'
REV='fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e'


def main():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    ROOT.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(OLD/'tokenizer')
    tok.save_pretrained(ROOT/'tokenizer')
    rng=random.Random(SEED)
    frozen=[json.loads(l) for l in (OLD/'benchmark_cases.jsonl').read_text().splitlines()]
    forbidden=set();grams=set()
    for r in frozen:
        text=canon(r['prompt']);forbidden.add(digest(text));words=text.split()
        grams.update(digest(' '.join(words[i:i+13])) for i in range(len(words)-12))
    # Exact prompt identities from all previous train/development examples.
    old_prompts=set();replay=[]
    for split in ['train','validation']:
        with gzip.open(OLD/f'{split}.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line);old_prompts.add(row['prompt_hash'])
                if split=='train' and not row['source'].startswith('linguai/') and 'smol-constraints' not in row['source']:
                    replay.append(row)
    # Also keep new development prompts out of the original SFT training set.
    with gzip.open('/home/serda/sft-run-20261013/train.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            text=tok.decode(row['input_ids'],skip_special_tokens=False)
            prompts=re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',text,re.S)
            old_prompts.add(digest(canon('\n'.join(p.rstrip('\n') for p in prompts))))
    print('OLD_IDENTITIES',len(old_prompts),flush=True)
    with gzip.open(ROOT/'excluded_old_prompts.json.gz','wt') as stream:json.dump(sorted(old_prompts),stream)
    handles={s:gzip.open(ROOT/f'{s}.jsonl.gz','wt',encoding='utf8') for s in ['train','validation']}
    raw={s:gzip.open(ROOT/f'{s}_messages.jsonl.gz','wt',encoding='utf8') for s in ['train','validation']}
    counts=Counter();tokens=Counter();assistant=Counter();rejected=Counter();seen=set();review=[]
    new_dev=[]
    def accept(row,split,encoded=None,reused=False):
        messages=normalize(row)
        if not messages:rejected['malformed']+=1;return False
        prompt='\n'.join(m['content'] for m in messages if m['role']=='user')
        text=canon(prompt);key=digest(text);words=text.split()
        if key in seen:rejected['duplicate']+=1;return False
        if not reused and key in old_prompts:rejected['previous_training_or_dev']+=1;return False
        if key in forbidden or any(digest(' '.join(words[i:i+13])) in grams for i in range(len(words)-12)):
            rejected['benchmark_overlap']+=1;return False
        data=encoded or encode(messages,tok)
        if len(data['input_ids'])>2048 or 519 not in [v for v in data['labels'] if v!=-100]:rejected['length_or_eos']+=1;return False
        # A short complete answer is the intervention, never a cropped answer.
        answer_ids=tok(messages[-1]['content'],add_special_tokens=False)['input_ids']
        if not reused and len(answer_ids)>384:rejected['long_answer']+=1;return False
        for m in messages:
            if m['role']!='assistant':continue
            w=m['content'].split();gg=[tuple(w[i:i+6]) for i in range(len(w)-5)]
            if len(gg)>30 and 1-len(set(gg))/len(gg)>.15:rejected['repetition']+=1;return False
        if row.get('checks') and not verify(row,messages[-1]['content']):raise ValueError('Invalid procedural target')
        seen.add(key);data={**data,'source':row['source'],'family':row['family'],'prompt_hash':key,'group':row['group']}
        handles[split].write(json.dumps(data,ensure_ascii=False)+'\n')
        full={**row,'messages':messages,'prompt_hash':key};raw[split].write(json.dumps(full,ensure_ascii=False)+'\n')
        counts[f'{row["source"]}/{split}']+=1;tokens[split]+=len(data['input_ids']);assistant[split]+=sum(v!=-100 for v in data['labels'])
        if row.get('checks') and split=='validation':new_dev.append({**full,'messages':messages[:-1],'expected':messages[-1]['content']})
        if len(review)<32 or rng.random()<.002:review.append(full)
        return True
    for split,n,seed in [('train',12000,SEED),('validation',480,SEED+1)]:
        for row in generated(n,seed,heldout=split=='validation'):accept(row,split)
    print('PROCEDURAL',dict(counts),flush=True)
    path=hf_hub_download(REPO,'data/train-00000-of-00001.parquet',repo_type='dataset',revision=REV)
    rows=pq.read_table(path).to_pylist();rng.shuffle(rows);selected=Counter()
    for row in rows:
        checked=external_check(row)
        if checked is not True:
            rejected['external_constraint_unverifiable_or_failed']+=1;continue
        split='validation' if int(digest(row['id'])[:8],16)%11==0 else 'train'
        if selected[split]>={'train':1000,'validation':100}[split]:continue
        candidate={'messages':row['messages'],'source':REPO,'family':'natural_constraints_en','group':'tulu/'+row['id'],'source_constraints':row['constraints'],'target_verification':'single_advertised_constraint_checked; factual_correctness_not_proven'}
        if accept(candidate,split):selected[split]+=1
        if selected=={'train':1000,'validation':100}:break
    assert selected=={'train':1000,'validation':100},selected
    print('TULU',dict(selected),flush=True)
    rng.shuffle(replay);used=Counter()
    for old in replay:
        family=old.get('family','general')
        # 2k math + 2k other assistance; original old train records only.
        bucket='math' if family=='math' else 'general'
        if used[bucket]>=2000:continue
        text=tok.decode(old['input_ids'],skip_special_tokens=False)
        msg=[{'role':role,'content':content.rstrip('\n')} for role,content in re.findall(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>',text,re.S)]
        row={'messages':msg,'family':bucket,'source':'replay/'+old['source'],'group':'replay/'+old['prompt_hash'],'target_verification':'previous_training_replay'}
        if accept(row,'train',reused=True):used[bucket]+=1
        if used=={'math':2000,'general':2000}:break
    assert used=={'math':2000,'general':2000},used
    for h in [*handles.values(),*raw.values()]:h.close()
    (ROOT/'probes.json').write_text(json.dumps(new_dev,ensure_ascii=False,indent=2))
    (ROOT/'sample-review.json').write_text(json.dumps(review[:70],ensure_ascii=False,indent=2))
    old_manifest=json.loads((OLD/'manifest.json').read_text())
    manifest={'seed':SEED,'counts':dict(counts),'input_tokens':dict(tokens),'assistant_tokens':dict(assistant),'rejected':dict(rejected),'assistant_only_loss':True,'truncation':False,'max_input_tokens':2048,'max_new_target_tokens':384,'new_turkish_targets_programmatically_checked':True,'external_targets_programmatically_proven':False,'replay':dict(used),'sources':[{'repo':REPO,'revision':REV,'file':'data/train-00000-of-00001.parquet','license':'ODC-BY-1.0','url':'https://huggingface.co/datasets/'+REPO},{'repo':'linguai/grounded-instructions-v3','license':'Apache-2.0','generated_by':'sft_v3/data.py','note':'Fresh deterministic records; shared task templates across train and development. Not natural dialogue quality evidence.'},{'source':'v2_train_only_replay','parent_manifest_sha256':hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest(),'inherited_sources':old_manifest['sources']}],'benchmark_exclusion':'Frozen 2320 conditions: normalized prompt hashes and contiguous 13-word matches. New data excludes prior v1/v2 train and development exact prompt identities. Semantic leakage not ruled out.','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.jsonl.gz')}}
    manifest['source_code_sha256']={name:hashlib.sha256((PROJECT/'sft_v3'/name).read_bytes()).hexdigest() for name in ['data.py','prepare.py']}
    manifest['external_advertised_single_constraints_checked']=True
    manifest['excluded_old_prompt_hashes_sha256']=hashlib.sha256((ROOT/'excluded_old_prompts.json.gz').read_bytes()).hexdigest()
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    config={'status':'PREPARED_NOT_TRAINED','seed':SEED,'base_repo':'serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015','base_revision':'c8ee3888c486fa49d8d9e798a632c9f7ce664d7a','base_subfolder':'checkpoints/final','output_repo':'serda-dev/Jamba2-3B-Turkish-SFT-v3-20261017','epochs':1,'learning_rate':2e-6,'micro_batch':2,'gradient_accumulation':16,'eval_micro_batch':2,'max_length':2048,'bf16':True,'full_weight':True,'gradient_checkpointing':False,'optimizer':'adamw_bnb_8bit','warmup_ratio':.05,'eval_save_steps':200,'max_hours':3,'max_run_cost_usd':5,'stop_credit':15,'hard_stop_credit':12,'rental_ceiling_usd_per_hour':1.35,'checkpoint_selection':'lowest_development_assistant_loss; benchmark never selects weights','external_test_protocol':'existing frozen 2320 conditions, unchanged generation settings','acceptance':{'ifeval_min_correct':60,'ifeval_max_truncated':12,'gsm8k_min_correct':68,'gsm8k_cot8_min_correct':48,'belebele_min_correct':435,'mmlu_min_correct':208,'turkish_nll_max_ratio_to_v2':1.02,'english_nll_max_ratio_to_v2':1.03}}
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps({'counts':counts,'input_tokens':tokens,'assistant_tokens':assistant,'rejected':rejected}),flush=True)


if __name__=='__main__':main()
