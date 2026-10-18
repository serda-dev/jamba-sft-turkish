"""Audited broad replay from V2 plus limited grounded instruction examples."""
import gzip, hashlib, json, random, re, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from prepare import encode
from sft_v2.prepare import canon,digest
ROOT=Path('/home/serda/sft-v5-run-20261018');OLD=Path('/home/serda/sft-v2-run-20261015');V4=Path('/home/serda/sft-v4-run-20261018')
SEED=20261018

def messages(tok,row):
 text=tok.decode(row['input_ids'],skip_special_tokens=False)
 return [{'role':role,'content':content} for role,content in re.findall(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>',text,re.S)]

def check_labels(tok,row,ms):
 ids=row['input_ids'];labels=row['labels'];assert len(ids)==len(labels)<=3072 and labels[0]==-100
 assert all(y==-100 or x==y for x,y in zip(ids,labels))
 decoded=tok.decode([x for x in labels if x!=-100],skip_special_tokens=False)
 expected=''.join(m['content']+'<|im_end|>\n' for m in ms if m['role']=='assistant')
 assert decoded==expected,(row.get('source'),decoded[:100],expected[:100])
 assert sum(x==519 for x in labels)==sum(m['role']=='assistant' for m in ms)

def main():
 from transformers import AutoTokenizer
 from huggingface_hub import hf_hub_download
 ROOT.mkdir(exist_ok=True);rng=random.Random(SEED);tok=AutoTokenizer.from_pretrained(OLD/'tokenizer');shutil.copytree(OLD/'tokenizer',ROOT/'tokenizer',dirs_exist_ok=True)
 cfg={'status':'AUDITED_PREPARATION','seed':SEED,'base_repo':'serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015','base_revision':'c8ee3888c486fa49d8d9e798a632c9f7ce664d7a','base_subfolder':'checkpoints/final','output_repo':'serda-dev/Jamba2-3B-Turkish-SFT-v5-20261018','epochs':1,'learning_rate':7.5e-7,'micro_batch':1,'gradient_accumulation':32,'max_length':3072,'eval_steps':400,'pilot_steps':100,'dev_max_new_tokens':512,'max_hours':6,'max_run_cost_usd':8,'stop_credit':7,'hard_stop_credit':5,'rental_ceiling_usd_per_hour':1.35,'warmup_steps':50,'loss_weighting':'assistant-token mean across each complete optimizer batch','acceptance':{'ifeval_min_correct':60,'ifeval_max_truncated':12,'gsm8k_min_correct':68,'gsm8k_cot8_min_correct':48,'belebele_min_correct':435,'mmlu_min_correct':208,'turkish_nll_max_ratio_to_v2':1.02,'english_nll_max_ratio_to_v2':1.03},'pilot_policy':{'instruction_min_gain':4,'natural_math_max_drop':4,'dev_max_extra_truncation':2,'family_loss_max_ratio':1.05},'training_selection':'natural-math and per-family dev-loss retention, then instruction strict, then fewer truncations; old external benchmark never selects weights','second_epoch_automatic':False}
 (ROOT/'config.json').write_text(json.dumps(cfg,indent=2));shutil.copy2(V4/'base_receipt.json',ROOT/'base_receipt.json');shutil.copy2(V4/'dev_cases.json',ROOT/'dev_cases.json')
 # Keep all upstream validation rows out of training and add explicit real-math probes.
 validation=[];retention=[]
 with gzip.open(OLD/'validation.jsonl.gz','rt') as f:
  for line in f:
   row=json.loads(line);validation.append(row)
   if row['family']=='math' and not row['source'].startswith('linguai/'):
    ms=messages(tok,row);gold=ms[-1]['content'].rsplit('####',1)[-1].strip();assert re.fullmatch(r'-?[\d.,]+',gold)
    retention.append({'id':f'dev/natural_math/{len(retention)}','family':'natural_math','source':row['source'],'gold':gold,'messages':ms,'expected':ms[-1]['content']})
 assert len(retention)==143
 (ROOT/'retention_cases.json').write_text(json.dumps(retention,ensure_ascii=False))
 forbidden={r['prompt_hash'] for r in validation};frozen=[]
 for line in Path('/home/serda/benchmark-run-20261014/cases.jsonl').read_text().splitlines():frozen.append(json.loads(line)['prompt'])
 # Entire GSM8K-TR test exclusion and a sealed extra test from previously unmeasured questions.
 path=hf_hub_download('malhajar/gsm8k-tr','main/test-1318.jsonl',repo_type='dataset',revision='1046702fc5d8e14330ae90fdfd98735194ffa4ce')
 all_math=[json.loads(l) for l in Path(path).read_text().splitlines()]
 old_math=[p for p in frozen if '####' in p or 'sayı' in p]
 excluded_words=set()
 for p in frozen:
  words=canon(p).split();excluded_words.update(tuple(words[i:i+13]) for i in range(len(words)-12))
 sealed=[]
 for i,row in enumerate(all_math):
  words=canon(row['question']).split()
  if any(tuple(words[j:j+13]) in excluded_words for j in range(len(words)-12)):continue
  if len(words)<15:continue
  ms=[{'role':'user','content':row['question']+'\nKısa çözümü açıkla. Son satır #### sayı biçiminde olsun.'},{'role':'assistant','content':re.sub(r'<<[^>]*>>','',row['answer'])}]
  sealed.append({'id':f'sealed/gsm8k_tr/{i}','family':'natural_math','gold':row['answer'].rsplit('####',1)[-1].strip(),'messages':ms,'expected':ms[-1]['content']})
 rng.shuffle(sealed);sealed=sealed[:128];assert len(sealed)==128
 (ROOT/'sealed_math_cases.json').write_text(json.dumps(sealed,ensure_ascii=False))
 frozen += [r['question'] for r in all_math]
 grams=set()
 for p in frozen:
  words=canon(p).split();grams.update(tuple(words[i:i+13]) for i in range(len(words)-12));forbidden.add(digest(canon(p)))
 def overlaps(ms):
  q=canon('\n'.join(m['content'] for m in ms if m['role']=='user'));w=q.split()
  return digest(q) in forbidden or any(tuple(w[j:j+13]) in grams for j in range(len(w)-12))
 # Source-level quotas retain broad coverage; report actual token shares instead of claiming proposed shares.
 quotas={'replay/AlicanKiraz0/Turkce-Atlas-Instruct':12000,'replay/merve/turkish_instructions':6000,'replay/CohereLabs/aya_dataset':3500,'replay/HuggingFaceH4/ultrachat_200k':3000,'malhajar/gsm8k-tr':7396,'openai/gsm8k':3000,'barandinho/turkish-reasoning-distilled-sft/webinstruct_verified':14000,'barandinho/turkish-reasoning-distilled-sft/openthoughts_science_verified':3000,'barandinho/turkish-reasoning-distilled-sft/acecode_verified':5000,'HuggingFaceTB/smoltalk/smol-constraints':2500,'HuggingFaceTB/smoltalk/smol-summarize':2500}
 pool=defaultdict(list)
 with gzip.open(OLD/'train.jsonl.gz','rt') as f:
  for line in f:
   row=json.loads(line)
   if row['source'] in quotas:pool[row['source']].append(row)
 selected=[]
 for source,items in pool.items():rng.shuffle(items);selected.extend(items[:quotas[source]])
 rng.shuffle(selected);counts=Counter();families=Counter();tokens=Counter();inputs=0;seen=set();reject=Counter();reviews=[];multi=0;longest=0
 # The V4 dev originals are disjoint from old training by its prior audited group exclusion.
 dev=json.loads((ROOT/'dev_cases.json').read_text());heldout_answers={r['source_answer_hash'] for r in dev};heldout_questions={r['source_question_hash'] for r in dev}
 out=gzip.open(ROOT/'train.jsonl.gz','wt');msgout=gzip.open(ROOT/'train_messages.jsonl.gz','wt')
 def accept(row,ms,new=False):
  nonlocal inputs,multi,longest
  q='\n'.join(m['content'] for m in ms if m['role']=='user');key=digest(canon(q))
  if key in seen or overlaps(ms):reject['duplicate_or_test_overlap']+=1;return
  answers=[m['content'] for m in ms if m['role']=='assistant']
  if key in heldout_questions or any(digest(canon(a)) in heldout_answers for a in answers):reject['v4_dev_source_overlap']+=1;return
  check_labels(tok,row,ms);row['prompt_hash']=key;row['n_target_tokens']=sum(x!=-100 for x in row['labels']);row['is_new_task']=new
  seen.add(key);out.write(json.dumps(row,ensure_ascii=False)+'\n');msgout.write(json.dumps({'messages':ms,'source':row['source'],'family':row['family'],'is_new_task':new},ensure_ascii=False)+'\n')
  counts[row['source']]+=1;families[row['family']]+=1;tokens[row['family']]+=row['n_target_tokens'];inputs+=len(row['input_ids']);longest=max(longest,len(row['input_ids']));multi+=sum(m['role']=='assistant' for m in ms)>1
  if len(reviews)<120 and counts[row['source']]<=12:reviews.append({'source':row['source'],'messages':ms})
 for i,row in enumerate(selected):
  accept(row,messages(tok,row))
  if i%5000==0:print('AUDIT',i,len(selected),flush=True)
 with gzip.open(V4/'train_messages.jsonl.gz','rt') as f:
  for line in f:
   row=json.loads(line)
   if row.get('component')!='instruction':continue
   ms=row['messages'];data=encode(ms,tok);data.update(source='linguai/grounded-v4-reuse',family='grounded_instruction',group=row['group']);accept(data,ms,True)
 out.close();msgout.close();assert 60000<=sum(counts.values())<=80000,dict(counts)
 with gzip.open(ROOT/'validation.jsonl.gz','wt') as f,gzip.open(ROOT/'validation_messages.jsonl.gz','wt') as mf:
  for row in validation:
   ms=messages(tok,row);assert row['prompt_hash'] not in seen;check_labels(tok,row,ms);f.write(json.dumps(row)+'\n');mf.write(json.dumps({'messages':ms,'source':row['source'],'family':row['family']},ensure_ascii=False)+'\n')
 (ROOT/'sample-review.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2))
 # Complete manifest is frozen before rental. Previously seen train replay is explicit.
 manifest={'seed':SEED,'train_rows':sum(counts.values()),'validation_rows':len(validation),'counts':dict(counts),'families':dict(families),'assistant_tokens_by_family':dict(tokens),'assistant_token_shares':{k:v/sum(tokens.values()) for k,v in tokens.items()},'input_tokens':inputs,'assistant_tokens':sum(tokens.values()),'multi_turn_rows':multi,'max_length':3072,'longest_tokens':longest,'rejected':dict(reject),'upstream_v2_manifest_sha256':hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest(),'sources':json.loads((OLD/'manifest.json').read_text())['sources'],'replay_sources':json.loads((OLD/'manifest.json').read_text())['replay_manifest']['sources'],'grounded_sources':json.loads((V4/'manifest.json').read_text())['sources'],'scope':'Broad V2-train replay plus 2000 grounded V4 tasks; not 60k newly generated teacher answers. Natural math development reuses V2 validation; translation/semantic overlaps remain possible. Source factual correctness is not independently certified. Sealed math uses previously unmeasured GSM8K-TR test questions; not used for model selection. Source quotas approximate coverage; actual target-token shares differ from proposal.','all_targets_independently_decoded':True,'truncated_targets':0}
 manifest['files']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['train.jsonl.gz','train_messages.jsonl.gz','validation.jsonl.gz','validation_messages.jsonl.gz','dev_cases.json','retention_cases.json','sealed_math_cases.json','base_receipt.json']}
 (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));audit={'verified':True,'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),'all_label_eos_decodes_checked':True,'train_validation_exact_overlap':0,'natural_math_development_rows':143,'sealed_math_rows':128,'new_task_rows':counts['linguai/grounded-v4-reuse']};(ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2))
 publish=ROOT/'hub_publish';publish.mkdir(exist_ok=True)
 for name in ['manifest.json','config.json','data_audit.json','sample-review.json']:shutil.copy2(ROOT/name,publish/name)
 for name in ['train.jsonl.gz','validation.jsonl.gz','train_messages.jsonl.gz','dev_cases.json','retention_cases.json','sealed_math_cases.json']:shutil.copy2(ROOT/name,publish/name)
 for path in (PROJECT/'sft_v5').glob('*.py'):shutil.copy2(path,publish/path.name)
 print('PREPARED',json.dumps(manifest),flush=True)

if __name__=='__main__':main()
