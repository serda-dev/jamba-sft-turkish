"""Prepare 2,000 source-grounded Turkish tasks and token-balanced retention replay."""
import gzip,hashlib,json,random,re,shutil,sys
from pathlib import Path
from collections import Counter
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from prepare import encode
from sft_v2.prepare import canon,digest
from sft_v4.data import make_case,math_case,verify,sentences,compact,FAMILIES
ROOT=Path('/home/serda/sft-v4-run-20261018');SEED=20261018

def reading_cases():
 result=[];rng=random.Random(SEED+99)
 for i in range(64):
  name=['Ece','Ozan','Ada','Deniz','İpek','Mert','Selin','Bora'][i%8];cities=['Bursa','Samsun','İzmir','Konya'];days=['pazartesi','salı','perşembe','cuma'];activities=['seramik','fotoğrafçılık','tiyatro','resim']
  city=cities[i%4];day=days[(i//4)%4];activity=activities[(i//16)%4]
  text=f'{name}, {city} şehrinde yapılan {activity} atölyesine kayıt oldu. Atölye {day} günü başlayacak. Derslerden sonra aynı binadaki kitap kulübüyle buluşacak.'
  opts,answer,question=(cities,city,'Atölye hangi şehirde yapılıyor?') if i%3==0 else (days,day,'Atölye hangi gün başlayacak?') if i%3==1 else (activities,activity,'Kayıt olunan atölyenin konusu ne?')
  opts=list(opts);rng.shuffle(opts);gold=opts.index(answer)
  q=text+'\n\n'+question+'\n'+'\n'.join(chr(65+j)+'. '+v for j,v in enumerate(opts))+'\nYalnızca doğru seçeneğin harfini yaz.'
  result.append({'id':f'dev/reading/{i}','family':'reading','gold':gold,'choices':4,'messages':[{'role':'user','content':q},{'role':'assistant','content':chr(65+gold)}],'expected':chr(65+gold)})
 return result

def main():
 from transformers import AutoTokenizer
 ROOT.mkdir(exist_ok=True)
 from huggingface_hub import hf_hub_download
 import pyarrow.parquet as pq
 import pyarrow.compute as pc
 if not (ROOT/'aya_turkish_source.json').exists():
  path=hf_hub_download('CohereLabs/aya_dataset','data/train-00000-of-00001.parquet',revision='f9ea04583f02a8f86404ff6c58bf75fe637df8a2',repo_type='dataset');table=pq.read_table(path);table=table.filter(pc.equal(table['language_code'],'tur'));(ROOT/'aya_turkish_source.json').write_text(json.dumps(table.to_pylist(),ensure_ascii=False))
 if not (ROOT/'atlas_path.txt').exists() or not Path((ROOT/'atlas_path.txt').read_text()).exists():
  path=hf_hub_download('AlicanKiraz0/Turkce-Atlas-Instruct','train.chat.jsonl',revision='c387738c5deacbb35667ab4057930adc34e647e8',repo_type='dataset');(ROOT/'atlas_path.txt').write_text(path)
 rng=random.Random(SEED);tok=AutoTokenizer.from_pretrained('/home/serda/sft-v2-run-20261015/tokenizer');tok.save_pretrained(ROOT/'tokenizer')
 forbidden=set();grams=set()
 for line in Path('/home/serda/benchmark-run-20261014/cases.jsonl').read_text().splitlines():
  text=canon(json.loads(line)['prompt']);forbidden.add(digest(text));words=text.split();grams.update(digest(' '.join(words[i:i+13])) for i in range(len(words)-12))
 def overlap(text):
  text=canon(text);words=text.split();return digest(text) in forbidden or any(digest(' '.join(words[i:i+13])) in grams for i in range(len(words)-12))
 with gzip.open('/home/serda/sft-v3-run-20261017/excluded_old_prompts.json.gz','rt') as f:old_prompts=set(json.load(f))
 excluded=Counter();pool=[];seen_q=set();seen_a=set()
 sensitive=re.compile(r'kanser|tedavi|ilaç|hastalık|doktor|hastane|hukuk|yönetmelik|kanun|mahkeme|sigorta|vergi|yatırım|kredi|borsa|kripto|intihar|cinsel|fetva|kumar|ateşli silah',re.I)
 def candidate(question,answer,source,index):
  question=compact(question);answer=compact(answer);words=answer.split();qh=digest(canon(question));ah=digest(canon(answer))
  if not 28<=len(words)<=130 or len(sentences(answer))<3:excluded['length_or_sentence_count']+=1;return
  if qh in seen_q or ah in seen_a:excluded['duplicate_source']+=1;return
  if sensitive.search(question+' '+answer):excluded['sensitive_or_time_dependent_topic']+=1;return
  if any(s in answer for s in ['<|','<think>','```','http','**','\n#']):excluded['markup']+=1;return
  if re.search(r'(^|\s)\d+[.)]\s',answer) or answer.count('"')>4:excluded['preformatted']+=1;return
  chars=[c for c in answer if c.isalpha()]
  if not chars or sum(c.lower() in 'çğıöşü' for c in chars)<2:excluded['language_signal']+=1;return
  w=canon(answer).split();gg=[tuple(w[j:j+6]) for j in range(len(w)-5)]
  if len(gg)>20 and 1-len(set(gg))/len(gg)>.1:excluded['repetition']+=1;return
  if overlap(question) or overlap(answer):excluded['benchmark_overlap']+=1;return
  seen_q.add(qh);seen_a.add(ah);pool.append({'question':question,'answer':answer,'source':source,'row':index,'group':qh,'question_hash':qh,'answer_hash':ah,'prior_seen':qh in old_prompts})
 for i,row in enumerate(json.loads((ROOT/'aya_turkish_source.json').read_text())):candidate(row['inputs'],row['targets'],'CohereLabs/aya_dataset',i)
 atlas=Path((ROOT/'atlas_path.txt').read_text())
 with atlas.open() as f:
  for i,line in enumerate(f):
   row=json.loads(line);ms=row['messages'];users=[m['content'] for m in ms if m['role']=='user'];answers=[m['content'] for m in ms if m['role']=='assistant']
   if len(users)==len(answers)==1:candidate(users[0],answers[0],'AlicanKiraz0/Turkce-Atlas-Instruct',i)
   if len(pool)>=9000:break
 rng.shuffle(pool);print('CANDIDATE_POOL',len(pool),'new',sum(not x['prior_seen'] for x in pool),flush=True)
 assert len(pool)>=2400
 used=set();dev=[];train=[]
 def build(dest,n,split,eligible):
  for source in eligible:
   if source['group'] in used:continue
   row=make_case(source,len(dest),split)
   if overlap(row['messages'][-2]['content']) or overlap(row['expected']):excluded['augmented_overlap']+=1;continue
   data=encode(row['messages'],tok)
   if len(data['input_ids'])>1536 or len(tok(row['expected'],add_special_tokens=False)['input_ids'])>384:excluded['encoded_length']+=1;continue
   row['encoded']=data;dest.append(row);used.add(source['group'])
   if len(dest)==n:break
  assert len(dest)==n,(split,len(dest),n)
 build(dev,128,'dev',[x for x in pool if not x['prior_seen']]);build(train,2000,'train',pool)
 assert len({r['group'] for r in dev}&{r['group'] for r in train})==0
 instruction_tokens=sum(sum(v!=-100 for v in r['encoded']['labels']) for r in train);target=instruction_tokens*.15/.70
 math_tokens=0;i=0
 while math_tokens<target:
  row=math_case(i,'train');i+=1
  if overlap(row['messages'][0]['content']):continue
  row['encoded']=encode(row['messages'],tok);math_tokens+=sum(v!=-100 for v in row['encoded']['labels']);train.append(row)
  assert i<2500
 replay_tokens=0;replay_used=0;english_tokens=0;heldout_answers={r['source_answer_hash'] for r in dev}
 with gzip.open('/home/serda/sft-v2-run-20261015/train.jsonl.gz','rt') as f:
  for line in f:
   old=json.loads(line);source=old['source']
   if source.startswith(('linguai/','replay/linguai/')):continue
   if not any(x in source for x in ['Aya','aya_dataset','Turkce-Atlas','turkish_instructions','ultrachat']):continue
   text=tok.decode(old['input_ids'],skip_special_tokens=False);pieces=re.findall(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>',text,re.S)
   ms=[{'role':role,'content':body.rstrip('\n')} for role,body in pieces]
   if len([x for x in ms if x['role']=='user'])!=1:continue
   answer=ms[-1]['content'];q=next(x['content'] for x in ms if x['role']=='user');key=digest(canon(q))
   if key in used or digest(canon(answer)) in heldout_answers or not 35<=len(answer.split())<=140 or any(x in answer for x in ['<think>','```','**']):continue
   if overlap(q) or overlap(answer) or sensitive.search(q+' '+answer):continue
   is_english='ultrachat' in source
   if is_english and english_tokens>=target*.2:continue
   row={'id':f'train/replay/{replay_used}','group':key,'source':'replay/'+source,'family':'replay','component':'general','messages':ms,'expected':answer,'verification':'previous natural training target replay; facts not independently certified'}
   row['encoded']=encode(ms,tok)
   if len(row['encoded']['input_ids'])>1536:continue
   if any(not verify(x,x['expected'])['correct'] for x in dev):raise AssertionError('Dev target')
   n=sum(v!=-100 for v in row['encoded']['labels']);train.append(row);used.add(key);replay_tokens+=n;replay_used+=1;english_tokens+=n if is_english else 0
   if replay_tokens>=target:break
 assert replay_tokens>=target*.95
 for r in train+dev:
  data=r['encoded'];assert len(data['input_ids'])==len(data['labels'])<=1536
  target_ids=[v for v in data['labels'] if v!=-100];expected=''.join(m['content']+'<|im_end|>\n' for m in r['messages'] if m['role']=='assistant')
  assert tok.decode(target_ids,skip_special_tokens=False)==expected
  assert target_ids.count(519)==sum(m['role']=='assistant' for m in r['messages'])
  if r['component']!='general':assert verify(r,r['expected'])['correct']
 rng.shuffle(train)
 for split,items in [('train',train),('validation',dev)]:
  with gzip.open(ROOT/f'{split}.jsonl.gz','wt') as enc,gzip.open(ROOT/f'{split}_messages.jsonl.gz','wt') as raw:
   for row in items:
    data=row.pop('encoded');data.update(id=row['id'],group=row['group'],component=row['component'],family=row['family']);enc.write(json.dumps(data,ensure_ascii=False)+'\n');raw.write(json.dumps(row,ensure_ascii=False)+'\n')
 controls=[math_case(i,'dev') for i in range(64)]+reading_cases()
 (ROOT/'dev_cases.json').write_text(json.dumps(dev,ensure_ascii=False));(ROOT/'retention_cases.json').write_text(json.dumps(controls,ensure_ascii=False))
 (ROOT/'sample_review.json').write_text(json.dumps(random.Random(SEED).sample(train,32),ensure_ascii=False,indent=2))
 shutil.copy2('/home/serda/sft-v2-run-20261015/final_receipt.json',ROOT/'base_receipt.json')
 counts=Counter(r['component'] for r in train);total=instruction_tokens+math_tokens+replay_tokens
 manifest={'seed':SEED,'train_rows':len(train),'dev_rows':len(dev),'counts':dict(counts),'assistant_tokens':{'instruction':instruction_tokens,'math':math_tokens,'general':replay_tokens},'assistant_token_shares':{'instruction':instruction_tokens/total,'math':math_tokens/total,'general':replay_tokens/total},'instruction_families':dict(Counter(r['family'] for r in train if r['component']=='instruction')),'source_counts':dict(Counter(r['source'] for r in train if r['component']=='instruction')),'system_rows':sum(r['messages'][0]['role']=='system' for r in train if r['component']=='instruction'),'source_groups_train_dev_disjoint':True,'dev_original_source_question_prior_exact_seen':0,'all_instruction_targets_format_and_content_checked':True,'assistant_masks_eos_independently_decoded':True,'max_length':1536,'truncated_training_targets':0,'rejected':dict(excluded),'sources':[{'repo':'CohereLabs/aya_dataset','revision':'f9ea04583f02a8f86404ff6c58bf75fe637df8a2','license':'apache-2.0'},{'repo':'AlicanKiraz0/Turkce-Atlas-Instruct','revision':'c387738c5deacbb35667ab4057930adc34e647e8','license':'mit'},{'source':'v2_train_only_retention_replay','parent_manifest_sha256':hashlib.sha256(Path('/home/serda/sft-v2-run-20261015/manifest.json').read_bytes()).hexdigest()}],'scope':'Source-grounded editing/extraction and formatting. Not new free-form teacher-generated answers. Content preservation independently checked; source factual truth not certified. Split groups by original question and unique answer. Prior/frozen exclusions are exact and 13-word matches, not proof of semantic absence.'}
 manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.name in ['train.jsonl.gz','validation.jsonl.gz','train_messages.jsonl.gz','validation_messages.jsonl.gz','dev_cases.json','retention_cases.json','base_receipt.json']}
 (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 (ROOT/'data_audit.json').write_text(json.dumps({'verified':True,'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),'train_rows':len(train),'dev_rows':128,'instruction_targets':2000,'all_targets_checked':True,'train_dev_original_groups_disjoint':True,'no_truncation':True},indent=2))
 print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
