"""Pinned sources, assistant-only labels, prompt-group split and provenance."""
import gzip
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

SYSTEM='Sen yardımsever bir Türkçe asistansın. Doğru, açık ve kullanıcının istediği biçimde yanıt ver.'


def encode(messages, tokenizer):
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    result=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    spans=[]
    for i,message in enumerate(messages):
        if message['role']!='assistant':continue
        prefix=tokenizer.apply_chat_template(messages[:i],tokenize=False,add_generation_prompt=True)
        complete=tokenizer.apply_chat_template(messages[:i+1],tokenize=False,add_generation_prompt=False)
        assert text.startswith(prefix) and text.startswith(complete)
        spans.append((len(prefix),len(complete)))
    labels=[token if any(start<=a and b<=end and b>a for start,end in spans) else -100
            for token,(a,b) in zip(result['input_ids'],result['offset_mapping'])]
    return {'input_ids':result['input_ids'],'labels':labels}


def normalize(row):
    if 'messages' in row:
        messages=[{'role':m['role'],'content':m.get('content') or ''} for m in row['messages']]
    elif 'inputs' in row:
        messages=[{'role':'user','content':row['inputs']},{'role':'assistant','content':row['targets']}]
    else:
        row={k.strip():v for k,v in row.items()}
        prompt=(row.get('talimat') or '').strip()
        if row.get('giriş'):prompt+='\n\n'+row['giriş'].strip()
        messages=[{'role':'user','content':prompt},{'role':'assistant','content':(row.get('çıktı') or '').strip()}]
    if not messages or messages[-1]['role']!='assistant':return None
    if messages[0]['role']!='system':messages.insert(0,{'role':'system','content':SYSTEM})
    expected='user'
    for i,m in enumerate(messages):
        if not isinstance(m['content'],str) or not m['content'].strip():return None
        if i==0 and m['role']=='system':continue
        if m['role']!=expected:return None
        expected='assistant' if expected=='user' else 'user'
        if any(t in m['content'] for t in ['<|im_start|>','<|im_end|>','<think>']):return None
    # Reject obvious generation boilerplate, degeneration and incomplete answers.
    for m in messages:
        if m['role']=='assistant':
            if any(t in m['content'].lower() for t in ['bu soruya cevap veriyoruz','bu soruya cevap verirken']):return None
            lines=[x.strip() for x in m['content'].splitlines() if x.strip()]
            if len(lines)>8 and len(set(lines))/len(lines)<.6:return None
    return messages


def controlled(seed,count,offset=0):
    rng=random.Random(seed)
    for index in range(count):
        a=rng.randint(1,999)+offset;b=rng.randint(1,999)+offset
        mode=index%4
        if mode==0:
            prompt=f'{a} + {b} işleminin sonucunu yalnızca {{"sonuc": sayı}} biçiminde JSON olarak yaz.'
            response=json.dumps({'sonuc':a+b},ensure_ascii=False)
        elif mode==1:
            name=rng.choice(['Defne','Selin','Arda','Eylül','Bora','İpek','Duru','Mert'])+str(a)
            city=rng.choice(['İzmir','Bursa','Konya','Antalya','Eskişehir','Samsun'])
            prompt=f'{name}, {city} şehrinde yaşıyor. Yaşı belirtilmedi. Yalnızca isim, sehir ve yas alanlarını içeren JSON yaz. Bilinmeyen yaş null olsun.'
            response=json.dumps({'isim':name,'sehir':city,'yas':None},ensure_ascii=False)
        elif mode==2:
            prompt=f'{a} sayısını {b} ile çarp. Yalnızca sonucu yaz, açıklama ekleme.'
            response=str(a*b)
        else:
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':f'Kodum K{a}-{b}. Bunu hatırla.'},
                {'role':'assistant','content':'Tamam, kodunu hatırlıyorum.'},
                {'role':'user','content':'Kodumu yalnızca kod alanını içeren JSON olarak yaz.'},
                {'role':'assistant','content':json.dumps({'kod':f'K{a}-{b}'})}]
            yield messages;continue
        yield [{'role':'system','content':SYSTEM},{'role':'user','content':prompt},{'role':'assistant','content':response}]


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('config.json').read_text());(root/'config.json').write_text(json.dumps(cfg,indent=2))
    tok=AutoTokenizer.from_pretrained(cfg['base_repo'],revision=cfg['base_revision'],subfolder=cfg['base_subfolder'])
    tok.save_pretrained(root/'tokenizer')
    seen=set();counts=Counter();rejected=Counter();tokens=Counter();rows=[]
    def accept(messages,source):
        if not messages:return False
        prompt='\n'.join(m['content'] for m in messages if m['role']=='user')
        key=hashlib.sha256(re.sub(r'\s+',' ',prompt).strip().casefold().encode()).hexdigest()
        if key in seen:rejected['duplicate_prompt']+=1;return False
        data=encode(messages,tok)
        if len(data['input_ids'])>cfg['max_length'] or sum(x!=-100 for x in data['labels'])<2:
            rejected['length_or_empty_target']+=1;return False
        seen.add(key);data.update(source=source,prompt_hash=key)
        # A prompt and its paraphrase-equivalent whitespace/case variants stay together.
        split='validation' if int(key[:8],16)%100==0 else 'train'
        rows.append((split,data));counts[f'{source}/{split}']+=1
        tokens[split]+=len(data['input_ids'])
        return True
    for source in cfg['sources']:
        print('SOURCE',source['repo'],flush=True)
        ds=load_dataset(source['repo'],split=source['split'],revision=source['revision'],streaming=True)
        if source.get('language_code'):ds=ds.filter(lambda r:r.get('language_code')=='tur')
        ds=ds.shuffle(seed=cfg['seed'],buffer_size=10000)
        accepted=0
        for row in ds:
            if accept(normalize(row),source['repo']):accepted+=1
            if accepted>=source['count']:break
        print('ACCEPTED',source['repo'],accepted,flush=True)
    for messages in controlled(cfg['seed'],8000):accept(messages,'linguai/validated-format-v1')
    random.Random(cfg['seed']).shuffle(rows)
    for split in ['train','validation']:
        with gzip.open(root/f'{split}.jsonl.gz','wt',encoding='utf-8') as f:
            for s,row in rows:
                if s==split:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    cases=list(controlled(cfg['seed']+1,80,offset=2000))
    (root/'eval_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    review=random.Random(cfg['seed']).sample(rows,min(30,len(rows)))
    (root/'sample-review.json').write_text(json.dumps([{'source':r['source'],'text':tok.decode(r['input_ids'])} for _,r in review],ensure_ascii=False,indent=2))
    manifest={'counts':dict(counts),'input_tokens':dict(tokens),'rejected':dict(rejected),'sources':cfg['sources'],
        'assistant_only_loss':True,'truncation':False,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.jsonl.gz')}}
    assert tokens['train']>10000000 and len(rows)>30000
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
