"""Targeted SFT revision: pinned data, benchmark exclusion and grouped splits."""
import sys,gzip,json,re,hashlib,random,unicodedata
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare import encode,normalize,SYSTEM
ROOT=Path('/home/serda/sft-v2-run-20261015')
SEED=20261015

def canon(s):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def messages(q,a,system=SYSTEM):return [{'role':'system','content':system},{'role':'user','content':q},{'role':'assistant','content':a}]

def generated(n,seed=SEED,heldout=False):
    """Executable targets; independent values and prompt wording for held-out probes."""
    rng=random.Random(seed); offset=10000 if heldout else 0
    for i in range(n):
        a=rng.randint(21,99)+offset;b=rng.randint(3,19);c=rng.randint(2,9)
        mode=i%28;name=rng.choice(['Defne','Arda','Eylül','Bora','İpek','Duru','Mert','Selin'])
        group=f'v2/{seed}/{i}';family='instruction'
        if mode<6:
            family='math'; variant=mode%3
            if variant==0:q=f'{name}, her birinde {b} kalem bulunan {c} kutu aldı. Elindeki {a} kalemle birlikte kaç kalemi oldu?';value=a+b*c;solution=f'Kutulardaki kalemler {b} × {c} = {b*c}. Önceki kalemlerle toplam {a} + {b*c} = {value}.'
            elif variant==1:q=f'Bir depoda {a+b*c} paket var. {c} gün boyunca her gün {b} paket gönderildi. Kaç paket kaldı?';value=a;solution=f'Gönderilen paket sayısı {c} × {b} = {b*c}. Kalan {a+b*c} − {b*c} = {value}.'
            else:q=f'{name} {a*c} TL parasının {b*c} TL kısmını harcadı. Kalan parayı {c} kişiye eşit bölerse kişi başına kaç TL düşer?';value=a-b;solution=f'Kalan para {a*c} − {b*c} = {(a-b)*c} TL. Kişi başına {(a-b)*c} ÷ {c} = {value} TL.'
            if mode<3:q+='\nKısa çözümü açıkla; son satırı #### sayı biçiminde yaz.';ans=solution+f'\n#### {value}'
            else:
                ex='Örnek: İki kutunun her birinde üç kalem varsa toplam 2 × 3 = 6 kalem vardır.\n\n'
                q=ex+'Asıl soru: '+q+'\nYalnızca asıl soruyu çöz. Son satır #### sayı olsun.';ans=solution+f'\n#### {value}'
        elif mode in [6,7,8,9]:
            family='reading';cities=['Bursa','Samsun','İzmir','Konya','Antalya'];rng.shuffle(cities);gold=i%5
            if mode==6:choices=cities;value=rng.choice(cities);text=f'{name}, {value} şehrinde yaşıyor. Her sabah saat {b} civarında kütüphaneye gidiyor.';question=f'{name} hangi şehirde yaşıyor?'
            elif mode==7:choices=[str(a+j*b) for j in range(5)];value=choices[rng.randrange(5)];text=f'Okul kütüphanesine pazartesi {value} kitap, salı {b} dergi geldi. Çarşamba günü kapalıydı.';question='Pazartesi kaç kitap geldi?'
            elif mode==8:choices=[str(a+j) for j in range(5)];value=str(a+2);text=f'{name} birinci gün {a} sayfa, ikinci gün iki sayfa fazla okudu.';question='İkinci gün kaç sayfa okudu?'
            else:choices=['Pazartesi','Salı','Çarşamba','Perşembe','Cuma'];value=rng.choice(choices);text=f'Atölye {value} günü yapılacak. Katılımcılar kayıt için bir önceki gün başvurmalı.';question='Atölye hangi gün yapılacak?'
            choices.remove(value);rng.shuffle(choices);choices.insert(gold,value)
            q='Metne göre yanıtla. Yalnızca doğru seçeneğin harfini yaz.\n\n'+text+'\n\n'+question+'\n'+'\n'.join(f'{chr(65+j)}. {v}' for j,v in enumerate(choices));ans=chr(65+gold)
        elif mode==10:q=f'{name} adlı kişinin şehir bilgisi {rng.choice(["Bursa","İzmir","Samsun"])}. Yaşı belirtilmedi. Sadece ad, sehir, yas alanlarıyla JSON yaz.';city=q.split('bilgisi ')[1].split('.')[0];ans=json.dumps({'ad':name,'sehir':city,'yas':None},ensure_ascii=False)
        elif mode==11:q=f'Şu {c} kayıt için birer madde yaz; her madde sadece "Kayıt: N" biçiminde olsun. N değerleri {a} ile {a+c-1} arasında. Başlık ekleme.';ans='\n'.join(f'- Kayıt: {j}' for j in range(a,a+c))
        elif mode==12:q=f'{name} için bir davetiye taslağı yaz. Tam iki cümle olsun. İlk cümlede {name}, ikinci cümlede "katılım" sözcüğü bulunsun.';ans=f'{name}, sizi buluşmamıza davet ediyoruz. Lütfen katılım durumunuzu bildirin.'
        elif mode==13:q=f'Sadece şu sözcükleri büyük harfe dönüştür: "izmir ışık {name}". Açıklama yazma.';ans=('izmir ışık '+name).translate(str.maketrans('iı','İI')).upper()
        elif mode==14:q=f'{a}, {b}, {c} sayılarını artan sırada, yalnızca virgülle ayırarak yaz.';ans=','.join(map(str,sorted([a,b,c])))
        elif mode==15:q=f'Yanıtın tam olarak "Başlangıç" ile başlasın ve "Bitti." ile bitsin. Arada {a} sayısının çift mi tek mi olduğunu belirt.';ans=f'Başlangıç: {a} '+('çift' if a%2==0 else 'tek')+' sayıdır. Bitti.'
        elif mode==16:q=f'Bir form için tam {c} ayrı satır yaz. Her satır farklı bir köşeli parantezli alan içersin; alan adları alan_{a} ile alan_{a+c-1} arasında olsun. Başka metin ekleme.';ans='\n'.join(f'[alan_{j}]' for j in range(a,a+c))
        elif mode==17:
            code=f'TR-{a}-{b}-{c}';m=messages(f'Proje kodum {code}. Bunu hatırla.','Proje kodunu not aldım.');m.extend([{'role':'user','content':'Kodumu sadece kod alanıyla JSON olarak ver.'},{'role':'assistant','content':json.dumps({'kod':code})}]);yield m,'conversation',group;continue
        elif mode==18:q=f'{name} için {c} satırlık bir tablo oluştur. Başlık satırı "ad,puan" olsun. Her veri satırında {name} ve {a} ile {a+c-1} arasında sıradaki puan bulunsun. Yalnızca CSV ver.';ans='ad,puan\n'+'\n'.join(f'{name},{j}' for j in range(a,a+c))
        elif mode==19:q=f'Elimde {a} bilet var. Kaçının satıldığını bilmiyorum. Kalan bilet sayısını hesaplamak için eksik olan tek bilgiyi sor; sayı uydurma.';ans='Kaç bilet satıldı?'
        elif mode==20:
            q=f'{name} için {c} kısa paragraf yaz. Paragrafları boş satırla ayır; numaralandırma yapma. Her paragrafta aynı kişi farklı bir gün {a} ile {a+c-1} arasında sıradaki sayıda sayfa okusun.'
            ans='\n\n'.join(f'{name} bugün {j} sayfa okudu.' for j in range(a,a+c))
        elif mode==21:
            words=['bugün','birlikte','yeni','bir','kitap','okumak','bize','farklı','düşünceler','kazandırır']
            k=rng.randint(4,10);q=f'Okuma hakkında tam {k} sözcükten oluşan tek satırlık bir ifade yaz. Noktalama kullanma.';ans=' '.join(words[:k])
        elif mode==22:
            k=rng.randint(2,5);q=f'Her satırda "merhaba" sözcüğünü bir kez kullanarak {k} satırlık karşılama yaz. Başlık ve başka merhaba ekleme.';ans='\n'.join(f'merhaba {name}{j}' for j in range(k))
        elif mode==23:q=f'{name} için kısa ve resmi iki karşılama yaz. İki yanıtı yalnızca altı yıldızla ayır.';ans=f'Merhaba {name}!\n******\nSayın {name}, hoş geldiniz.'
        elif mode==24:q=f'Her biri ayrı satırda tam {c} farklı vurgulu etiket yaz. Etiketler etiket{a} ile etiket{a+c-1} arasında olsun ve ** ile sarılsın. Başka metin ekleme.';ans='\n'.join(f'**etiket{j}**' for j in range(a,a+c))
        elif mode==25:q=f'Tam iki bölüm yaz: "Bölüm 1" başlığında {a} sayısını, "Bölüm 2" başlığında {b} sayısını ver. Her başlığın ardından tek satır sayı gelsin.';ans=f'Bölüm 1\n{a}\n\nBölüm 2\n{b}'
        elif mode==26:q=f'Yalnızca şu ifadeyi Türkçe küçük harfe dönüştür: "İZMİR IŞIK {name.upper()}".';ans=('İZMİR IŞIK '+name.upper()).translate(str.maketrans('İI','iı')).lower()
        else:q=f'{name} için virgül kullanmadan tam bir cümlelik davet yaz. Yanıtın tamamını çift tırnak içine al.';ans=f'"{name} seni yarın buluşmaya bekliyoruz."'
        if heldout:q='İsteği verilen koşullara uyarak yerine getir:\n'+q
        yield messages(q,ans),family,group

def main():
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi,hf_hub_download
    import pyarrow.parquet as pq
    ROOT.mkdir(exist_ok=True);api=HfApi();rng=random.Random(SEED)
    tok=AutoTokenizer.from_pretrained('/home/serda/sft-run-20261013/tokenizer');tok.save_pretrained(ROOT/'tokenizer')
    cfg={'seed':SEED,'max_length':3072,'epochs':2,'learning_rate':5e-6,'micro_batch':4,'gradient_accumulation':8,'eval_save_steps':400,'pilot_steps':120,
      'output_repo':'serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015','max_hours':18,'stop_credit':12,'hard_stop_credit':7,
      'candidates':[{'name':'sft','repo':'serda-dev/Jamba2-3B-Turkish','revision':'089da0e1c0f95932f7610e1d3fec0ad10e66c33c','subfolder':''},{'name':'cpt','repo':'serda-dev/Jamba2-3B-Turkish-CPT-1B-20261007','revision':'8b1a9c812791fa7930e64b9444b41519c023269e','subfolder':'checkpoints/final'}]}
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2))
    cases=[json.loads(l) for l in Path('/home/serda/benchmark-run-20261014/cases.jsonl').read_text().splitlines()]
    forbidden=set();ngrams=set()
    def add_forbidden(text):
        x=canon(text);forbidden.add(digest(x));words=x.split();ngrams.update(tuple(words[i:i+13]) for i in range(len(words)-12))
    for r in cases:add_forbidden(r['prompt'])
    # Entire Turkish GSM test is excluded, not only the 200 benchmark sample.
    p=hf_hub_download('malhajar/gsm8k-tr','main/test-1318.jsonl',repo_type='dataset',revision='1046702fc5d8e14330ae90fdfd98735194ffa4ce')
    for line in Path(p).read_text().splitlines():add_forbidden(json.loads(line)['question'])
    for r in cases:
        if r['type']=='mcq':add_forbidden(r['prompt'].split('\n\n',1)[-1].split('\nA.')[0])
    seen=set();counts=Counter();rejected=Counter();tokens=Counter();loss_tokens=Counter();sources=[];reviews=[]
    handles={s:gzip.open(ROOT/f'{s}.jsonl.gz','wt') for s in ['train','validation']}
    pilot=gzip.open(ROOT/'pilot.jsonl.gz','wt');pilot_counts=Counter()
    def accept(msg,source,family,group=None):
        msg=normalize({'messages':msg})
        if not msg:rejected['malformed']+=1;return False
        prompt='\n'.join(m['content'] for m in msg if m['role']=='user');normal=canon(prompt);key=digest(normal)
        if key in seen:rejected['duplicate']+=1;return False
        words=normal.split()
        if key in forbidden or any(tuple(words[i:i+13]) in ngrams for i in range(len(words)-12)):
            rejected['benchmark_overlap']+=1;return False
        for m in msg:
            if m['role']=='assistant':
                w=m['content'].split()
                if len(w)>50:
                    grams=[tuple(w[i:i+6]) for i in range(len(w)-5)]
                    if 1-len(set(grams))/len(grams)>.12:rejected['repetition']+=1;return False
        data=encode(msg,tok)
        if len(data['input_ids'])>cfg['max_length'] or sum(x!=-100 for x in data['labels'])<2:rejected['length']+=1;return False
        seen.add(key);group=group or key
        split='validation' if int(digest(group)[:8],16)%100==0 else 'train'
        data.update(source=source,family=family,prompt_hash=key,group=group)
        handles[split].write(json.dumps(data,ensure_ascii=False)+'\n');counts[f'{source}/{split}']+=1;tokens[split]+=len(data['input_ids']);loss_tokens[split]+=sum(x!=-100 for x in data['labels'])
        if split=='train' and pilot_counts[family]<1400:pilot.write(json.dumps(data,ensure_ascii=False)+'\n');pilot_counts[family]+=1
        if len(reviews)<2 or rng.random()<.0005:reviews.append({'source':source,'family':family,'messages':msg})
        return True
    # Diverse replay only, not the previous four-template synthetic curriculum.
    quota={'AlicanKiraz0/Turkce-Atlas-Instruct':12000,'merve/turkish_instructions':6000,'CohereLabs/aya_dataset':3500,'HuggingFaceH4/ultrachat_200k':5000};used=Counter()
    with gzip.open('/home/serda/sft-run-20261013/train.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);s=r['source']
            if used[s]>=quota.get(s,0):continue
            text=tok.decode(r['input_ids'],skip_special_tokens=False)
            msg=[{'role':role,'content':content.rstrip('\n')} for role,content in re.findall(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>',text,re.S)]
            if accept(msg,'replay/'+s,'general'):used[s]+=1
    print('REPLAY',dict(used),flush=True)
    def record(repo,revision,path,license):sources.append({'repo':repo,'revision':revision,'file':path,'license':license})
    for repo,rev,path,count,lang in [('malhajar/gsm8k-tr','1046702fc5d8e14330ae90fdfd98735194ffa4ce','main/train-7472.jsonl',7472,'tr'),('openai/gsm8k','740312add88f781978c0658806c59bc2815b9866','main/train-00000-of-00001.parquet',6000,'en')]:
        p=hf_hub_download(repo,path,repo_type='dataset',revision=rev);record(repo,rev,path,'MIT')
        rows=[json.loads(l) for l in Path(p).read_text().splitlines()] if path.endswith('.jsonl') else pq.read_table(p).to_pylist();rng.shuffle(rows);n=0
        for r in rows:
            a=re.sub(r'<<[^>]*>>','',r['answer']);q=r['question']+ ('\nKısa çözümü açıkla. Son satır #### sayı biçiminde olsun.' if lang=='tr' else '\nGive a short worked solution. End with #### and the numeric answer.')
            if accept(messages(q,a, SYSTEM if lang=='tr' else 'You are a helpful assistant. Follow the requested format accurately.'),repo,'math'):n+=1
            if n>=count:break
        print('SOURCE',repo,n,flush=True)
    repo='barandinho/turkish-reasoning-distilled-sft';rev='265ebf82fc75b89ec2097ab1dcd4f417fd5dcd39'
    for subset,quota,family in [('webinstruct_verified',26000,'reasoning'),('openthoughts_science_verified',6000,'reasoning'),('acecode_verified',6000,'code')]:
        path=f'data/{subset}/train.jsonl.gz';p=hf_hub_download(repo,path,repo_type='dataset',revision=rev);record(repo,rev,path,'Apache-2.0');n=0
        with gzip.open(p,'rt') as f:
            for line in f:
                r=json.loads(line)
                if not r.get('verified'):continue
                # Use complete user-facing solutions, not private deliberation traces.
                final=r['final_answer'].strip()
                if any(s in final for s in ['<think>','</think>']):continue
                msg=messages(r['user_prompt'],final,r.get('system_prompt') or SYSTEM)
                if accept(msg,repo+'/'+subset,family):n+=1
                if n>=quota:break
        print('SOURCE',subset,n,flush=True)
    repo='HuggingFaceTB/smoltalk';rev='5feaf2fd3ffca7c237fc38d1861bc30365d48ffa'
    for subset,quota,family in [('smol-constraints',9000,'instruction'),('smol-summarize',3500,'reading')]:
        path=f'data/{subset}/train-00000-of-00001.parquet';p=hf_hub_download(repo,path,repo_type='dataset',revision=rev);record(repo,rev,path,'Apache-2.0');n=0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=512):
            rows=batch.to_pylist();rng.shuffle(rows)
            for r in rows:
                msg=r['messages']
                if accept(msg,repo+'/'+subset,family):n+=1
                if n>=quota:break
            if n>=quota:break
        print('SOURCE',subset,n,flush=True)
    for m,f,g in generated(40000):accept(m,'linguai/verified-v2',f,g)
    for h in handles.values():h.close()
    pilot.close()
    # Held-out probes use new values and wording; no external benchmark test target.
    probes=[{'messages':m[:-1],'expected':m[-1]['content'],'family':f} for m,f,g in generated(120,SEED+1,True)]
    (ROOT/'probes.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2))
    (ROOT/'sample-review.json').write_text(json.dumps(reviews[:100],ensure_ascii=False,indent=2))
    manifest={'counts':dict(counts),'input_tokens':dict(tokens),'assistant_tokens':dict(loss_tokens),'rejected':dict(rejected),'sources':sources,'replay_manifest':json.loads(Path('/home/serda/sft-run-20261013/manifest.json').read_text()),'pilot_family_counts':dict(pilot_counts),'max_length':cfg['max_length'],'assistant_only_loss':True,'truncation':False,'decontamination':'NFKC/casefold prompt hash and contiguous 13-word matching against frozen tests, including full GSM8K-TR test; semantic/translation overlap unknown','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.jsonl.gz')}}
    assert sum(v for k,v in counts.items() if k.endswith('/train'))>60000
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('PREPARED',json.dumps(manifest),flush=True)
if __name__=='__main__':main()
