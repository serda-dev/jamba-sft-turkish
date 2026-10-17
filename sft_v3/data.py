"""Fresh grounded Turkish tasks with executable content and format checks."""
import json
import random
import re

TOPICS = [('kütüphane','kitap','okuma'),('müze','eser','kültür'),
 ('bahçe','fidan','doğa'),('atölye','ürün','üretim'),('okul','defter','eğitim'),
 ('kooperatif','paket','dayanışma'),('laboratuvar','örnek','bilim'),
 ('spor merkezi','top','spor'),('tiyatro','bilet','sanat'),('arşiv','belge','tarih'),
 ('barınak','mama','bakım'),('gözlemevi','kayıt','gökyüzü')]
NAMES = ['Deniz','Ceren','Ozan','Elif','Aslı','Yalın','Seda','Eren','Naz','Umut']
FAMILIES = ['paragraphs','ending','placeholders','quotes','frequency','bullets',
 'sections','two_answers','uppercase','lowercase','json','csv','word_budget',
 'forbidden','dialogue','grounded_json','bounded_explanation']


def external_check(row):
    """Conservative subset: one advertised, directly executable constraint."""
    if len(row['constraints']) != 1: return None
    kind=row['constraints'][0];answer=row['messages'][-1]['content'].strip()
    prompt=row['prompt']
    if kind=='punctuation:use no comma': return ',' not in answer
    if kind=='case:in english and lowercase': return answer.islower()
    if kind=='in english and capital': return answer.isupper()
    if kind=='format:use json format':
        try: json.loads(answer);return True
        except ValueError: return False
    if kind=='use quotation': return answer.startswith('"') and answer.endswith('"')
    if kind=='format:title': return bool(re.match(r'^<<[^<>\n]+>>',answer))
    if kind=='specific ending':
        found=re.findall(r'(?:end|finish|conclude)[^\n]*?(?:phrase|sentence|words?|statement)\s*["\u201c\']([^"\u201d\']+)["\u201d\']',prompt,re.I)
        if len(found)==1:return answer.endswith(found[0])
    return None


def verify(r, response):
    """Check both supplied facts and independently parsed output constraints."""
    if not response.strip():
        return False
    for c in r['checks']:
        kind = c['kind']
        if kind == 'contains': ok = all(v in response for v in c['values'])
        elif kind == 'paragraphs': ok = len(re.split(r'\n\s*\n', response.strip())) == c['n']
        elif kind == 'ending': ok = response.rstrip().endswith(c['value'])
        elif kind == 'no_comma': ok = ',' not in response
        elif kind == 'quotes': ok = response.startswith('"') and response.endswith('"') and response.count('"') == 2
        elif kind == 'frequency': ok = len(re.findall(r'(?<!\w)' + re.escape(c['value']) + r'(?!\w)', response)) == c['n']
        elif kind == 'placeholders': ok = re.findall(r'\[[^\[\]\n]+\]',response) == c['values']
        elif kind == 'bullets': ok = len(response.splitlines()) == c['n'] and all(s.startswith('- ') for s in response.splitlines())
        elif kind == 'sections': ok = re.findall(r'^## (.+)$',response,re.M) == c['values']
        elif kind == 'separator': ok = response.count('******') == 1 and len(response.split('******')) == 2
        elif kind == 'uppercase': ok = response.isupper()
        elif kind == 'lowercase': ok = response.islower()
        elif kind == 'json':
            try: ok = json.loads(response) == c['value']
            except (ValueError,TypeError): ok = False
        elif kind == 'lines': ok = response.splitlines() == c['values']
        elif kind == 'words': ok = c['min'] <= len(response.split()) <= c['max']
        elif kind == 'forbidden': ok = all(not re.search(r'(?<!\w)'+re.escape(v)+r'(?!\w)',response,re.I) for v in c['values'])
        else: raise ValueError('Unknown check: '+kind)
        if not ok: return False
    return True


def generated(n, seed, heldout=False):
    rng=random.Random(seed)
    for i in range(n):
        family=FAMILIES[i%len(FAMILIES)];site,item,theme=rng.choice(TOPICS)
        name=rng.choice(NAMES);city=rng.choice(['İzmir','Isparta','Çorum','Edirne','Rize','Van'])
        # Disjoint values and different instruction wording in development.
        code=('D' if heldout else 'T')+str(rng.randrange(100000,999999))
        a=rng.randrange(12,180);b=rng.randrange(2,8);count=rng.randrange(2,5)
        facts=f'{city} kentindeki {site} için {name} sorumludur. Kayıt kodu {code}. Stokta {a} {item} vardır. Yeni teslimatta {b} {item} gelmiştir.'
        rules=[]
        def check(kind,**kw): rules.append({'kind':kind,**kw})
        check('words',min=1,max=120)
        if family=='paragraphs':
            q=f'{count} kısa paragrafla kaydı aktar. Paragrafları bir boş satırla ayır. Başlık veya liste kullanma. İlk paragrafta sorumlu ve kayıt kodu bulunsun.'
            parts=[f'{name} bu kaydın sorumlusudur. Kayıt kodu {code}.',f'Merkez {city} kentindedir. Çalışma alanı {theme}.',f'Stokta {a} {item} vardır.',f'Teslimatta {b} {item} geldi.']
            answer='\n\n'.join(parts[:count]);check('paragraphs',n=count);check('contains',values=[name,code])
        elif family=='ending':
            ending=rng.choice(['Kayıt onaylandı.','İşlem burada tamamlandı.','Bilgi aktarımı bitti.'])
            q=f'Sorumluyu ve kayıt kodunu tek paragrafta bildir. Yanıtın son ifadesi aynen "{ending}" olsun. Sonuna başka şey ekleme.'
            answer=f'Sorumlu {name}. Kayıt kodu {code}. {ending}';check('ending',value=ending);check('paragraphs',n=1);check('contains',values=[name,code])
        elif family=='placeholders':
            fields=[f'[{x}_{code}]' for x in ['sorumlu','telefon','tarih','adres'][:count]]
            q=f'Doldurulacak form için yalnızca şu alanları bu sırayla alt alta yaz: '+', '.join(fields)+'. Alanları doldurma ve açıklama ekleme.'
            answer='\n'.join(fields);check('placeholders',values=fields);check('lines',values=fields)
        elif family=='quotes':
            q='Sorumluyu ve kayıt kodunu belirten kısa cümlenin tamamını çift tırnağa al. Virgül ve açıklama kullanma.'
            answer=f'"{name} {code} kodlu kayıttan sorumludur."';check('quotes');check('no_comma');check('contains',values=[name,code])
        elif family=='frequency':
            q=f'{count} kısa satır yaz. Her satırda "{theme}" kelimesi tam bir kez geçsin. İlk satır kayıt kodunu içersin. Başka başlık ekleme.'
            answer='\n'.join([f'{code} kodlu kayıt {theme} alanındadır.',f'{theme} kaydının sorumlusu {name}.',f'{city} kentindeki kayıt {theme} alanındadır.',f'{theme} kaydındaki stok {a} {item} olarak belirtilmiştir.'][:count])
            check('frequency',value=theme,n=count);check('contains',values=[code])
        elif family=='bullets':
            q=f'Tam {count} madde ile kayıt bilgilerini aktar. Her satır "- " ile başlasın. İlk madde sorumlu ve kodu içersin. Giriş ve sonuç paragrafı ekleme.'
            answer='\n'.join([f'- Sorumlu {name} ve kod {code}.',f'- Merkez {city} kentindedir.',f'- Stokta {a} {item} vardır.',f'- Yeni teslimat {b} {item} içerir.'][:count]);check('bullets',n=count);check('contains',values=[name,code])
        elif family=='sections':
            titles=rng.choice([['Kimlik','Envanter'],['Sorumlu Kaydı','Stok Durumu'],['Kayıt Bilgisi','Teslimat']])
            q=f'Yanıtı yalnızca "## {titles[0]}" ve "## {titles[1]}" başlıklı iki bölüme ayır. İlkinde sorumlu ve kodu ver. İkincisinde mevcut stok ve teslimat sayılarını aktar. Kısa yaz.'
            answer=f'## {titles[0]}\n{name} — {code}.\n\n## {titles[1]}\nMevcut stok {a} {item}. Teslimat {b} {item}.';check('sections',values=titles);check('contains',values=[name,code,str(a),str(b)])
        elif family=='two_answers':
            q='Aynı kaydı anlatan iki farklı kısa yanıt ver. Aralarına yalnızca ****** koy. İkisinde de kod bulunsun. Virgül kullanma.'
            answer=f'{name} {code} kodlu kaydın sorumlusudur.\n******\n{code} kodlu kayıt için sorumlu kişi {name}.';check('separator');check('frequency',value=code,n=2);check('no_comma')
        elif family in ['uppercase','lowercase']:
            text=f'{city} kentindeki {site} için kayıt kodu {code}. Bu çalışma {theme} alanında düzenli bilgi aktarımına yardımcı olur.'
            q='Aşağıdaki metni Türkçe yazım kurallarıyla tamamen '+('BÜYÜK' if family=='uppercase' else 'küçük')+' harfe dönüştür. Yalnızca dönüştürülmüş metni yaz:\n'+text
            answer=text.translate(str.maketrans('iı','İI')).upper() if family=='uppercase' else text.translate(str.maketrans('İI','iı')).lower();check(family);check('lines',values=[answer])
        elif family in ['json','grounded_json']:
            obj={'kod':code,'sorumlu':name,'telefon':None} if family=='json' else {'kod':code,'toplam':a+b}
            q=('Yalnızca kod, sorumlu ve telefon alanlarını içeren JSON nesnesi ver. Verilmeyen telefon null olsun.' if family=='json' else 'Mevcut stok ile teslimat toplamını hesapla. Sadece kod ve toplam alanlarını içeren JSON ver. Sayı sayısal türde olsun.')+' Kod bloğu ve açıklama ekleme.'
            answer=json.dumps(obj,ensure_ascii=False);check('json',value=obj)
        elif family=='csv':
            lines=['kod,sorumlu,stok',f'{code},{name},{a}'];q='Sadece kod,sorumlu,stok başlıklı CSV ver. Bir başlık ve bir veri satırı olsun. Kod bloğu ekleme.';answer='\n'.join(lines);check('lines',values=lines)
        elif family=='word_budget':
            q='Sorumluyu ve kodu içeren en fazla 18 kelimelik tek cümle yaz. Açıklama veya başlık ekleme.';answer=f'{name} {city} kentindeki {site} için {code} kodlu kaydı yönetmektedir.';check('words',min=5,max=18);check('contains',values=[name,code])
        elif family=='forbidden':
            q='Sorumluyu ve kodu bildir. "elbette", "tabii" ve "sonuç" kelimelerini kullanma. Virgülsüz tek kısa cümle yaz.';answer=f'{name} {code} kodlu kayıttan sorumludur.';check('forbidden',values=['elbette','tabii','sonuç']);check('no_comma');check('contains',values=[name,code])
        elif family=='bounded_explanation':
            q='Kaydı temel alan 100 ile 120 kelimelik bir devir notu yaz. Tam iki paragraf kullan. İlk paragraf sorumlu ve kayıt koduyla ilgili olsun. İkincisi stok ve teslimatı anlatsın. Yeni kişi veya sayı uydurma. Son ifade "Devir tamamlandı." olsun.'
            first=f'{name} bu kaydın sorumlusudur ve kaydı devralacak kişi önce {code} kodunu kontrol etmelidir. Merkez {city} kentinde bulunmaktadır. Kayıt bilgilerini aktarırken sorumlu kişinin adını açıkça belirtmek ve kodu değiştirmeden kullanmak gerekir. Bilgilerin hangi kayda ait olduğu böylece anlaşılır. Telefon veya adres verilmediği için bu alanlara ilişkin yeni bilgi eklenmemelidir. Bu kısa not yalnızca verilen bilgileri içerir.'
            second=f'Stok bölümünde mevcut {a} {item} ile yeni teslim edilen {b} {item} ayrı ayrı belirtilmelidir. Mevcut stok ve teslimat aynı kavram değildir. Devir sırasında bu ayrımı korumak kaydın anlaşılmasını kolaylaştırır. Teslimatın geldiği bilgisi verilmiştir ancak teslimat tarihi belirtilmemiştir. Bu nedenle tarih yazmak yerine verilen kayıt bilgileri olduğu gibi aktarılmalıdır. Devir tamamlandı.'
            answer=first+'\n\n'+second;check('words',min=100,max=120);check('paragraphs',n=2);check('ending',value='Devir tamamlandı.');check('contains',values=[name,code,str(a),str(b)])
        else:
            obj={'kod':code,'sorumlu':name};q='Bu kaydı hatırla.';answer='Kaydı not aldım.'
        intro='Verilen kaydı kullanarak isteği yerine getir.' if not heldout else 'Kayıt aşağıdadır. Yanıtını belirtilen biçime uydur.'
        messages=[{'role':'system','content':'Sen yardımsever bir Türkçe asistansın. Kullanıcının koşullarına uyarak kısa ve tamamlanmış yanıt ver.'},{'role':'user','content':intro+'\n'+facts+'\n\n'+q},{'role':'assistant','content':answer}]
        if family=='dialogue':
            messages += [{'role':'user','content':'Önceki kaydı yalnızca kod ve sorumlu alanları olan JSON ile hatırlat. Başka metin yazma.'},{'role':'assistant','content':json.dumps(obj,ensure_ascii=False)}]
            check('json',value=obj)
        row={'messages':messages,'family':family,'group':f'fresh/{"dev" if heldout else "train"}/{seed}/{i}','checks':rules,'source':'linguai/grounded-instructions-v3'}
        assert verify(row,messages[-1]['content']),family
        yield row
