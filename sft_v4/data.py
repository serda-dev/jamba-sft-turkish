"""Grounded natural text edits: explicit formatting plus independent content fidelity."""
import json,re
FAMILIES=['paragraphs_stars','paragraphs_blank','bullets','numbered','json','uppercase','lowercase','ending','sections','keyword_lines','two_texts','word_limit','quoted','comma','conversation','grounded_answer']

def compact(s):return ' '.join(s.split())
def upper(s):return s.translate(str.maketrans('iı','İI')).upper()
def lower(s):return s.translate(str.maketrans('İI','iı')).lower()
def sentences(s):return [p.strip() for p in re.split(r'(?<=[.!?])\s+',compact(s)) if p.strip()]
def strict_json(text):
 def unique(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ValueError('Duplicate JSON key')
   out[k]=v
  return out
 return json.loads(text,object_pairs_hook=unique)

def verify(row,response):
 if row['family']=='math':
  match=re.search(r'####\s*(-?\d+)\s*$',response)
  ok=bool(match and int(match[1])==row['answer']);return {'format':bool(match),'content':ok,'correct':ok}
 if row['family']=='reading':
  ok=response.strip()==row['expected'];return {'format':response.strip() in 'ABCD','content':ok,'correct':ok}
 spec=row['spec'];family=spec['mode'];s=response.strip();fmt=False;body='';want=spec['body']
 try:
  if family in ['paragraphs_stars','paragraphs_blank','grounded_answer']:
   parts=s.split('***') if spec['separator']=='***' else re.split(r'\n\s*\n',s)
   fmt=len(parts)==spec['count'] and all(p.strip() for p in parts);body=compact(' '.join(parts))
  elif family in ['bullets','numbered','keyword_lines']:
   lines=s.splitlines();items=[];fmt=len(lines)==len(spec['items'])
   for i,line in enumerate(lines):
    prefix='- ' if family=='bullets' else f'{i+1}. ' if family=='numbered' else spec['keyword']+': '
    fmt=fmt and line.startswith(prefix)
    items.append(line[len(prefix):] if line.startswith(prefix) else line)
    if family=='keyword_lines':fmt=fmt and len(re.findall(r'(?<!\w)'+re.escape(spec['keyword'])+r'(?!\w)',line))==1
   body=compact(' '.join(items))
  elif family in ['json','conversation']:
   value=strict_json(s);fmt=isinstance(value,dict) and list(sorted(value))==list(sorted(spec['json']))
   ok=json.dumps(value,sort_keys=True,ensure_ascii=False)==json.dumps(spec['json'],sort_keys=True,ensure_ascii=False)
   return {'format':fmt,'content':ok,'correct':fmt and ok}
  elif family in ['uppercase','lowercase','comma','word_limit','quoted']:
   fmt=s==spec['target'];body=s;want=spec['target']
  elif family=='ending':
   fmt=s.endswith(spec['ending']);body=s[:-len(spec['ending'])].rstrip() if fmt else s
  elif family=='sections':
   parts=s.split('\n\n');fmt=len(parts)==2 and all(p.startswith(h+'\n') for p,h in zip(parts,spec['headings']))
   body=compact(' '.join(p[len(h)+1:] if p.startswith(h+'\n') else p for p,h in zip(parts,spec['headings'])))
  elif family=='two_texts':
   parts=s.split('******');fmt=len(parts)==2 and parts[0].strip()!=parts[1].strip() and all(p.strip() for p in parts);body=compact(' '.join(parts))
  else:raise ValueError('Unsupported mode '+family)
  content=compact(body)==compact(want)
  return {'format':bool(fmt),'content':content,'correct':bool(fmt and content)}
 except (ValueError,TypeError,KeyError):return {'format':False,'content':False,'correct':False}

def make_case(source,index,split):
 body=compact(source['answer']);parts=sentences(body);assert len(parts)>=3
 mode=FAMILIES[index%len(FAMILIES)];spec={'mode':mode,'body':body};count=2+((index//16)%2)
 chunks=[' '.join(parts[j*len(parts)//count:(j+1)*len(parts)//count]) for j in range(count)]
 intro=(['Metni aşağıdaki yayın koşullarına göre düzenle.','Bu açıklamayı belirtilen biçimde yeniden sun.','Aşağıdaki metin için bir düzenleme yap.','Metni istenen sunum biçimine getir.'] if split=='train' else ['Yayın öncesi şu düzenlemeyi uygula.','Metni şu biçim kurallarına uygun olarak hazırla.'])[index//16%(4 if split=='train' else 2)]
 task='';answer='';source_text=body
 preserve=' Kaynak cümlelerin sözcüklerini ve sırasını değiştirme. Açıklama veya ek bilgi ekleme.'
 if mode in ['paragraphs_stars','paragraphs_blank','grounded_answer']:
  sep='***' if mode=='paragraphs_stars' or mode=='grounded_answer' else '\n\n';spec.update(separator=sep,count=count)
  task=f'Tam {count} paragraf oluştur. Paragrafları '+('ayrı satırdaki *** ile ayır.' if sep=='***' else 'bir boş satırla ayır.')
  answer=('\n***\n' if sep=='***' else '\n\n').join(chunks)
  if mode=='grounded_answer':task='Verilen açıklamayı okuyucuya aktar. '+task
 elif mode in ['bullets','numbered','keyword_lines']:
  selected=parts[:min(5,len(parts))];body=' '.join(selected);source_text=body;spec['body']=body;spec['items']=selected
  if mode=='bullets':task=f'Her kaynak cümleyi ayrı bir maddeye koy; tam {len(selected)} madde olsun. Madde işareti olarak "- " kullan.';prefix=lambda j:'- '
  elif mode=='numbered':task=f'Kaynak cümleleri 1 ile başlayan numaralı liste yap. Tam {len(selected)} satır olsun; satır biçimi "1. cümle" olsun.';prefix=lambda j:f'{j+1}. '
  else:
   keyword=next(k for k in ['not','bilgi','öneri','hatırlatma','ayrıntı'] if not re.search(r'(?<!\w)'+k+r'(?!\w)',body))
   spec['keyword']=keyword;task=f'Her cümleyi ayrı satıra yaz. Tam {len(selected)} satır olsun ve her satır "{keyword}: " ile başlasın. Her satırda {keyword} sözcüğü tam bir kez bulunsun.';prefix=lambda j:keyword+': '
  answer='\n'.join(prefix(i)+p for i,p in enumerate(selected))
 elif mode in ['json','conversation']:
  key=['metin','aciklama','icerik'][index//16%3];value={key:body,'dil':'Türkçe'};spec['json']=value
  task=f'Yalnızca {key} ve dil alanları olan bir JSON nesnesi ver. {key} alanına kaynak metni aynen koy; dil alanı "Türkçe" olsun. Kod bloğu kullanma.';answer=json.dumps(value,ensure_ascii=False)
 elif mode in ['uppercase','lowercase']:
  source_text=parts[0];spec['body']=source_text
  task='Tüm harfleri Türkçe yazım kurallarına göre '+('BÜYÜK' if mode=='uppercase' else 'küçük')+' harfe çevir. Yalnızca dönüştürülmüş cümleyi ver.'
  answer=upper(source_text) if mode=='uppercase' else lower(source_text);spec['target']=answer;preserve=' Harf dönüşümü dışında metni değiştirme.'
 elif mode=='ending':
  end=['Bilgilendirme tamamlandı.','Metnin sonu.','Düzenleme tamamlandı.','İyi okumalar.'][index//16%4];spec['ending']=end
  task=f'Metni tek paragrafta sun. Sonuna aynen "{end}" ifadesini ekle.';answer=body+' '+end
 elif mode=='sections':
  spec['headings']=['İlk bölüm','İkinci bölüm'];count=2;chunks=[' '.join(parts[:len(parts)//2]),' '.join(parts[len(parts)//2:])]
  task='Tam iki bölüm oluştur. Başlıklar ayrı satırda "İlk bölüm" ve "İkinci bölüm" olsun. Her başlıktan sonra ilgili cümleler gelsin; iki bölümü boş satırla ayır.'
  answer='İlk bölüm\n'+chunks[0]+'\n\nİkinci bölüm\n'+chunks[1]
 elif mode=='two_texts':
  a=' '.join(parts[:len(parts)//2]);b=' '.join(parts[len(parts)//2:]);source_text='Birinci metin:\n'+a+'\n\nİkinci metin:\n'+b
  task='İki kaynak metni ayrı iki yanıt olarak yaz. Yanıtlar arasına yalnızca altı yıldız (******) koy. Birinci metin/ikinci metin başlıklarını yanıta alma.';answer=a+'\n******\n'+b
 elif mode=='word_limit':
  n=min(len(body.split())-3,[18,25,32,40][index//16%4]);task=f'Kaynak metnin ilk {n} sözcüğünü al. Sadece bu sözcükleri tek satırda yaz; başlık ekleme.';answer=' '.join(body.split()[:n]);spec['target']=answer
 elif mode=='quoted':
  task='Kaynak metni tek paragrafta ver ve yanıtın tamamını düz çift tırnak içine al. Tırnak dışında metin yazma.';answer='"'+body+'"';spec['target']=answer
 elif mode=='comma':
  task='Virgülleri noktalı virgüle çevir. Tek paragraf yaz ve bunun dışında hiçbir değişiklik yapma.';answer=body.replace(',',';');spec['target']=answer;preserve=''
 messages=[{'role':'user','content':intro+'\n\nKaynak metin:\n'+source_text+'\n\nİstek:\n'+task+preserve},{'role':'assistant','content':answer}]
 if mode=='conversation':messages=[{'role':'user','content':'Düzenleme sırasında kullanacağımız kaynak metin şu:\n'+body},{'role':'assistant','content':'Kaynak metni aldım.'},{'role':'user','content':task},{'role':'assistant','content':answer}]
 if (index//16)%2:messages.insert(0,{'role':'system','content':['Kullanıcının istediği biçime uyan, doğru ve açık bir Türkçe asistansın.','Türkçe yanıt ver. Verilen kaynak metne sadık kal.'][index//32%2]})
 row={'id':f'{split}/natural/{index}','group':source['group'],'source':source['source'],'source_question_hash':source['question_hash'],'source_answer_hash':source['answer_hash'],'source_row':source['row'],'family':mode,'spec':spec,'messages':messages,'expected':answer,'component':'instruction','verification':'format plus exact source-content preservation; no external factual claims added'}
 assert verify(row,answer)['correct'],row['id']
 return row


def math_case(i,split):
 import random
 rng=random.Random(20261018+i+(100000 if split!='train' else 0));a=rng.randint(31,190);b=rng.randint(4,19);c=rng.randint(3,11);d=rng.randint(2,8);mode=i%4
 if mode==0:q=f'Bir okul gezisine {a} öğrenci katılıyor. Her birinde {b} öğrenci bulunan {c} grup daha ekleniyor. Sonra {d} öğrenci geziye katılmaktan vazgeçiyor. Son durumda kaç öğrenci var?';value=a+b*c-d;steps=f'Eklenen öğrenci sayısı {b} × {c} = {b*c}. İlk grupla birlikte {a} + {b*c} = {a+b*c} öğrenci olur. Vazgeçenler çıkarılınca {a+b*c} − {d} = {value} öğrenci kalır.'
 elif mode==1:q=f'Bir atölye {c} günde, her gün {a} ürün üretiyor. Ürünlerin {b} tanesi kontrolden geçemiyor. Kalanlar {d} kutuya eşit dağıtılıyor. Her kutuya kaç ürün konur? Artan varsa kaçtır?';total=a*c-b;value=total//d;steps=f'Toplam üretim {a} × {c} = {a*c} üründür. Kontrolden geçemeyenler çıkarılınca {a*c} − {b} = {total} ürün kalır. Bölme işlemi {total} = {d} × {value} + {total%d} biçimindedir. Her kutuda {value} ürün bulunur ve {total%d} ürün artar.'
 elif mode==2:q=f'Bir okur {c} gün boyunca her gün {a} sayfa okudu. Sonraki {d} gün boyunca günde {b} sayfa okudu. Toplam kaç sayfa okumuştur?';value=a*c+b*d;steps=f'İlk dönemde okunan sayfa sayısı {a} × {c} = {a*c}. İkinci dönemde okunan sayfa sayısı {b} × {d} = {b*d}. İki dönem birlikte {a*c} + {b*d} = {value} sayfa eder.'
 else:q=f'Bir depoda {a*c} paket var. İlk sevkiyatta {b*c} paket çıkarılıyor. Sonra {d*c} paket geliyor. Kalan paketler {c} araca eşit yükleniyor. Araç başına kaç paket düşer?';value=a-b+d;steps=f'İlk sevkiyat sonrası {a*c} − {b*c} = {(a-b)*c} paket kalır. Yeni paketlerle toplam {(a-b)*c} + {d*c} = {value*c} paket olur. Araç başına {value*c} ÷ {c} = {value} paket düşer.'
 suffix=' Son satıra yalnızca kutu başına sayıyı "#### sayı" biçiminde yaz.' if mode==1 else ' Son satırı "#### sayı" biçiminde yaz.'
 answer=steps+f'\n#### {value}'
 return {'id':f'{split}/math/{i}','family':'math','component':'math','source':'linguai/verified-arithmetic-v4','group':f'math/{split}/{i}','messages':[{'role':'user','content':q+'\nKısa hesap adımlarını göster.'+suffix},{'role':'assistant','content':answer}],'answer':value,'expected':answer,'verification':'integer arithmetic recomputed from operands','operands':[a,b,c,d],'operation':mode}
