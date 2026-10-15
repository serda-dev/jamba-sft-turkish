"""Guard executable data labels before spending GPU credit."""
import re,json
from collections import Counter
from sft_v2.prepare import generated

def test_procedural_targets_and_choice_balance():
 counts=Counter()
 for msgs,family,group in generated(2800):
  q=msgs[-2]['content'];a=msgs[-1]['content']
  if family=='reading':
   options=dict(re.findall(r'^([A-E])\. (.*)$',q,re.M));assert len(options)==5 and a in options;counts[a]+=1
   if 'hangi şehirde' in q:
    assert options[a] in re.search(r'\n\n(.*?)\n\n',q,re.S)[1]
  if family=='math':
   assert len(re.findall(r'#### -?\d+',a))==1
   if 'kalemle birlikte' in q:
    b,c,v=map(int,re.search(r'her birinde (\d+) kalem bulunan (\d+) kutu aldı. Elindeki (\d+)',q).groups());assert int(a.rsplit(' ',1)[-1])==b*c+v
   if 'kişiye' in q:assert int(a.rsplit(' ',1)[-1])>=0
  if 'JSON' in q:assert isinstance(json.loads(a),dict)
  if 'tam iki cümle' in q:assert a.count('.')==2
  if 'sözcükten' in q:assert len(a.split())==int(re.search(r'tam (\d+)',q)[1])
 assert counts==dict.fromkeys('ABCDE',80)

def test_probe_inputs_do_not_repeat_training_inputs():
 train={'\n'.join(m['content'] for m in msgs if m['role']=='user') for msgs,_,_ in generated(40000)}
 probes={'\n'.join(m['content'] for m in msgs if m['role']=='user') for msgs,_,_ in generated(120,20261016,True)}
 assert not train&probes

def test_probe_gold_and_semantically_equivalent_prose():
 from sft_v2.probe_score import check
 for msgs,family,group in generated(280):
  assert check({'messages':msgs[:-1],'expected':msgs[-1]['content'],'family':family},msgs[-1]['content'])
 row={'messages':[{'role':'user','content':'Okuma hakkında tam 5 sözcükten oluşan tek satırlık bir ifade yaz. Noktalama kullanma.'}],'family':'instruction','expected':'bugün birlikte yeni bir kitap'}
 assert check(row,'kitap okumak bize yeni kapılar')
 assert not check(row,'kitap okumak')
