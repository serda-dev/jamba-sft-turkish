from sft_v4.data import make_case,verify,FAMILIES,math_case,strict_json
import pytest
SOURCE={'answer':'Bir kitabı okumadan önce başlık ve içindekiler bölümü incelenir. Okuma sırasında önemli görülen cümleler kısa notlarla kaydedilir. Bölüm sonunda notlar gözden geçirilir ve metnin ana düşüncesi belirlenir. Bu yöntem bilgilerin düzenlenmesine yardımcı olur.','question':'Okuma notları nasıl tutulur?','source':'test','row':0,'group':'g','question_hash':'q','answer_hash':'a'}

def test_all_families_targets_and_wrong_content():
 for i in range(64):
  r=make_case(SOURCE,i,'train');assert verify(r,r['expected'])['correct']
  assert not verify(r,'Yanlış içerik.')['correct']

def test_paragraph_and_per_line_content_checks():
 r=make_case(SOURCE,0,'dev');bad=r['expected'].replace('***','\n\n');assert not verify(r,bad)['format']
 r=make_case(SOURCE,2,'dev');assert not verify(r,'- başka\n- metin\n- geldi')['correct']
 r=make_case(SOURCE,9,'dev');assert not verify(r,r['expected'].replace('not: ','',1))['correct']

def test_json_duplicates_and_wrong_type_rejected():
 with pytest.raises(ValueError):strict_json('{"x":1,"x":1}')
 r=make_case(SOURCE,4,'dev');assert not verify(r,'{"metin":1,"dil":"Türkçe"}')['correct']

def test_derived_math_targets_recomputed():
 for i in range(160):
  r=math_case(i,'dev');a,b,c,d=r['operands'];expected=[a+b*c-d,(a*c-b)//d,a*c+b*d,a-b+d][r['operation']]
  assert expected==r['answer'] and verify(r,r['expected'])['correct']
  assert not verify(r,'#### '+str(expected+1))['correct']


def test_pilot_gate_rejects_retention_regression_and_incomplete_results():
 from sft_v4.runtime import pilot_gate
 prior={'heldout_manifest_sha256':'fixed','scores':{n:{'tokens':100,'mean_nll':1.} for n in ['turkish','english']}}
 newlang={'complete':True,**prior};cfg={'acceptance':{'turkish_nll_max_ratio_to_v2':1.02,'english_nll_max_ratio_to_v2':1.03}}
 baseline={'n':128,'correct':30,'truncated':4};new={'n':128,'correct':43,'truncated':1}
 before={n:{'n':64,'correct':40} for n in ['math','reading']};after={n:{'n':64,'correct':36} for n in ['math','reading']}
 assert pilot_gate(baseline,new,before,after,newlang,prior,cfg)['passed']
 after['math']['correct']=35
 assert not pilot_gate(baseline,new,before,after,newlang,prior,cfg)['passed']
 after['math']={'n':63,'correct':40}
 assert not pilot_gate(baseline,new,before,after,newlang,prior,cfg)['passed']
 after['math']={'n':64,'correct':40};newlang={**newlang,'heldout_manifest_sha256':'changed'}
 assert not pilot_gate(baseline,new,before,after,newlang,prior,cfg)['passed']
