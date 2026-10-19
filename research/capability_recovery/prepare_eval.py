"""Prepare new evaluation pools; final data stays outside Git and cannot select weights."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re

from .common import digest, rows, write_json
from .evaluation import group_split, normalize, shingles

SOURCES = [
    {'repo': 'husnu/tquad-v1v2', 'revision': '34d25445b10a49c177bc5fc6179fa260fb74f8c0',
     'file': 'tquad_dev_data_v1-v2.json', 'license': 'not_declared_in_repository; research-only pending review'},
    {'repo': 'HuggingFaceH4/MATH-500', 'revision': '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',
     'file': 'test.jsonl', 'license': 'not_declared_in_card; upstream MATH terms require review'},
    {'repo': 'google-research-datasets/mbpp', 'revision': '4bb6404fdc6cacfda99d4ac4205087b89d32030c',
     'file': 'sanitized/test-00000-of-00001.parquet', 'license': 'CC-BY-4.0'},
    {'repo': 'allenai/ai2_arc', 'revision': '210d026faf9955653af8916fad021475a3f00453',
     'file': 'ARC-Challenge/validation-00000-of-00001.parquet', 'license': 'CC-BY-SA-4.0'},
]


def make_case(identifier, family, language, prompt, scorer, group=None, **kwargs):
    return {'id': identifier, 'family': family, 'language': language,
            'messages': [{'role': 'user', 'content': prompt}],
            'scorer': scorer, 'group': group or identifier,
            'max_new_tokens': 768, **kwargs}


def authored_cases():
    # Individually authored distinct tasks, not a name/number substitution generator.
    # Open responses require human semantic review and never count as auto-correct.
    examples = [
        ('instruction', 'Bir etkinlik ertelendi: 14 Kasım yerine 21 Kasım saat 18.00. Eski tarihi belirtmeden yalnızca yeni tarih ve saati tek satırda yaz.', 'exact', {'answers': ['21 Kasım 18.00', '21 Kasım saat 18.00']}),
        ('instruction', 'Bu kaydı JSON nesnesine dönüştür: ürün adı Kalem, adet 3, teslim edildi hayır. Anahtarlar ad, adet, teslim; teslim değeri boolean olsun. Başka metin yazma.', 'contracts_with_verified_content', {'contracts': [{'kind':'json','value':{'ad':'Kalem','adet':3,'teslim':False}}], 'content_verifier':'exact_json_or_literal'}),
        ('multiconstraint', 'Aşağıdaki üç durumu aynen, her biri ayrı satırda yaz: Başladı, Bekliyor, Bitti. Madde işareti ve açıklama ekleme.', 'exact', {'answers':['Başladı\nBekliyor\nBitti']}),
        ('multiconstraint', 'İki satır yaz. İlk satır yalnızca Merhaba, ikinci satır yalnızca Hoşça kal olsun. Tırnak, başlık veya başka satır olmasın.', 'exact', {'answers':['Merhaba\nHoşça kal']}),
        ('termination', 'Yalnızca şu sözcüğü yaz ve yanıtı bitir: Evet', 'exact', {'answers':['Evet']}),
        ('termination', 'Arkadaşıma kısa bir teşekkür yaz. Son sözlerin İyi ki varsın olsun. İki cümleyi geçme.', 'manual', {'contracts':[{'kind':'endswith','value':'İyi ki varsın'}]}),
        ('termination', 'Acil olmayan bir yazılım güncellemesini neden yedek aldıktan sonra kurmalıyız? Üç gerekçeyi açıklayan 120–170 kelimelik bir yanıt ver. Sadece bir cümleyle geçiştirme.', 'manual', {'contracts':[{'kind':'word_range','value':[120,170]}]}),
        ('math', 'Bir depodaki pirincin beşte ikisi satılıyor. Kalan 84 kilogramın dörtte biri bağışlanıyor. Başlangıçtaki pirinç kaç kilogramdı? Kısa çözüm ver; son satır #### sayı olsun.', 'number', {'answer':'140'}),
        ('math', 'Bir tren 180 kilometreyi giderken saatte 60, dönerken saatte 90 kilometre gidiyor. Duraklama yok. Gidiş dönüşte ortalama hızı kaç kilometre/saattir? Çözümü açıkla; son satır #### sayı olsun.', 'number', {'answer':'72'}),
        ('math', 'Bir ürün önce yüzde 20 indirim görüyor, sonra indirimli fiyatına yüzde 25 zam geliyor. İlk fiyatı 240 liraysa son fiyatı kaç liradır? İşlemleri göster; son satır #### sayı olsun.', 'number', {'answer':'240'}),
        ('math', 'Bir kutuda 4 kırmızı ve 6 mavi top var. Yerine koymadan iki top çekiliyor. İkisinin de kırmızı olma olasılığını sade kesir olarak yaz; kısa açıklama ekle.', 'manual', {'reference':'4/10 * 3/9 = 2/15'}),
        ('math', 'Aynı işi Ali 6 saatte, Ece 3 saatte yapıyor. Birlikte bir saat çalıştıktan sonra Ali ayrılıyor. Ece kalan işi kaç saatte bitirir? Son satır #### sayı olsun.', 'number', {'answer':'1.5'}),
        ('math', 'Bir sınıftaki kız sayısı erkeklerin iki katıdır. Altı kız ayrılınca kız ve erkek sayıları eşit oluyor. İlk başta sınıfta kaç kişi vardı? Son satır #### sayı olsun.', 'number', {'answer':'18'}),
        ('reading', 'Duyuru: Kütüphane pazartesi bakım nedeniyle kapalıdır. Salı 10.00–16.00 arası açıktır. İade kutusu her gün kullanılabilir. Pazartesi kitabını bırakmak isteyen biri hangi hizmeti kullanabilir? Yalnızca hizmetin adını yaz.', 'exact', {'answers':['İade kutusu','İade kutusu.']}),
        ('reading', 'Rapora göre 2023 yılında 120, 2024 yılında 150 başvuru kabul edildi. Ancak toplam başvuru sayıları açıklanmadı. Bu bilgiden kabul oranının arttığı sonucunu çıkarabilir miyiz? Gerekçeni açıkla.', 'manual', {'reference':'Hayır, paydalar bilinmiyor; kabul sayısı ile kabul oranı farklıdır.'}),
        ('summary', 'Şu toplantı notunu yöneticinin okuyacağı iki cümleyle özetle: Prototip hazır. Güvenlik testi bitmediği için yayın cuma yerine salıya ertelendi. Tasarım ekibi görselleri pazartesi teslim edecek. Müşteri bilgilendirildi. Yeni bir tarih veya sonuç uydurma.', 'manual', {'reference':'Erteleme nedeni, salı yayın hedefi ve pazartesi görsel teslimi kaynakla tutarlı korunmalı.'}),
        ('summary', 'Şu e-postadan yalnızca teslim tarihini ve sorumlu kişiyi JSON olarak çıkar: Merhaba, sunumun ilk taslağını Derya 6 Aralık akşamına kadar gönderecek. Son kontrolü daha sonra yapacağız. Anahtarlar tarih ve sorumlu olsun.', 'contracts_with_verified_content', {'contracts':[{'kind':'json','value':{'tarih':'6 Aralık','sorumlu':'Derya'}}], 'content_verifier':'exact_json_or_literal'}),
        ('grounded_qa', 'Stok kaydı: A deposu 12, B deposu 7, C deposu bilinmiyor. C deposunda kaç ürün bulunduğunu yalnızca bu kayda dayanarak söyle. Kesin bir sayı uydurma.', 'manual', {'reference':'Bilinmiyor; kayıtta C için sayı verilmemiştir.'}),
        ('dialogue', 'Yeni başlayan bir çalışma arkadaşıma hata yaptığını kırmadan nasıl söyleyebilirim? Kısa bir konuşma örneği ver.', 'manual', {}),
        ('dialogue', 'Bir arkadaşım mesajıma dönmedi. Kesin olarak bana kızdığını söyleyebilir misin? Varsayımlarını gerçek gibi sunmadan yanıtla.', 'manual', {}),
        ('code', 'Python fonksiyonundaki hatayı düzelt ve yalnızca düzeltilmiş fonksiyonu yaz: def first(xs):\n    if len(xs) >= 0:\n        return xs[0]\n    return None\nBoş listede None dönmeli.', 'code', {'tests':['assert first([]) is None','assert first([4,5]) == 4']}),
        ('knowledge', 'Bir ışık yılı zaman mı yoksa uzaklık birimi midir? Yalnızca Uzaklık veya Zaman yaz.', 'exact', {'answers':['Uzaklık']}),
        ('english_retention', 'Explain why correlation alone does not establish causation. Give one concrete example and one possible confounder.', 'manual', {}),
        ('english_retention', 'Return only a JSON array containing the even numbers strictly between 2 and 10, in ascending order.', 'contracts_with_verified_content', {'contracts':[{'kind':'json','value':[4,6,8]}], 'content_verifier':'exact_json_or_literal'}),
    ]
    cases = [make_case(f'authored/{i}', family, 'en' if family == 'english_retention' else 'tr', prompt, scorer,
                       source='project-authored-20261019', **kw)
             for i, (family, prompt, scorer, kw) in enumerate(examples)]
    multi = make_case('authored/dialogue-memory', 'dialogue', 'tr', '', 'exact', answers=['Çarşamba'])
    multi['messages'] = [{'role':'user','content':'Toplantım salıydı ama çarşambaya ertelendi. Bunu aklında tut.'},
                         {'role':'assistant','content':'Toplantının çarşambaya ertelendiğini not ettim.'},
                         {'role':'user','content':'Bu arada sunum dosyasını da hazırladım.'},
                         {'role':'assistant','content':'Sunum dosyan hazır.'},
                         {'role':'user','content':'Toplantım hangi gün? Yalnızca gün adını yaz.'}]
    cases.append(multi)
    return cases


def fetch_pools():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    cases, receipts = authored_cases(), []
    for source in SOURCES:
        file = Path(hf_hub_download(source['repo'], source['file'], repo_type='dataset', revision=source['revision']))
        receipts.append({**source, 'sha256': digest(file), 'bytes': file.stat().st_size})
        if source['repo'].startswith('husnu/'):
            docs = json.loads(file.read_text())['data']
            for doc in docs:
                for paragraph in doc['paragraphs']:
                    context = paragraph['context']
                    group = 'tquad/' + hashlib.sha256(normalize(context).encode()).hexdigest()
                    for qa in paragraph['qas'][:2]:
                        answers = sorted({a['text'] for a in qa.get('answers', []) if a['text'] in context})
                        if not answers:
                            continue
                        prompt = f'Bağlam:\n{context}\n\nSoru: {qa["question"]}\nYanıtı yalnızca bağlamdaki ilgili kısa ifadeyle ver.'
                        identifier = 'tquad/' + hashlib.sha256((group+'\n'+normalize(qa['question'])).encode()).hexdigest()
                        cases.append(make_case(identifier, 'reading', 'tr', prompt, 'exact', group=group,
                                               answers=answers, source=source['repo']))
        elif 'MATH-500' in source['repo']:
            for row in rows(file):
                cases.append(make_case('math500/'+row['unique_id'], 'math', 'en',
                    row['problem']+'\nExplain your solution and put the final answer in \\boxed{}.',
                    'boxed_exact', answer=row['answer'], reference=row['solution'], source=source['repo'],
                    semantic_scope='Exact boxed-string final answer only; symbolic equivalents may be false negatives; reasoning unverified'))
        elif source['repo'].endswith('/mbpp'):
            for row in pd.read_parquet(file).to_dict('records'):
                cases.append(make_case('mbpp/'+str(row['task_id']), 'code', 'en',
                    row['prompt']+'\nReturn only the complete Python function and necessary imports.', 'code',
                    tests=list(row['test_list']), test_imports=list(row['test_imports']),
                    reference=row['code'], source=source['repo']))
        else:
            for row in pd.read_parquet(file).to_dict('records'):
                choices = row['choices']
                answers = dict(zip(choices['label'], choices['text']))
                prompt = row['question']+'\n'+'\n'.join(k+') '+v for k,v in answers.items())+'\nReturn only the answer label.'
                cases.append(make_case('arc/'+row['id'], 'knowledge', 'en', prompt, 'exact',
                                       answers=[row['answerKey']], source=source['repo']))
    unique = {}
    for c in cases:
        if c['id'] not in unique:
            unique[c['id']] = c
        elif c['messages'] != unique[c['id']]['messages']:
            raise ValueError('Conflicting duplicate source identity')
        elif 'answers' in c:
            unique[c['id']]['answers'] = sorted(set(unique[c['id']]['answers']) | set(c['answers']))
    receipts.append({'source_id_duplicates_removed': len(cases)-len(unique)})
    cases = list(unique.values())
    rng = random.Random(20261019)
    rng.shuffle(cases)
    counts = Counter()
    bounded = []
    for case in cases:
        if case['id'].startswith('authored/') or counts[case['family']] < 240:
            bounded.append(case)
            counts[case['family']] += 1
    return bounded, receipts


def exclude_historical(cases, training_files, tokenizer, historical_cases=None):
    index = defaultdict(set)
    exact = defaultdict(set)
    for i, case in enumerate(cases):
        text = '\n'.join(m['content'] for m in case['messages'])
        for view in [text, case.get('source_text', ''), case.get('parallel_text', '')]:
            if not view:
                continue
            exact[normalize(view)].add(i)
            for span in shingles(view):
                index[span].add(i)
    rejected = set()
    checked = 0
    historical_checked = 0
    if historical_cases:
        for case in rows(historical_cases):
            text = case['prompt']
            rejected.update(exact.get(normalize(text), set()))
            for span in shingles(text):
                rejected.update(index.get(span, set()))
            historical_checked += 1
    for path in training_files:
        for row in rows(path):
            # Match full training input, including supervised answers: a prior answer
            # containing a new evaluation passage is also potential contamination.
            text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
            # Full chat decoding includes role names even when special tokens are
            # stripped. Exact matching must also inspect message contents so short
            # (<13 word) repeated user prompts cannot slip through.
            contents = re.findall(r'<\|im_start\|>(?:system|user|assistant)\n(.*?)<\|im_end\|>', text, re.S)
            for view in [text, *contents]:
                rejected.update(exact.get(normalize(view), set()))
            for span in shingles(text):
                rejected.update(index.get(span, set()))
            checked += 1
        print('leakage_checked', path.name, checked, flush=True)
    rejected_groups = {cases[i]['group'] for i in rejected}
    return [c for c in cases if c['group'] not in rejected_groups], {
        'training_rows_checked': checked, 'historical_conditions_checked': historical_checked,
        'historical_cases_sha256': digest(historical_cases) if historical_cases else None,
        'case_matches': len(rejected),
        'groups_removed': len(rejected_groups), 'semantic_or_translation_leakage_excluded': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--training-files', type=Path, nargs='+', required=True)
    p.add_argument('--historical-cases', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit('New output directory required; do not overwrite sealed data')
    args.output_dir.mkdir(parents=True, mode=0o700)
    cases, sources = fetch_pools()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cases, leakage = exclude_historical(cases, args.training_files, tok, args.historical_cases)
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate evaluation case IDs')
    assignment = group_split(cases)
    manifest = {'seed': 20261019, 'sources': sources, 'leakage': leakage,
                'training_hashes': {str(p.name)+'/'+str(i): digest(p) for i,p in enumerate(args.training_files)},
                'splits': {}, 'final_ready': False,
                'readiness_blockers': ['Insufficient natural Turkish math/instruction/dialogue/summary groups',
                    'Open-response semantic review pending', 'Reasoning validity scoring unavailable',
                    'MATH-500 exact-string equivalence scorer is incomplete', 'License review pending for TQuAD',
                    'Translation/semantic cross-source deduplication incomplete'],
                'scope': 'New evaluation pools, partly public datasets and partly project-authored smoke tasks; not a fully qualified final suite'}
    for split in ['dev', 'final']:
        selected = [{**c, 'group': assignment[c['id']][1]} for c in cases if assignment[c['id']][0] == split]
        path = args.output_dir/(split+'.jsonl')
        path.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in selected))
        os.chmod(path, 0o600)
        manifest['splits'][split] = {'file': path.name, 'sha256': digest(path), 'n': len(selected),
            'families': dict(Counter(c['family'] for c in selected)),
            'languages': dict(Counter(c['language'] for c in selected)),
            'source_groups': len({c['group'] for c in selected}),
            'manual_semantic_cases': sum(c['scorer'] in ['manual','code'] for c in selected)}
    write_json(args.output_dir/'manifest.json', manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
