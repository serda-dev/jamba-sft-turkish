"""Render saved measurements; never load weights or modify the GPU job."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path('/home/serda/benchmark-run-20261014')
NAMES={'jamba_sft':'Jamba Türkçe SFT (3B)', 'qwen25_3b':'Qwen2.5 Instruct (3B)',
       'qwen3_4b_2507':'Qwen3 Instruct 2507 (4B)', 'jamba_cpt':'Jamba Türkçe CPT (3B)',
       'jamba_original':'Orijinal Jamba2 (3B)'}
TASKS={'turkish_mmlu':('Türkçe MMLU',900),'belebele_tr':('Türkçe Belebele',900),
       'gsm8k_tr':('GSM8K-TR kısa yanıt',200),'gsm8k_tr_cot8':('GSM8K-TR 8 örnekli çözüm',200),
       'ifeval_tr_subset':('IFEval-TR alt küme (strict)',120)}


def main():
    data=json.loads((ROOT/'results.json').read_text());results=data['results']
    complete=len(results)==5 and all(all(r.get(t,{}).get('n')==n for t,(_,n) in TASKS.items()) for r in results.values())
    lines=['# Türkçe model yetenek ve hız karşılaştırması', '',
           'Durum: '+('Tüm testler tamamlandı.' if complete else '**Ara sonuç; eksik testler sıralama için kullanılmamalı.**'), '',
           '## Yetenek testleri', '',
           '| Model | MMLU | Belebele | Matematik kısa | Matematik 8 örnekli | Yönerge strict |',
           '|---|---:|---:|---:|---:|---:|']
    order=['turkish_mmlu','belebele_tr','gsm8k_tr','gsm8k_tr_cot8','ifeval_tr_subset']
    for ident,r in results.items():
        cells=[]
        for task in order:
            x=r.get(task);cells.append('—' if not x else f"%{100*x['accuracy']:.1f} ({x['correct']}/{x['n']})")
        lines.append('| '+NAMES[ident]+' | '+' | '.join(cells)+' |')
    if data['manifest'].get('eos_fix'):
        lines+=['','Orijinal/CPT üretim sonuçları, kalıtılan EOS=2 ile sohbet bitişi EOS=519 uyuşmazlığı düzeltilerek yeniden ölçüldü. Her üç yapılandırmanın bitiş tokenları kabul edilir. Bu iki sürümün üretim düzeltmesi A100 SXM4 üzerinde, aynı BF16 backend ve sorularla yapıldı; ilk üç modelin geçerli yanıtları korundu. Donanım/batch farkları küçük sayısal değişiklikler oluşturabilir. İlk hatalı baseline üretim raporu geçersizdir; arşivlenmiştir.','']
    lines+=['','## Hız: RTX 4090, BF16, Transformers SDPA', '',
            '512 giriş tokenı, 128 zorunlu çıkış tokenı, batch=1, ısınma hariç üç tekrarın medyanı.', '',
            '| Model | Parametre (B) | Decode token/sn | Uçtan uca token/sn | İlk token (ms) | VRAM (GiB) | Batch 4 toplam token/sn |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for ident,r in results.items():
        if 'speed' not in r:continue
        speed=r['speed'];x=next(v['median'] for v in speed['batch1'] if v['input_tokens']==512)
        lines.append(f"| {NAMES[ident]} | {speed['parameters']/1e9:.2f} | {x['decode_tokens_s']:.1f} | {x['end_to_end_tokens_s']:.1f} | {x['ttft_ms']:.1f} | {x['peak_vram_gib']:.2f} | {speed['batch4_aggregate_tokens_s']:.1f} |")
    lines+=['','## Yorum ve sınırlar', '',
            '- Qwen3 yaklaşık 4B parametredir; 3B modellerle birebir aynı büyüklükte değildir.',
            '- MMLU, yayımlanmış 900 soruluk alt kümedir. 0-shot seçenek harfi olasılığı kullanılır; makaledeki 5-shot liderlik tablosu yeniden üretilmemiştir.',
            '- Belebele Türkçe okuduğunu anlama ölçer; 900 sorunun tümü kullanıldı.',
            '- Matematikte aynı 200 sorunun iki istem biçimi saklandı. 8 örnekli test, lm-eval GSM8K-CoT örneklerinin Türkçe çevirisini kullanır. Kısa istem Qwen2.5 yanıtlarının 171/200’ünde tek satır ürettirdi; testin isteme duyarlılığı nedeniyle ikinci koşul tüm modellere eklendi.',
            '- IFEval 120 deterministik Türkçeye uygun soruyla sınırlıdır; tüm benchmark skoru değildir. Strict, bir yanıttaki bütün koşulların sağlanmasıdır; olgusal doğruluk ölçmez.',
            '- Her skorun Wilson %95 güven aralığı CSV/JSON dosyalarında bulunur. Özellikle 200 soruluk matematikte yakın farkları kesin üstünlük saymayın.',
            '- SFT verileriyle 2.120 özgün test isteminde tam eşleşme yoktur. Yeni matematik koşulu aynı 200 soruyu kullanır. Anlamsal benzerlik ve ön eğitimdeki olası test verileri dışlanmış değildir.',
            '- Greedy üretim ve BF16 ağırlıklar kullanıldı. Matematik kısa/8 örnekli bütçeleri 512/1.024, yönerge bütçesi 2.048 token. Sınıra dayanan çıktılar aşağıda ayrıca gösterilir.',
            '- Hız ölçümleri bu donanım ve backend için geçerlidir. İndirme/derleme/tokenizasyon/ağ gecikmesi hariçtir; vLLM gibi optimize sunucuların ulaşabileceği azami hız değildir.',
            '- Tokenizer farkları nedeniyle token/sn, aynı Türkçe metin miktarı/sn anlamına gelmez. Ham hız dosyalarında karakter/sn bulunur; zorunlu 128 token testinin ürettiği içerik modeller arasında farklıdır.', '',
            '## Üretim bütçesine dayanan çıktılar ve yönerge ayrıntısı','',
            '| Model | Matematik kısa | Matematik 8 örnekli | Yönerge | Tek tek yönerge başarısı |',
            '|---|---:|---:|---:|---:|']
    for ident,r in results.items():
        caps=[str(r[t]['truncated_outputs'])+'/'+str(r[t]['n']) if t in r else '—' for t in ['gsm8k_tr','gsm8k_tr_cot8','ifeval_tr_subset']]
        ia=r.get('ifeval_tr_subset',{}).get('instruction_accuracy');lines.append('| '+NAMES[ident]+' | '+' | '.join(caps)+(' | — |' if ia is None else f' | %{ia*100:.1f} |'))
    if all(i in results for i in ['jamba_sft','jamba_cpt','jamba_original']):
        lines+=['','## Eğitim aşamalarının etkisi','',
                'Aynı test koşulunda gözlenen yüzde puan farkları; farklı görevler tek bir genel başarı oranında birleştirilmez.','',
                '| Test | CPT − orijinal | SFT − CPT |','|---|---:|---:|']
        for task,(label,n) in TASKS.items():
            if all(results[i].get(task,{}).get('n')==n for i in ['jamba_sft','jamba_cpt','jamba_original']):
                a,b,c=[results[i][task]['accuracy'] for i in ['jamba_sft','jamba_cpt','jamba_original']]
                lines.append(f'| {label} | {(b-c)*100:+.1f} | {(a-b)*100:+.1f} |')
    lines+=['','## Koruma ve maliyet','']
    state=json.loads((ROOT/'supervisor.json').read_text())
    lines.append(f"GPU instance: {state.get('instance_id')}; durum: {state.get('stage')}; saatlik ücret: ${state.get('hourly_rate',0):.4f}.")
    if 'credit' in state:lines.append(f"Team kredi farkı: ${state.get('total_initial_credit',state['initial_credit'])-state['credit']:.2f} (hesaptaki diğer hareketler ve gecikmeli tahakkuk etkileyebilir).")
    lines.append('Ham yanıtlar, test manifesti, hız tekrarları ve loglar özel Hugging Face SFT reposunda `benchmarks/20261014` dizinine yüklenir. Dosyaların hash/ölçümleri doğrulandıktan sonra instance silinir.')
    (ROOT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n')
    with (ROOT/'comparison.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['model','task','correct','n','accuracy','ci95_low','ci95_high','truncated_outputs'])
        for ident,r in results.items():
            for t in TASKS:
                if t in r:
                    x=r[t];writer.writerow([ident,t,x['correct'],x['n'],x['accuracy'],*x['wilson95'],x['truncated_outputs']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    colors={'jamba_sft':'#087f8c','qwen25_3b':'#7952b3','qwen3_4b_2507':'#d9480f','jamba_cpt':'#5c7cfa','jamba_original':'#868e96'}
    for ax,(task,(label,n)) in zip(axes.flat,TASKS.items()):
        ids=[i for i,r in results.items() if task in r]
        values=[results[i][task]['accuracy']*100 for i in ids]
        ax.barh([NAMES[i] for i in ids],values,color=[colors[i] for i in ids])
        for y,i in enumerate(ids):
            x=results[i][task];low,high=x['wilson95'];ax.errorbar(x['accuracy']*100,y,xerr=[[100*(x['accuracy']-low)],[100*(high-x['accuracy'])]],color='black',capsize=3)
        ax.set_xlim(0,100);ax.set_title(label+f' (n={n})');ax.set_xlabel('Başarı (%)');ax.invert_yaxis()
    ax=axes.flat[-1];ids=[i for i,r in results.items() if 'speed' in r]
    values=[next(v['median']['decode_tokens_s'] for v in results[i]['speed']['batch1'] if v['input_tokens']==512) for i in ids]
    ax.barh([NAMES[i] for i in ids],values,color=[colors[i] for i in ids]);ax.invert_yaxis();ax.set_title('RTX 4090, BF16, batch 1');ax.set_xlabel('Decode token/sn')
    fig.savefig(ROOT/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps({'complete':complete,'report':str(ROOT/'REPORT_TR.md')}))


def publish():
    from huggingface_hub import HfApi
    folder=ROOT/'report_publish';folder.mkdir(exist_ok=True)
    for name in ['REPORT_TR.md','comparison.csv','comparison.png','supervisor.json','supervisor.log','overlap_audit.json']:
        shutil.copyfile(ROOT/name,folder/name)
    for name in ['cleanup_receipt.json','prior_cleanup_receipt.json','code_provenance.json','source_hashes.json']:
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,folder/name)
    api=HfApi();repo='serda-dev/Jamba2-3B-Turkish-SFT-20261013';prefix='benchmarks/20261014/report'
    commit=api.upload_folder(repo_id=repo,folder_path=str(folder),path_in_repo=prefix,commit_message='Turkish model comparison report, figures and GPU cleanup record')
    remote={p.rfilename:p for p in api.model_info(repo,revision=commit.oid,files_metadata=True).siblings}
    hashes={}
    for p in folder.iterdir():
        data=p.read_bytes();digest=hashlib.sha256(data).hexdigest();item=remote[prefix+'/'+p.name]
        actual=(item.lfs.sha256 if hasattr(item.lfs,'sha256') else item.lfs['sha256']) if item.lfs else item.blob_id
        wanted=digest if item.lfs else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert actual==wanted and item.size==len(data);hashes[p.name]=digest
    receipt={'verified':True,'repo':repo,'revision':commit.oid,'path':prefix,'sha256':hashes}
    (ROOT/'report_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'report_uploaded':True,'revision':commit.oid}))


if __name__=='__main__':
    main()
    if '--publish' in sys.argv:publish()
