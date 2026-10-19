"""Summarize completed real generation results without publishing raw responses."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .common import digest, write_json
from .evaluation import wilson
from .merge_experiment import compare_results


def summarize_run(root):
    models = {}
    paths = [root/'v2.json', root/'v5.json'] + [root/f'alpha_{a:.2f}.json' for a in [.15, .30, .50, .70]]
    data = {}
    for p in paths:
        result = json.loads(p.read_text())
        if not result.get('complete') or len(result['records']) != 272:
            raise ValueError('Six complete frozen-suite evaluations required')
        data[p.stem] = result
        records = result['records']
        math = {}
        for label, prefix in [('gsm8k_tr', 'gsm-new/'), ('amc_tr', 'amc/')]:
            selected = [r for r in records if r['id'].startswith(prefix)]
            n = len(selected); correct = sum(r['correct'] is True for r in selected)
            math[label] = {'n': n, 'correct': correct, 'ci95': wilson(correct, n)}
        models[p.stem] = {'summary': result['summary'], 'math_sources': math,
                         'seconds': result['seconds'], 'batch_size': result['batch_size'],
                         'source_file_sha256': digest(p), 'model_receipt': result['model_receipt'],
                         'finite_logits_checked_each_decode_step': result['finite_logits_checked_each_decode_step'],
                         'generated_tokens': sum(r['output_tokens'] for r in records),
                         'truncated_all': sum(r['truncated'] for r in records),
                         'eos_all': sum(r['eos_terminated'] for r in records),
                         'correct_eos_nonrepetitive_all': sum(r['correct'] is True and r['eos_terminated'] and
                                                            r['repetition_8gram'] <= .2 for r in records)}
    hashes = {r['manifest_sha256'] for r in data.values()}
    if len(hashes) != 1:
        raise ValueError('Results do not use the same frozen manifest')
    return {'models': models, 'comparison': compare_results(data), 'manifest_sha256': hashes.pop(),
            'production_accepted': False, 'historical_benchmark_used_for_selection': False}


def markdown(result):
    lines = ['# V2/V5 gerçek merge deneyi — sonuçlar', '',
             'Altı model aynı yeni development setinde, aynı greedy protokol ve örnek başına aynı çıktı bütçesiyle değerlendirildi. Tarihî benchmark checkpoint seçiminde kullanılmadı.', '',
             '| Model | Matematik /64 | GSM-TR /32 | AMC-TR /32 | Yönerge /80 | Çoklu kısıt /64 | Okuma /64 | Konuşma belleği /64 | Cap /272 | EOS /272 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in result['models'].items():
        s = r['summary']; m = r['math_sources']
        cells = [name, s['math']['correct'], m['gsm8k_tr']['correct'], m['amc_tr']['correct'],
                 s['instruction']['correct'], s['multiconstraint']['correct'], s['reading']['correct'],
                 s['dialogue']['correct'], r['truncated_all'], r['eos_all']]
        lines.append('| '+' | '.join(map(str, cells))+' |')
    lines += ['', '## Önceden belirlenen seçim kapısı', '',
              '| Aday | İki ebeveyne karşı tüm ailelerde nonregression | İki ebeveyne karşı pozitif paired CI bulunan aileler |',
              '|---|---|---|']
    for name, c in result['comparison']['comparisons'].items():
        lines.append(f'| {name} | {"Geçti" if c["nonregression_both"] else "Geçmedi"} | '+(', '.join(c['clear_gain_families_both']) or 'Yok')+' |')
    lines += ['', '**Karar:** '+result['comparison']['decision']+'.', '',
              'Üstünlük gösterilememesi bütün interpolasyonların her görevde kötü olduğu anlamına gelmez. Görev bazındaki farklar ve paired kaynak-grubu bootstrap aralıkları makine raporunda korunur. Dört aday/örtüşen ailelerde çoklu karşılaştırma olduğundan keşifsel CI sonuçları düzeltilmiş istatistiksel anlamlılık diye sunulmaz.', '',
              '## Ölçümün sınırları', '',
              '- Yönerge: beyan edilen kısıtların uyumu; olgusal/konusal yanıt doğruluğu değil.',
              '- Matematik: kaynak anahtarına göre sayısal son yanıt. Ara muhakemenin geçerliliği bağımsız doğrulanmadı.',
              '- Okuma: kısa kaynak ifadesinin exact matchi; eş anlamlı uzun yanıtlar false negative olabilir.',
              '- Konuşma: gerçek Türkçe insan diyaloglarında bilgi hatırlama; serbest sohbet akıcılığı puanı değil.',
              '- Sonlandırma: cap/EOS/tekrar/uzunluk ve doğru yanıtla birlikte EOS ayrı tutulur. Zorla kısa cevap genel başarı yerine geçmez.',
              '- Ayrılmış çoklu kısıt havuzu24 örnektir; tam64/32 final kabulü kapalıdır. Coding, English retention ve dil NLL bu merge feasibility deneyinde ölçülmedi. Production kabulü verilmedi.', '',
              '## Gerçek süre', '', '| Model | Saniye | Üretilen token |', '|---|---:|---:|']
    for name, r in result['models'].items():
        lines.append(f'| {name} | {r["seconds"]:.1f} | {r["generated_tokens"]} |')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    result = summarize_run(a.results_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(a.output_dir/'summary.json', result)
    (a.output_dir/'RESULTS_TR.md').write_text(markdown(result))
