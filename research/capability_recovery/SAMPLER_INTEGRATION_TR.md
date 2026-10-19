# Gerçek eğitim arayüzü ve doğrulanan exposure

Bu arayüz veri hazırlar; eğitim başlatmaz. En son gerçek dry run: **3.000.085 assistant target token, 5.525 batch, en fazla 2 tekrar, bütün ailelerde shortage 0**. Kaynak başına token sayımı ve her accumulation penceresi [receiptte](results/merge_20261019/sampler_dry_run_v4.json) ve [CSV'de](results/merge_20261019/accumulated_step_tokens_v4.csv).

## Kullanım

```python
import json
from pathlib import Path
from research.capability_recovery.sampling import training_loader, optimizer_windows

root = Path('/home/serda/capability-recovery-artifacts/training-pool-v4')
metadata = json.loads((root / 'metadata.json').read_text())
loader, exposure = training_loader(
    root / 'train.jsonl', metadata, token_budget=3_000_000,
    pad_token_id=tokenizer.pad_token_id, micro_batch_size=4,
    max_reuse=2, seed=20261019,
)

# Mevcut runtime'ın denetlenmiş BF16 forward / FP32 master optimizer yoluna bağla.
model.train()
model.config.use_cache = False
model.config.num_logits_to_keep = 0  # Assistant-only full-sequence training loss.
for window in optimizer_windows(loader, accumulation_batches=8):
    # Mevcut optimizer/master runtime burada bir defa gradientleri temizler.
    for model_kwargs, token_weight in window:
        model_kwargs = {k: v.to('cuda') for k, v in model_kwargs.items()}
        loss = model(**model_kwargs).loss
        (loss * token_weight).backward()
    # Mevcut runtime burada master transferi, clip, optimizer step ve sync yapar.
```

Örnekteki micro-batch4, CPU DataLoader dry run arayüzünü gösterir; GPU eğitim belleğine sığdığı ölçülmüş bir training ayarı değildir. Gerçek eğitim batch/context ayarı mevcut runtime memory preflightından geçmelidir.

Bu örnek optimizer precision politikasını değiştirmez. A/B precision araştırmasının üretim yoluna geçiş izni yerine kullanılamaz. `family`, `sources`, target token ve örnek ID metadatası modele gönderilmez. Eski `objective_weight` kullanılmaz; sampling ve eski ağırlıklar üst üste bindirilmez. Son kısa accumulation penceresi kendi gerçek token toplamına göre normalize edilir.

## Veriler ve sınırlar

- 17.000 generic distilled reasoning örneği doğrulanmış matematik diye yeniden etiketlenmez; sınıflandırılmamış olarak dışlanır.
- `grounded_instruction`, kaynak metin düzenleme görevidir; `instruction` olarak kalır. QA kotasını doldurmak için adı değiştirilmez.
- 5.197 yeni `grounded_qa` örneğinde kaynak cevap offseti ve literal destek cümlesi kontrol edilir. Bu bağımsız doğrulanmış serbest akıl yürütme verisi değildir.
- 1.000 yeni Türkçe çok turlu diyalog gerçek insan yazımı görev odaklı kaynaktan gelir. Upstream train kullanılır, test/validation dışlanır. Dry run'da bu kaynak gerçekten 32.246 target token almıştır.
- Truncation uygulanmaz; fazla uzun satırlar reddedilir. Assistant-only etiketler ve her assistant turunun EOS labelı denetlenir.
- Evaluation exact/13-word spanları dışlanır; bunun semantik/çeviri sızıntısının tamamını dışladığı iddia edilmez.
- En dar aile reading, 2.500 örnek/232.763 target token. Reuse 2 ile bu başlangıç kompozisyonu yaklaşık **3,10M toplam token** destekler. Daha yüksek bütçe otomatik tekrar artırarak karşılanmaz.
- Türkçe/İngilizce dil oranı, serbest sohbet/okuma çeşitliliği ve upstream lisans açıklamaları için çalışma devam eder. TQuAD prepared metinleri Git'e yayımlanmamıştır. Bu receipt V6 eğitime hazır kabulü değildir.

## CI

Padding sonrası gerçek shifted label sayımı, deterministik sıra, eksik ailede erken duruş, metadata uyuşmazlığı, kısa son optimizer penceresi, family metadatasının forward'dan çıkarılması ve model merge/scorer/bütçe sözleşmeleri CPU testleriyle kontrol edilir. GitHub'daki başarılı 41-test koşuları [receiptte](results/merge_20261019/ci_success.json). CUDA kalite sonuçları bu testlerle temsil edilmez.
