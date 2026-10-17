# $2 sınırında yönerge tanı pilotu

## Amaç ve izin

17 Ekim 2026: kullanıcı onayıyla V2 modelinden başlayan tek A100 80 GB tanısı. En fazla 1 saat ve $2 bakiye farkı; 40 dakika veya $1 harcamada yeni adımları durdurma, kalan süreyi log aktarımı için ayırma. Saatlik teklif tavanı $1.35, toplam tahmini transfer dahil $1.60 altında. Üretim eğitimi yapılmaz, deneme ağırlıkları kaydedilmez.

## Hazırlık

32 kısa eğitim ve 128 ayrı tanı örneği: iki paragraf (`***` ve boş satır ayrı), satır başına anahtar kelime, iki farklı yanıt, son ifade, JSON/toplama, Türkçe büyük/küçük harf. İstenen olgular da doğrulanır. Tüm hedefler sözleşmelerden geçer; asistan etiketleri tokenizer ile bire bir çözülerek denetlenir. Eğitim ve tanı promptları ayrıdır, görev şablonları paylaşılır; bu genel başarı benchmarkı değildir. Sekiz CPU testi geçmiştir.

## Üretim yolu kontrolleri

V2 üzerinde aynı 16 yeni soruda batch 1/16, cache açık/kapalı ve nötr sistem mesajı karşılaştırılır. İlk token, ilk logit farkı, ilk iki token marjı, tüm yanıt tokenları, EOS 519, EOS 2 ve uzunluk sınırına ulaşma saklanır. Ek bir kolda EOS 2 ve 519 birlikte kabul edilerek bitiş politikası karşılaştırılır. Küçük BF16 sayısal farklarından doğan farklı yanıt tek başına cache hatası sayılmaz.

## Sayısal güncelleme hipotezi

Eski eğitim tüm parametreleri BF16 saklıyor ve LR 2e-6 kullanıyordu. CPU gösteriminde 0.001, 0.01, 0.03 ve 0.1 büyüklüğünde BF16 sayılara 2e-6 ekleme/çıkarma yuvarlanarak değişimsiz kalabiliyor. Bu gerçek optimizer sorununun kanıtı değildir. Aynı V2 snapshotından bağımsız başlayan üç kol ölçülür:

| Kol | Parametre saklama | Hesaplama | LR | Adım |
|---|---|---|---:|---:|
| bf16_lr2e6 | BF16 | BF16 autocast | 2e-6 | 100 |
| fp32_lr2e6 | FP32 | BF16 autocast | 2e-6 | 100 |
| fp32_lr1e5 | FP32 | BF16 autocast | 1e-5 | 100 |

Her kol full weight AdamW8bit, microbatch 1, accumulation 4, aynı seed/sıra kullanır. İlk adımda gradyan dtype/sıfır olmayan oranı ve örneklenen gerçek parametre değişimi; sonunda toplam değişim, loss, VRAM ve eğitim/tanı başarısı kaydedilir. FP32 saklama ara hesaplamaları da değiştirebilir; fark yalnızca master weight etkisi diye sunulmaz.

## Karar ve korunma

32 eğitim sorusunda >=%90 doğru bir öğrenebilme eşiğidir. 128 tanı sorusuyla aktarım ölçülür. Sonuçlar görülmeden 2–4 bin doğal Türkçe örnekle yeni SFT başlatılmaz. Bağımsız 2 dakikalık watchdog ana izleyicinin yokluğunda GPUyu durdurur. Terminal DONE/FAILED logları HF özel tanı reposuna aktarılır, boyut ve hash doğrulanır, sonra yalnızca bu pilot etiketli instance silinir. Yedek doğrulanmazsa durdurulur ve korunur.

Çalışma günlüğü: `/home/serda/instruction-pilot-20261017/OPERATIONS_TR.md`. Sonuçlar: aynı dizinde `pilot_result.json`, kol raporları ve `log_receipt.json`. Özel Hub: `serda-dev/Jamba2-3B-Turkish-Instruction-Diagnostic-20261017`.

## Kaynaklar

- [bitsandbytes AdamW](https://huggingface.co/docs/bitsandbytes/reference/optim/adamw)
- [Stochastic rounding talebi](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1165)
- [Sabit causal-conv1d sürümünde ayrı girdi/ağırlık dtype dispatch](https://github.com/Dao-AILab/causal-conv1d/blob/v1.5.0/csrc/causal_conv1d.cpp)
