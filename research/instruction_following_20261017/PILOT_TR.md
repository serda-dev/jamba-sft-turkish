# $2 sınırında yönerge tanı pilotu

## Amaç ve izin

17 Ekim 2026: kullanıcı onayıyla V2 modelinden başlayan tek A100 80 GB tanısı. En fazla 1 saat ve $2 bakiye farkı; 40 dakika veya $1 harcamada yeni adımları durdurma, kalan süreyi log aktarımı için ayırma. Saatlik teklif tavanı $1.35, toplam tahmini transfer dahil $1.60 altında. Üretim eğitimi yapılmaz, deneme ağırlıkları kaydedilmez.

## Hazırlık

32 kısa eğitim ve 128 ayrı tanı örneği: iki paragraf (`***` ve boş satır ayrı), satır başına anahtar kelime, iki farklı yanıt, son ifade, JSON/toplama, Türkçe büyük/küçük harf. İstenen olgular da doğrulanır. Tüm hedefler sözleşmelerden geçer; asistan etiketleri tokenizer ile bire bir çözülerek denetlenir. Eğitim ve tanı promptları ayrıdır, görev şablonları paylaşılır; bu genel başarı benchmarkı değildir. Sekiz CPU testi geçmiştir.

## Üretim yolu kontrolleri

V2 üzerinde aynı 16 yeni soruda batch 1/16, cache açık/kapalı ve nötr sistem mesajı karşılaştırılır. İlk token, ilk logit farkı, ilk iki token marjı, tüm yanıt tokenları, EOS 519, EOS 2 ve uzunluk sınırına ulaşma saklanır. Ek bir kolda EOS 2 ve 519 birlikte kabul edilerek bitiş politikası karşılaştırılır. Küçük BF16 sayısal farklarından doğan farklı yanıt tek başına cache hatası sayılmaz.

## Sayısal güncelleme hipotezi

Eski eğitim tüm parametreleri BF16 saklıyor ve LR 2e-6 kullanıyordu. CPU gösteriminde 0.001, 0.01, 0.03 ve 0.1 büyüklüğünde BF16 sayılara 2e-6 ekleme/çıkarma yuvarlanarak değişimsiz kalabiliyor. Bu gerçek optimizer sorununun kanıtı değildir. Aynı V2 snapshotından bağımsız başlayan üç kol ölçülür:

| Kol | Working / optimizer ana ağırlığı | Hesaplama | LR | Adım |
|---|---|---|---:|---:|
| bf16_lr2e6 | BF16 / ayrı ana ağırlık yok | BF16 autocast | 2e-6 | 100 |
| fp32_lr2e6 | BF16 / FP32 ana ağırlık | BF16 autocast | 2e-6 | 100 |
| fp32_lr1e5 | BF16 / FP32 ana ağırlık | BF16 autocast | 1e-5 | 100 |

Her kol full weight AdamW8bit, microbatch 1, accumulation 4, aynı seed/sıra kullanır. İlk adımda gradyan dtype/sıfır olmayan oranı ve örneklenen gerçek parametre değişimi; sonunda toplam değişim, loss, VRAM ve eğitim/tanı başarısı kaydedilir. Tüm kollar hem forward hem üretimde BF16 working ağırlık kullanır. FP32 kollarında optimizer ayrı FP32 ana ağırlıkları günceller ve working ağırlıklara kopyalar; küçük güncellemeler ana ağırlıklarda birikir. Working gradyanları BF16dan FP32ye çevrilir, clipping FP32de yapılır. Saf BF16 kolda clipping BF16dır; sayısal karşılaştırmada bu fark da dikkate alınmalıdır.

## Karar ve korunma

32 eğitim sorusunda >=%90 doğru bir öğrenebilme eşiğidir. 128 tanı sorusuyla aktarım ölçülür. Sonuçlar görülmeden 2–4 bin doğal Türkçe örnekle yeni SFT başlatılmaz. Bağımsız 2 dakikalık watchdog ana izleyicinin yokluğunda GPUyu durdurur. Terminal DONE/FAILED logları HF özel tanı reposuna aktarılır, boyut ve hash doğrulanır, sonra yalnızca bu pilot etiketli instance silinir. Yedek doğrulanmazsa durdurulur ve korunur.

Çalışma günlüğü: `/home/serda/instruction-pilot-20261017/OPERATIONS_TR.md`. Sonuçlar: aynı dizinde `pilot_result.json`, kol raporları ve `log_receipt.json`. Özel Hub: `serda-dev/Jamba2-3B-Turkish-Instruction-Diagnostic-20261017`.

## Kaynaklar

- [bitsandbytes AdamW](https://huggingface.co/docs/bitsandbytes/reference/optim/adamw)
- [Stochastic rounding talebi](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1165)
- [Sabit causal-conv1d sürümünde ayrı girdi/ağırlık dtype dispatch](https://github.com/Dao-AILab/causal-conv1d/blob/v1.5.0/csrc/causal_conv1d.cpp)


## İlk deneme kaydı

Instance `56456827` (~$1.08/saat): cache/batch ve baseline tamamlandı (train 7/32, dev 60/128); optimizer öncesinde ölçüm indeksleri CUDA assert ile durdu. Neden, embedding gibi büyük matrislerde float32 linspace son indeksinin yukarı yuvarlanmasıydı. Bu **pilot ölçüm kodunun hatasıdır**, önceki SFT eğitiminin hatası değildir. İndeks hesabı int64 yapıldı ve büyük-matris sınır regresyon testi eklendi. Doğrulanan log revizyonu `8edc34a8ca38b14d42d36dd738735dd0f2baf008`; GPU silindi, ilk maliyet yaklaşık $0.17836.

İkinci deneme `56457754`, A100 80 GB ~$0.81944/saat. İlk başlangıç bakiyesi ve süre sınırı korunuyor, **iki denemenin toplamı** $2 tavanına tabidir. Sekiz tamamlanmış üretim raporu SHA256 manifestiyle sabitlenerek yeniden kullanılıyor; yeni eğitim kolları aynı V2 snapshotından başlıyor.


## İkinci deneme ve çalışma yöntemi düzeltmesi

`56457754` üzerinde saf BF16 kol 100 adımı tamamladı: train **15/32**, dev **79/128**, peak **18.43 GiB**, eğitim **60.93 saniye**. Sonraki tümü FP32 model kolu, Jamba selective-scan değişken B dtype uyumsuzluğunda ilk forwardda durdu. Log revizyonu `d3fc4a74d8c07d7d13e6cd46cab7a9559364ca39` doğrulandı ve instance silindi. O andaki toplam bakiye farkı **$0.25889**.

Kalıcı tanı düzeltmesi: FP32 ana optimizer ağırlıkları, BF16 working model. Böylece Jamba çekirdeğinin zaten çalışan BF16 yolu korunur, küçük optimizer güncellemeleri FP32 ana ağırlıkta birikir. İlk kol ve sekiz baseline raporu hash ile doğrulanıp yeniden kullanılır. CPU testi, 100 adet 2e-6 güncellemenin saf BF16da kaybolurken FP32 ana ağırlıklarda biriktiğini doğruladı; **10 test geçti**. Üçüncü kiralama aynı toplam $2 ve ilk başlangıçtan itibaren 1 saat sınırı içinde yalnızca kalan iki kolu çalıştırır.


## Tamamlanan sonuç — 17 Ekim 2026, 22:26 UTC

**Tanı tamamlandı; FP32 ana ağırlık ile küçük güncellemeleri biriktirme düzeltmesi gerçek GPUda doğrulandı.** Tüm denemeler dahil ölçülen maliyet **$0.45661** (yaklaşık **$0.46**). Üç instance (`56456827`, `56457754`, `56458873`) silindi; canlı instance sayısı sıfır. İş servisi ve watchdog timer kapatıldı.

| Koşul | Eğitim / 32 | Ayrı tanı / 128 | Tanıda kesilen | Peak VRAM |
|---|---:|---:|---:|---:|
| V2, eğitim öncesi | 7 (%21.88) | 60 (%46.88) | 3 | — |
| Saf BF16, LR 2e-6, 100 adım | 15 (%46.88) | 79 (%61.72) | 0 | 18.43 GiB |
| BF16 + FP32 ana ağırlık, LR 2e-6, 100 adım | 32 (%100) | 112 (%87.50) | 0 | 39.75 GiB |
| BF16 + FP32 ana ağırlık, LR 1e-5, 100 adım | 32 (%100) | 128 (%100) | 0 | 39.75 GiB |

Her kol 100 optimizer adımı, 4 örnek/adım, aynı başlangıç, sıra ve seed. Saf BF16/FP32-master eğitim süreleri sırasıyla **60.93 / 60.00 / 57.85 saniye**; kaynak/sunucu yükleme, doğrulama ve üretim süreleri buna dahil değildir. FP32 ana ağırlık uygulaması bu küçük koşuda anlamlı eğitim yavaşlaması yaratmadı.

### Sayısal kanıt

İlk adımdaki örneklenmiş matris gradyanlarının neredeyse tamamı sıfırdan farklı. Ancak saf BF16da 2e-6 güncellemeler ağırlığın temsil aralığının altında kalabiliyor. Her parametre için 2.048 değer ölçüldü; bunlar bütün modelin tam sayımı değildir. Örneğin ilk Mamba `in_proj.weight`:

| Kol | İlk adım optimizer ağırlığında değişen | 100 adım sonunda BF16 working ağırlığında değişen |
|---|---:|---:|
| Saf BF16, 2e-6 | %1.71 | %1.71 |
| FP32 ana ağırlık, 2e-6 | %100 | %21.14 |
| FP32 ana ağırlık, 1e-5 | %100 | %50.34 |

FP32 ana ağırlıktaki küçük adımlar birikince BF16 working ağırlık da değişiyor. Saf BF16da birikim saklanmıyor. `dt_proj` ve feed-forward projeksiyonlarında benzer farklar ölçüldü; tam rapor `PILOT_RESULT.json` içinde. Forward yolları aynı BF16dır; clipping dtype saf BF16 kolunda BF16, master kollarında FP32 olduğundan deney bu küçük sayısal farkı da içerir.

**Bu sonuç, sorunun yalnızca veri olmadığını gösteriyor.** Veri kapsamı ve checker açıkları yine geçerli. Ancak düşük LR + doğrudan BF16 optimizer parametresi birleşimi sonraki eğitime taşınmamalı. Eski tamamlanmış modellerin tüm kayıplarının tek nedenini bu küçük deneyle açıklayamayız.

### Üretim yolları

16 ayrı soruda cache açık batch 1/16: 7/16; cache kapalı: 6/16. İlk tokenlar tüm karşılaştırmalarda aynı; tam token dizileri cache/batch yoluna göre 13–16/16 aynı. Batch farkında ilk logitlerde max 0.53125 sapma var; greedy devamlar farklılaşabiliyor. Nötr system 6/16; EOS [2,519] 7/16 ve bu örneklerde EOS 2 hiç üretilmedi. Cachei kapatmayı veya EOS 2 eklemeyi çözüm olarak destekleyen bir artış ölçülmedi. Bu küçük test bütün olası cache sorunlarını dışlamaz.

### Son kontrol ve sınırlar

- **736 kayıtlı yanıt** sözleşmelerle yeniden puanlandı: **0 fark**.
- Final log arşivi SHA256: `fdd02aee164283ff7cf873c20cc055db3bb097b3a6d7ec6b33c374cc23e244fe`.
- Özel Hub log revizyonu: `02de3aa48e122cff49c748f829503d134b7b69cb`. Hub boyut/hash ve VPS arşiv hash kontrolü geçti.
- **10 CPU testi geçti.** İndeks sınırı ve FP32 ana ağırlık biriktirme regresyonları dahil.
- Bu %100 **genel yönerge başarısı değildir**: 8 basit görev ailesi, train/dev arasında ortak şablonlar var. Yeni üretim modeli veya checkpoint yayımlanmadı; deneme ağırlıkları silindi. V2 korunuyor; sabit IFEval skoru değiştirilmedi.

### Sonraki uygulama

1. Sonraki SFTde BF16 working model + FP32 optimizer ana ağırlıklarını kullan. Kanıtlanan yardımcılar `pilot_runtime.py`: `make_master_parameters`, `copy_gradients_to_master`, `copy_master_to_working`. Tüm modeli FP32 yükleme denemesi bu Jamba sürümünde kernel dtype hatası verdi; o yol kullanılmasın.
2. [Ana rapordaki](REPORT_TR.md) 2–4 bin doğal Türkçe, tüm koşulları doğrulanmış veri paketini hazırla; ayırıcı ve görev kapsamı ile satır/farklı-yanıt checker açıklarını düzelt. Küçük stok-kayıt kalıplarını büyütmek yeterli değil.
3. FP32 ana ağırlıkla **1e-5**, sonraki sınırlı doğal-veri pilotu için güçlü adaydır. Bu tanı matematik/okuma korunumunu ölçmedi; tam eğitim LR seçimi garanti edilmez. İlk 100–150 adımda yeni doğal dev başarısı ve matematik/okuma korunumunu birlikte kontrol et.
4. Bu pilotun 128 tanı örneğini yeni doğal geliştirme seti yerine kullanma; donmuş benchmark yanıtlarını eğitime alma. Büyük eğitime veya DPOya doğrudan geçilmedi.
