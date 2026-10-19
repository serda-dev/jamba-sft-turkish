# V1–V5 tarihsel denetim

Tarih: 19 Ekim 2026. GitHub varsayılan `main`: `81e96e820de966ecdaa31124437599d6463fff85`. İlgili CPT reposunun varsayılan dalı `master`, revision `3b64e741121493d0ee24acea6e7c92d1950b1435`. Yerel SFT ağacının uzak ağaçla blob karşılaştırmasında yalnız eksik `results/SFT_20261013.json` bulundu; uzak blobdan geri getirildi. Eski belge başlıklarından önce tamamlanmış sonuç/provenans kayıtları esas alındı. Kaynak envanteri: [source_inventory.json](results/source_inventory.json).

## Soy ağacı

```text
AI21 Jamba2-3B, 525c6c8e… (65.536 vocab)
  └─ tokenizer genişletme + CPT, 8b1a9c81… (69.632 vocab)
      └─ V1, seçilen checkpoint-3900; toplam eğitim 3944 adım
          └─ yayımlanmış V1 aliası 089da0e1… üzerinden V2 SFT pilotu120
              └─ yeni optimizer ile V2 tam eğitim6782 → c8ee3888…
                  ├─ V3 → 024e83b9… (reddedildi)
                  ├─ V4-A:100 adım (reddedildi)
                  ├─ V4-B:771 adım; final seçim100 → 7acde389… (reddedildi)
                  └─ V5:100 adım → 0b37b264… (pilot reddedildi)
                      └─ yeni optimizer ile 1897 ek adım
                          ├─ seçilen1600 → 7cca33be… (kabul edilmedi)
                          └─ ham1997 → checkpoints/stopped (korunuyor)
```

V2'nin CPT başlangıçlı karşı pilotu da çalıştı; daha düşük aynı-dev loss nedeniyle **V1 başlangıçlı pilot** seçildi. V3/V4/V5 birbirinin devamı değildir. V1 final receipt revisionı `0b22f86c21610f620826067f573aa5ef2821e9cf`; V2 pilotunda kullanılan yayımlama aliası farklı repoda `089da0e1c0f95932f7610e1d3fec0ad10e66c33c`. İki revisionın shard payload SHA256 beyanları yeniden karşılaştırıldı ve aynı bulundu: [v1_alias_verification.json](results/v1_alias_verification.json). Ağırlıklar yeniden indirilmedi; bu metadata doğrulaması ile pilot seçim/provenansı birbirini destekler.

19 Ekim sonraki sayısal pilotta V2den iki bağımsız4-update diagnostik dal daha oluştu: `precision_existing` ve `precision_fp32_micro`. Bunlar yeni kabul edilmiş SFT sürümü değildir. [Sabit revision/receipt ve sonuçları](PILOT_RESULTS_TR.md) korunur; V2 veya V5 üzerine yazılmadı.

## Reçete ve gerçek veri

Tüm koşular full-weight; QLoRA/LoRA yok. AdamW8bit, optimizer durumlarının biçimidir.

| Koşu | Train satırı | Giriş / assistant tokenı, epoch başına | LR | Micro×accum | Epoch / gerçekleşen adım | Güncelleme yolu |
|---|---:|---:|---:|---:|---|---|
| V1 | 63.083 | 32.549.318 / **17.618.670** | 1e-5 | 4×8 | 2 /3944; seçilen3900 | BF16 parametre, ayrı FP32 master yok |
| V2 | 108.483 | 59.921.811 /30.628.366 | 5e-6 | 2×16, kurtarma sonrası | 2 /6782; öncesinde120 pilot | BF16 parametre, ayrı master yok |
| V3 | 17.000 | 5.185.245 /1.804.048 | 2e-6 | 2×16 | 1 /532 | BF16 parametre, ayrı master yok |
| V4-A | 3.082 | veri:427.348 assistant | 1e-5 | 1×8 | plan2;100 adımda durdu | BF16 working +FP32 master |
| V4-B | aynı3.082 | aynı427.348 assistant | 2e-6 | 1×8 | 2 /771; seçilen100 | BF16 working +FP32 master |
| V5 pilot | 63.896 | 39.468.325 /19.884.747 | 7.5e-7 | 1×32 | plan1;100/1997 | BF16 working +FP32 master |
| V5 devam | aynı63.896 | aynı karışım; ilk3200 örnek tekrar edilmedi | 7.5e-7,20 adım restart ramp | 1×32 | toplam1997/1997; seçilen1600 | yeni FP32 master/optimizer; birebir resume değil |

V1/V2 iki epoch nominal hedef maruziyeti sırasıyla35.237.340 ve61.256.732 assistant tokenıdır. V2 seçilen pilotu ilave maruziyet içerir; pilotun tam hedef-token toplamı burada yeniden hesaplanmadı. Kaydedilmeden kaybolan ilk50 full-run adımı korunmuş son modele ek eğitim olarak sayılmaz. V4 custom loopun son kısmi batch davranışı nedeniyle nominal epoch çarpımını gerçek maruziyetin tam sayımı diye kullanmayın. V5 seçilen1600 checkpointi **51.200 örnek /15.938.479 assistant tokenı** görmüştür; tüm1997 adımın tamamlanması seçilen modelin tam epoch gördüğü anlamına gelmez. Ayrıntı [data_history.json](results/data_history.json).

V1 kaynakları Atlas, merve, Aya, Turkish knowledge, UltraChat ve dört prosedürel aile. V2 bunlara doğal GSM8K-TR/EN, distilled web/science/code, SmolTalk constraints/summary ve28 prosedürel mod ekledi. V3:12.000 prosedürel Türkçe+1000 Tulu İngilizce+4000 eski train tekrarı. V4:2000 kaynak koruma/biçim görevi+717 math+365 genel tekrar. V5:61.896 geniş eski train tekrarı+2000 V4 görevi. Kaynak başına kabul sayısı, revision, lisans ve gzip hashleri makine tarafından okunabilen `data_history.json` içindeki manifestlerde korunur. Bunlar64bin yeni öğretmen yanıtı değildir.

## Token dengesi

V2: genel%22,53; math%6,15; genel reasoning%58,91; code%6,49; instruction%4,70; reading%1,10; prosedürel conversation%0,123. Genel reasoning'i doğrulanmış matematik diye adlandırmak hatalıdır.

V3:12bin Türkçe yeni hedefler%38,18;1000 İngilizce%5,40;4000 tekrar%56,41. Satır sayısıyla denge yanıltıcıdır. V4 assistant tokenı%69,98 kaynak/biçim; math%15,01; genel%15,01.

V5 ham assistant: reasoning%54,57; doğal math%6,18; genel%26,47; code%8,38; reading%1,17; instruction%1,72; grounded instruction%1,50. Global ağırlıklı hedefler25/25/20/15/10/5 iken gerçek batch-normalize katsayı toplamı sırasıyla **26,16/26,18/19,03/13,20/10,21/5,21**. Bu katsayı payları gerçek gradient katkısı değildir.

## Sabit tarihsel regresyon sonuçları

| Model | MMLU /900 | Belebele /900 | Math /200 | 8-shot /200 | IFEval /120 | Kesilen IFEval |
|---|---:|---:|---:|---:|---:|---:|
| Orijinal, EOS düzeltmesi sonrası |205|441|38|40|56|8|
| CPT, EOS düzeltmesi sonrası |178|259|19|6|16|97|
| V1 |233|449|45|25|38|33|
| V2, kabul edilmiş referans |226|453|73|53|41|34|
| V3, reddedilen |228|457|71|49|41|41|
| V4-B, reddedilen |245|447|43|34|57|24|
| V5 devam1600, kabul edilmemiş aday |242|450|68|53|51|33|

V4-A ve V5 ilk pilotta tam dış test çalışmadı; boş benchmarkı sıfır başarı diye yorumlamayın. V4-B final100 seçimi, dev-best600'ün dil korunumunu geçememesinden sonraki fallbacktir. Ham training_provenance600 ile final receipt100 çelişkisini kabul/provenans kayıtları açıklıyor.

Önemli bir beceri kaybı SFT'den **önce CPT aşamasında** zaten görülüyor: original→CPT Belebele441→259, IFEval56→16. Dolayısıyla tüm gerilemeyi V3–V5 veri/optimizerına bağlamak yanlış olur. V2 bazı becerileri geri kazandırıyor; tüm orijinal becerileri geri getirmiş değil. Bu gözlem CPT'nin nedenini tek başına ayrıştırmaz.

V5-V2 eşleşmiş eski math sorularında15 kazanım/20 kayıp; IFEval14 kazanım/4 kayıp. Math farkı−2,5pp, eşleşmiş bootstrap aralığı−8…+3pp. IFEval+8,33pp, aralık+1,67…+15pp. Bu aralıklar aynı görülmüş setin yeniden örneklemesidir; seçim yanlılığını, çoklu karşılaştırmayı ve ön eğitim sızıntısını ortadan kaldırmaz. MMLU/Belebele'nin kaynak/pasaj bağımlılığı tarihsel kayıtta tam grup olarak bulunmadığından burada soru kimliği kullanıldı; kaynak-grup CI diye yorumlanmamalı. Ayrıntı [historical_statistics.json](results/historical_statistics.json).

## Hatalar, düzeltmeler ve kanıt düzeyi

- **Gözlenen/düzeltilen:** original/CPT generation-config EOS2, chat EOS519 uyuşmazlığı; yalnız etkilenen baseline üretimleri yeniden ölçüldü. Eski hatalı raporlar saklandı. V5 tekrar sorunu düzeltmeden sonra da sürüyor.
- **Gözlenen/düzeltilen:** Accelerate FP32 output wrapperın hybrid-cache nesnesini bozması; generation için unwrap düzeltmesi. Cacheyi kapatmak genel kalite çözümü olarak kanıtlanmadı.
- **Gözlenen/düzeltilen:** V2 ilk full-run OOM; iki gerçek optimizer adımıyla micro2/accum16 yolu50,30GiB doğrulandı. Kaybolan50 adım için birebir optimizer resume iddiası yok.
- **Gözlenen/düzeltilen:** final scorer eksik `absl`; eğitim tekrarlanmadan finalizasyon kurtarıldı.
- **Gözlenen/düzeltilen:** V3 paragraf ayırıcı kapsamı ve satır/farklı-yanıt checker açıkları; sonraki checkerlar düzeltildi. V3'ün eski başarısı geriye dönük değiştirilmedi.
- **GPUda ayrıştırılmış mekanizma:** düşük-LR saf BF16 güncellemelerinin kaybolması; kontrollü32train/128 şablon tanısında master yöntemi112/128, saf BF1679/128. Clipping dtype farkı da vardı. Genel yetenek gerilemesinin tüm nedenini açıklamaz.
- **Gözlenen/düzeltilen:** pilot ölçümünde float linspace indeks taşması; int64 örnekleme düzeltmesi. Model eğitiminin tarihsel bugı değildi.
- **Gözlenen:** V5 sıfıra yakın konuşma NLL'sinin küçük göreli artışı faydalı modeli dışlıyordu.15 gerçek prosedürel yanıt V2/1600 için15/15. Politika1600 görüldükten sonra değişti; ön kayıt değildir. Nihai60/120 ve≤12 eşikleri değişmedi.
- **Gözlenen:** V4 NLL fallback sırasında reddedilen aday raporu aynı dosyaya yazılmış; kayıp sayısal rapor uydurulmadı. Sonraki kod ayrı dosya koruyor.

## Açık kanıt boşlukları

Yeni bağımsız doğal TR yönerge/math/diyalog/code ölçümleri eksik; doğal Türkçe çok turlu veri kıt. Ara reasoning doğruluğu, kod çalıştırma, semantic/çeviri sızıntısı ve gerçek task-gradient yönleri ölçülmedi. BF16 mikro birikiminin gerçek Jamba'daki büyüklüğü, çatışmaların nedensel etkisi, model merge kalitesi ve DPO yararı henüz GPU deneyi gerektirir. Düşen loss ve küçük NLL farkları genel beceri kararı değildir. Reddedilmiş V3/V4/V5 üretim modeli olarak sunulamaz.
