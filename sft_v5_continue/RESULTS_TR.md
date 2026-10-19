# V5 kapsamlı devam eğitimi — nihai sonuçlar

**Eğitim tamamlandı: 1.997/1.997 optimizer güncellemesi, bir epoch.** Final testleri tamamlandı; seçilen model kaydedildi. Instance `56585829`, **18 Ekim 2026 20:39:50 UTC** tarihinde otomatik silindi. 19 Ekim 12:00 UTC kontrolünde açık Vast instance yoktu.

Bu kez tam eğitim, erken pilotta kesilmedi. Kullanıcının isteğiyle aynı 63.896 örneklik veri kullanıldı. Önceki 100 adımdan sonra aynı shuffle sırasındaki 60.696 örnekle 1.897 ek güncelleme yapıldı. Başlangıçta optimizer/FP32 ana ağırlıklar yeniden oluşturuldu; birebir optimizer resume olmadığı önceki belgelerde açıklandı. Sonraki optimizer süreci yeniden başlatılmadı.

## Seçilen checkpoint

- Son adım 1.997: kaynak temelli dev 30/128, doğal math **63/143**, konuşma 15/15.
- Adım 1.600: kaynak temelli dev 28/128, doğal math **69/143**, konuşma 15/15.
- V2 math baselineı 71/143. Daha iyi matematik korunumu nedeniyle **1.600. adım seçildi**. Son adım yalnızca daha çok güncelleme gördüğü için otomatik daha iyi sayılmadı.
- Tam epochun optimizer işi bitmiştir; yayımlanan seçilmiş ağırlıklar 1.600. adıma aittir. Son 1.997. adımın ham ağırlıkları da `checkpoints/stopped` altında korunmuştur. Buradaki `stopped`, tüm güncellemelerden sonraki kontrollü finalizasyon devridir; yarım kalmış optimizer eğitimi anlamına gelmez.

## Sabit regresyon testi — seçilen adım 1.600

| Test | V2 | Yeni aday |
|---|---:|---:|
| IFEval-TR strict | 41/120 (%34,2) | **51/120 (%42,5)** |
| GSM8K-TR | 73/200 (%36,5) | 68/200 (%34,0) |
| GSM8K-TR, 8-shot | 53/200 (%26,5) | 53/200 (%26,5) |
| Türkçe MMLU | 226/900 (%25,1) | **242/900 (%26,9)** |
| Belebele Türkçe | 453/900 (%50,3) | 450/900 (%50,0) |
| Kesilen IFEval yanıtı | 34/120 | 33/120 |

Ayrı tutulan 128 ek matematik sorusunda **V2 35/128 → aday 38/128**. Bu test checkpoint seçildikten sonra açıldı; seçim için kullanılmadı. 15 prosedürel konuşma-hatırlama kontrolünde ikisi de 15/15; genel doğal diyalog başarısı için %100 iddiası değildir.

Türkçe heldout NLL 1,737054 → 1,739809 (yaklaşık **+%0,159**); İngilizce 1,837996 → 1,839722 (**+%0,094**). Aynı manifest ve eşit token sayılarıyla dil korunum sınırları geçti. NLL ile gerçek görev doğruluğu aynı ölçü değildir.

Eski 2.320 koşullu set çok kez görüldüğünden burada regresyon karşılaştırmasıdır; tamamen görülmemiş genel yetenek tahmini diye sunulmaz. Kaynak koruma devindeki 28/128 ile dış IFEval'deki 51/120 ayrı ölçümlerdir.

## Ne iyi, ne eksik?

Yönerge testinde +10/120, MMLU'da +16/900; 8-shot math aynı. Kısa matematikte -5/200, okumada -3/900 var. Ek matematik testi +3/128. Sonuçlar karışıktır; tüm alanlarda üstünlük iddiası yoktur.

**Yanıtı durdurma problemi büyük ölçüde sürüyor:** 120 dış yönerge yanıtından 33'ü üretim sınırına ulaştı. Bu, önceden konmuş ≤12 kesilme hedefini geçmedi. IFEval 51/120 de ≥60 hedefinin altında. Bu iki nedenle `accepted=false`; diğer final tabanları, ayrı math ve dil korunum koşulları geçti.

`accepted=false`, eğitimin yarım kaldığı veya modelin silindiği anlamına gelmez. Aday model hazır ve özel Hub reposunda korunuyor. Üretim V2 otomatik değiştirilmedi. Bu kontrolde yeni eğitim veya kiralama yapılmadı.

## Yedek, maliyet ve doğrulama

- Model: [serda-dev/Jamba2-3B-Turkish-SFT-v5-cont-20261018](https://huggingface.co/serda-dev/Jamba2-3B-Turkish-SFT-v5-cont-20261018), özel.
- Sabit final revision **`7cca33bea32d5d7f745b76fa15fcd224c23fe3a2`**, alt klasör **`checkpoints/final`**.
- Final 10 dosyanın uzak boyut/hashleri yeniden doğrulandı. Büyük ağırlıklar VPSye indirilmedi.
- Final log arşivi revision `8dcb5f8e4962b7c3dd30b69cc8817287301d48f5`, SHA256 `4c034100e1baeee824393cc188f0cb34f532c472c85d47131bf2fb46b0ea5ca0`; yerel arşiv hash doğrulaması geçti.
- Kaydedilmiş 2.320 testin kimlik/girdileri ve sonuç özeti/sayıları doğrulandı. Bu son kontrol yeni inference veya yeniden eğitim değildir.
- İlk V5 pilotu dahil bakiye farkı **yaklaşık $5,25**; son gözlenen bakiye **$11,39**. $8 sınırı aşılmadı. Bu, bakiye farkıdır; fatura kalemi ayrıştırması değildir.
- Normal bitişte model/log doğrulamasından sonra GPU silindi; tamamlanmış VPS supervisor ve watchdog timer devre dışı bırakıldı.
- Sayısal kanıtlar ve receiptler: [results_final](results_final). Ham çalışma günlüğü `/home/serda/sft-v5-cont-run-20261018/OPERATIONS_TR.md` ve özel Hub `run/` alanında.

Eski runnerın ham duraklama/acceptance kayıtları, yeni final inceleme provenansından ayrı korunur. Son karar `acceptance.json`, tüm epochun ve checkpoint seçiminin açıklaması `review_training_provenance.json` içindedir. 1.600 seçimi dış benchmark sonuçlarına göre yapılmadı.
