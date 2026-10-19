# Veri dengesi ve kalite denetimi

## Yeniden ölçülen eski veri

[data_history.json](results/data_history.json) V1–V5 dosyalarını streaming okuyup satır, gerçek shifted assistant tokenı, kaynak/aile payı, uzunluk histogramı, kopya prompt hashleri, dil beyanı ve mask/EOS hata sayısını yeniden üretir. V1 assistant hedefleri17.618.670 olarak ilk kez bu denetimde tekrar sayıldı. V5'in63.896 train ve870 validation kaydının tüm assistant maskeleri/EOSları bağımsız decode edildi: hata0. Diğer koşularda bu tur token-label/EOS varlık kontrolleri yapıldı; hepsine bağımsız decode yapıldığı iddia edilmez.

V5 giriş39.468.325, hedef19.884.747 token; en uzun3071, limit3072. Gerçek uzunluk histogramı raporda. Eski manifestte3125 multi-turn kayıt olması doğal Türkçe sohbet çeşitliliği anlamına gelmiyor: kaynak/kategori filtresiyle eski `general` içinde doğal TR multi-turn diagnostik dilimi bulunamadı. Türkçe grounded conversation ve İngilizce UltraChat ayrı sayılmalı.

Dil otomatik olarak task ailesi yapılmaz. Tarihsel arşiv birçok satırda language taşımıyor; belirli kaynaklarda İngilizce kaynak-türetilmiş etiket, kalanında`unknown` yazıldı. Bu alanın tamamı dil-detektörüyle doğrulanmış değildir. Kaynak/provenans eksikliği nedeniyle gerçek bütünsel%10–15 EN hedefine ulaşıldığı iddia edilemez.

## Öneri ile ham hedef-token payı

| Aile | Yeni başlangıç hedefi | V5 ham pay / durum |
|---|---:|---|
| Genel Türkçe yardım/doğal konuşma |%25| Genel toplam%26,47; TR/EN ve doğallık ayrı denetlenmeli |
| Doğrulanmış math |%20| Math%6,18; reasoning%54,57 buna otomatik eklenmez |
| Doğal yönerge/çoklu kısıt |%20| Instruction+grounded instruction%3,23; çoğu görev biçimi ayrıca sınıflandırılmalı |
| Okuma/özet/çıkarım |%15|%1,17 |
| Kod/teknik |%10|%8,38; unit-test doğruluğu tümünde bilinmiyor |
| Grounded QA/çıkarım |%10| Ayrı güvenilir family etiketi yok; iki kategoriye çift sayılmaz |

V5 loss ağırlıkları reading12,81, instruction6,20 gibi yüksek çarpanlar içeriyordu. Bu daha çok bağımsız kaynak/örnek üretmez. Batch-normalize gerçek maruziyet [HISTORICAL_AUDIT.md](HISTORICAL_AUDIT.md) içinde. Bütün dataset kopyası yerine düşük-bellek metadata ve sampled-index schedule saklanabilir.

## Uygulanan sampler ve gerçek dry-run

[sampling.py](sampling.py), assistant token açığına göre aile seçer; seed sabit, reuse tavanı2. Aile biterse eksik payı başka aileye sessizce aktarmadan durur. Gerçek sample-index sırası, aile tokenları, unique örnekler ve tekrar sayısı yeniden üretilebilir. Kod sayısal samplerı sağlar; araştırma verisinin sınıflandırılması ve production training entegrasyonu henüz yapılmadı.

Eski etiketlerle3M hedef-token denemesi ayrı grounded_qa ailesi bulunmadığından **910 tokenda durdu**. Bu başarısız dry-run bilerek korunur;910 tokenlık çıktı dengeli eğitim dataseti değildir. Başlangıç hedefindeki%10=300.000 grounded QA tokenı eksik.17.000 genel reasoning kaydı doğrulanmış math etiketiyle geçirilmedi. Daha büyük token bütçeleri reading/instruction ailelerinde de reuse tavanına dayanır.

Diagnostikler:128 eski train örneği precision için; math/reasoning/instruction/reading/code/English retention başına3 kaynak-grup. Diyalog eksik. İngilizce diagnostik general/math/reading içerir; English code eksiktir. [diagnostic_manifest.json](results/diagnostic_manifest.json) ham metin içermez.

## Yeni dev/final havuzunun kalite sınırı

Kaynak revision/dosya SHA256, split hashleri/sayıları ve sızıntı taraması [eval_manifest.json](results/eval_manifest.json) içinde. TQuAD ve MATH-500 kartlarında lisans beyanı doğrulanamadı; lokal araştırma havuzudur, yeniden yayımlama/eğitime ekleme öncesi açıklık gerekir. MBPP CC-BY-4.0 ve ARC CC-BY-SA-4.0 kaynak beyanları; yayıncı etiketi bağımsız semantik doğrulama değildir.

Yeni havuz ağırlıklı İngilizce math/code/knowledge ve Türkçe extractive okumadır. Doğal TR çoklu yönerge, math, diyalog, özetleme/termination kapsamı final kabulü için yetersiz; havuzu büyük göründüğü için yeterli saymıyoruz. Yeni final henüz ölçülmedi. Exact/13-kelime kontrolü tamamlandı; semantic/çeviri near-duplicate denetimi tamamlanmadı. Kaynak yanıtlarının tamamına bağımsız insan incelemesi yapılmadı.

## Sınırlı veri tamamlama ihtiyacı

Önce eksik her aile için ayrı kaynak/metin gruplarında64dev+64final elde et; source-group rubric incelemesi olmadan onları hazır sayma. Türkçe mathte farklı işlem yapıları ve doğal problemler, dialogue'da gerçek takip/hatırlama, yönergede kısa ve uzun yanıt/uyumlu çoklu kısıtlar. Kodda en az bir yürütülebilir test; mathte çözüm ve sonuç ayrı denetim. Yeni öğretmen API harcaması yetkisi yok. Yeni sentetik üretim gerekirse sınırlı plan+ayrı doğrulama/bütçe gerekir. İsim/sayı değiştirerek çeşitlilik iddiası yapılmaz.


## 19 Ekim — gerçek DataLoader exposure düzeltmesi

Önceki910-token duruşu düzeltildi. `grounded_instruction` instruction olarak korunur; gerçek kaynak soru+cevap+literal kanıt cümlesi taşıyan5,197 TQuAD-train örneği `grounded_qa` ailesine eklendi. 17,000 generic distilled reasoning math olarak etiketlenmedi.52,093 benzersiz örnekli havuzda max reuse2 ile **3,000,027** shifted target token gerçek **5,540** padded batch üzerinden sayıldı; bütün ailelerde shortage0; plan/label sayımı eşit. `optimizer_windows` model kwargslarını ve token-normalized accumulation ağırlıklarını verir; hiçbir eğitim başlatmadı.

[Makine receiptı](results/merge_20261019/sampler_dry_run.json), [her accumulation penceresinin gerçek token dağılımı](results/merge_20261019/accumulated_step_tokens.json), [ayrıntılar](MERGE_EXECUTION_TR.md). En dar aile reading:2,500 örnek/232,763 target token; reuse2 ile toplam yaklaşık3.10M bütçe sınırı. Daha büyük dengeli run için gerçek yeni okuma çeşitliliği gerekir. QA evidence extraction kapsamı, dil oranı/serbest sohbet eksikleri ve upstream lisans belirsizliği devam eder;52k örneği tüm V6 veri gereksinimleri tamamlandı diye sunmuyoruz.


### Sonraki v4 hazırlığı

Upstream train bölümünden1,000 insan yazımı Türkçe çok turlu görev diyaloğu eklendi; test/validation ve evaluation overlapları dışlandı. Yeni prepared havuz53,093 örnek. Son gerçek3M dry run **3,000,085 target token /5,525 batch**; yeni insan diyaloğu source'u gerçekte32,246 target token aldı. Ailelerin shortage0 ve reuse<=2 şartı korunuyor. Bu dar görev konuşması kaynağı open-domain serbest sohbetin yeterli çeşitlilikte olduğu iddiası değildir. [En yeni sampler receiptı](results/merge_20261019/sampler_dry_run_v4.json); eski dry run korunur.
