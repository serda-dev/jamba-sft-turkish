# V2/V5 merge ve gerçek token sampler çalışması — 19 Ekim 2026

## Yetki ve kapsam

Kullanıcı bu deney için açıkça **en fazla 5 USD / 4 saat** onayladı. Yetki iki ebeveyn, alpha 0.15/0.30/0.50/0.70 ve uygun adayın kaydı/ayrılmış doğrulaması içindir. V6, DPO, PCGrad veya production değişimi bu çalışmada yoktur.

Önceki 2 USD precision pilotunun yetkisi kullanılmıyor. Yeni GPU için farklı label, state dizini, service ve bağımsız watchdog var. Watchdog 3 saat 45 dakikada veya konservatif harcama 4.70 USD olduğunda kapatır; 0.30 USD transfer/hesap gecikmesi payı bırakılır. Saatlik teklif tavanı 1.10 USD. Kaynak checkpointler değişmez. Yedek/hash kontrolü yapılır; hard bütçe kapanışı yedekleme hatası yüzünden süresiz ertelenmez.

## Geliştirme verisi ve skor kapsamı

- IFEval-TR: Eski 120 sorunun tüm paralel `key` kimlikleri çıkarılır. Tarihî puanlayıcı değişmez. Yeni puanlayıcı yalnız argümanları belirli kontrolleri destekler; rastgele parametre fallbacki reddedilir. Skor **yönerge sözleşmesi uyumudur**, olgusal doğruluk değildir.
- Çoklu kısıt: Birden fazla orijinal kısıt içeren gerçek IFEval soruları; tek bir şablonun isim/sayı çeşitleri üretilmez. Yönerge ailesiyle örtüşür.
- Türkçe matematik: AMC kaynağının Türkçe çevirisi, sayısal cevap anahtarı ve İngilizce paraleli. Görsel gerektiren veya desteklenmeyen cevaplar elenir. Ara çözümün doğruluğu bağımsız olarak puanlanmıyor.
- Okuma: TQuAD gerçek pasaj ve soru; yanıt kaynakta bulunmalı. Kaynak pasajı grup kimliğine dahildir.
- Konuşma: Multi3WOZ insan diyaloglarının Türkçe geçmişinde, kullanıcının gerçekten söylediği literal slot bilgisini hatırlama. Sonraki slot güncellemeleri belirsizlik yaratırsa örnek elenir. Bu bir **konuşma belleği testi**; serbest sohbet akıcılığı veya empati doğrulaması değildir.
- Sonlandırma: Aynı bütün üretimlerde EOS, cap, tekrar ve doğru içerikle birlikte EOS ölçülür. Yönerge/matematik/okuma/konuşma ile örtüşür; aile sayıları toplanmaz.

Her development ailesi en az **64 puanlanan örnek / 32 birleştirilmiş kaynak grubu** sağlamalı. Eski eğitimler ve tarihî benchmark exact/13-word span taraması; matematik için kaynak İngilizcesi de taranır. Bu işlem semantik/çeviri sızıntısının matematiksel olarak tamamen dışlandığı anlamına gelmez. Kaynak revizyonları ve ham dosya hashleri manifesttedir. Ham development/final metinleri Git'e konmaz. Kaynağında lisans açıklaması olmayan TQuAD/AMC için redistribution kabulü yapılmaz; yerel araştırma kapsamı kaydedilir.

## Önceden belirlenen karar

Aynı manifest, tokenizer, greedy decoding, batch 16 ve örneğe göre önceden belirlenmiş aynı çıktı token bütçeleri kullanılır. Dört alpha iki ebeveynle ayrı ayrı karşılaştırılır. Aile başına en fazla 3 puan düşüş ve paired source-group bootstrap CI alt sınırı en az -8 puan şartı korunur. Matematik veya okuma kaybı yönerge artışıyla gizlenmez.

Üstün aday demek için bütün ailelerde iki ebeveyne karşı nonregression ve en az aynı bir ailede iki ebeveyne karşı pozitif CI alt sınırı aranır. Birden fazla aday olursa önceden belirlenen tie-break küçük alphadır. Çoklu karşılaştırmalar nedeniyle bu keşifsel development bulgusudur; genel yetenek kazanımı kanıtı değildir. Üstünlük yoksa sonuç `no_demonstrated_superior_merge` olur; bütün adayların kötü olduğu veya merge'in hiçbir zaman çalışmayacağı iddia edilmez.

Başarılı dev adayının FP32 merge ile oluşturulan BF16 checkpointi private Hub'a tam hash doğrulamasıyla kaydedilir. Seçim dondurulduktan sonra ayrılmış örneklerde yalnız confirmation tanısı yapılabilir. Eksik aileler, serbest konuşma/semantik değerlendirme ve TR/EN NLL yüzünden `final_ready=false` kaldığı sürece production kabulü verilemez; eski 11 ailelik final eşikleri değişmez.

## Sampler'ın gerçek eğitim arayüzü

`prepare_training_pool` eski encoded veriyi okur; `grounded_instruction` yalnız instruction olur, genel distilled reasoning doğrulanmış math olmaz. Eksik `grounded_qa` için TQuAD **train** kaynağından soru + kaynak cevap + literal destek cümlesi eklenir; evaluation pasaj/spanları dışlanır. Kaynak cevabın offseti doğrulanır, hedef yanıtlar kesilmez. Eski objective ağırlıkları kaldırılır; exposure sampling ile loss weighting üst üste bindirilmez.

`training_loader` gerçek PyTorch DataLoader döndürür. `input_ids`, `labels`, `attention_mask` doğrudan modele verilebilir. Family, target token ve sample ID alanları exposure kayıtlarıdır. Accumulation penceresinde micro loss, o microbatchin shifted assistant token sayısının tüm pencereye oranıyla ağırlanır. Eksik aile/capacity varsa eğitim başlamadan hata verir. Dry run 3 milyon gerçek shifted target tokenı DataLoader'dan sayar ve planla eşitliğini doğrular. Maksimum tekrar 2. Dil hedefleri/serbest sohbet çeşitliliği için ayrıca audit gerekir; bu hazırlık eğitim onayı değildir.

## Çalıştırma

```bash
python -m research.capability_recovery.prepare_merge_eval --help
python -m research.capability_recovery.prepare_training_pool --help
python -m research.capability_recovery.merge_experiment --help
python -m pytest research/capability_recovery/tests -q
```

CI: `.github/workflows/capability-recovery.yml`; CPU tensor/merge/scorer/token budget/gerçek DataLoader ve bütçe sınır testleri. CUDA çalışması CPU sonuçlarıyla temsil edilmez.

## İşlem kaydı

- İlk veri hazırlama denemesinde CLI path tipi hatası görüldü, `Path` dönüşümü düzeltildi.
- İkinci denemede parquet içindeki nested NumPy dizileri JSON'a çevrilemiyordu; recursive JSON-safe dönüşüm eklendi. İki başarısız klasör korunuyor.
- CPU contract testleri şu an **41/41 geçti**. Gerçek 3M-token dry run ve GPU sonuçları tamamlandığında aşağıya makine makbuzları eklenecek.

### GPU başlangıcı

- Instance **56745476**, A100 SXM4 80GB, saatlik **1.0674074074 USD**.
- Dev: **272 benzersiz örnek / 272 grup**; yönerge80, bunun içinde çoklu kısıt64; math64 (32 AMC +32 GSM8K-TR); okuma64; gerçek konuşmada bellek64. Sonlandırma272; örtüşen aileler toplanmaz.
- Kaynak 255,544 tarihî train satırı ve 2,320 benchmark koşuluyla tarandı. Eski IFEval paralel IDleri ve V5 eski math probları ayrıca dışlandı.
- 17 GSM çeviri sorusu, anlam/birim/isim tutarsızlığı nedeniyle model çıktısı görülmeden elendi; aynı splitteki kaynak sorularıyla değiştirildi. `curate_math.py` nedenleri ve yeni soruların aritmetik incelemesini içerir. Cevap anahtarı değiştirilmedi.
- Gerçek scorer çalıştırma kontrolünde uppercase-word checker'ın İngilizce NLTK Punkt dosyasına bağımlılığı bulundu. Tarihî benchmark korunarak yeni development kontrolü Unicode kelime sınırlarıyla düzeltildi; GPU'da gizli runtime veri indirmesi gerekmiyor.
- 800 kelimelik isteğe 768 token cap uygulanmasının geçersizliği GPU öncesi düzeltildi. Dev output budget:263×768,5×2048,4×4096. Aynı bucketlar bütün6 modele uygulanır.
- Dev prompt üst sınırı ölçümü1919 token. CPU preflight tüm272 dev ve320 ayrılmış sorunun puanlayıcılarını gerçekten çalıştırdı; source-group kesişimi yok. Bu, anlam doğruluğunun tamamının bağımsız incelendiği iddiası değildir.
- Dondurulmuş manifest SHA256: `cf5b5a2bf7bd4d6ebe633e38eb4a8987c4c3f1b4806915ced86d640801298d74`.
- Local veri: `/home/serda/capability-recovery-artifacts/merge-eval-v5/`; rental state: `/home/serda/capability-merge-20261019/`.
- Ayrılmış havuzda320 örnek var; çoklu kısıt24/24 grubuyla tam final kapısını **geçmiyor**. Başarılı adaya yapılacak confirmation, tam production kabulünün yerine geçmez.

### Sampler gerçek veri sonucu

Dry run tamamlandı: **3,000,027** shifted assistant target token / **5,540** gerçek padded batch; whole-example seçiminden dolayı hedefin27 token üstünde. Plan ile gerçek label sayımı birebir eşit. Her aile için shortage0, max reuse2. Hazırlanan havuz52,093 benzersiz örnek; QA5,197. Tarihî17,000 generic reasoning örneği altı aileye zorla etiketlenmedi. Evaluation overlap nedeniyle352 kaynak QA adayı, incomplete evidence89, uzunluk12 ve exactduplicate5,063 elendi. İlk QA denemesinde string `answer_start` görüldü; int dönüşümü+literal offset doğrulamasıyla düzeltildi, başarısız v1 korunuyor.

| Aile | Gerçekte örneklenen target token | Oran |
|---|---:|---:|
|general|749,955|%24.9983|
|math|600,061|%20.0019|
|instruction|599,918|%19.9971|
|reading|449,989|%14.9995|
|code|300,172|%10.0056|
|grounded_qa|299,932|%9.9976|

Okuma havuzu2,500 örnek /232,763 target token ile en dar aile. Max reuse2 korunursa bu kompozisyon yaklaşık **3.10M toplam tokenı** destekler;20M dengeli exposure hazır olduğu söylenmez. Dil oranı ve doğal serbest sohbet çeşitliliği ayrı eksiklerdir. Eğitim başlatılmadı. Gerçek DataLoader ve optimizer window arayüzü `sampling.py`; dry-run makbuzları `results/merge_20261019/`.

### CI ilk koşu düzeltmesi

İlk yayın commitinde testlerin kullandığı `budget.py` ve `merge_experiment.py` sonraki committe kaldığı için GitHub CI başarısız oldu. Temiz checkout üzerinde aynı iki `ModuleNotFoundError` yeniden üretildi (38 pass/2 fail). Eksik modüller ikinci committe yayımlanıyor; en son temiz CI sonucunu ayrıca kaydedeceğiz. Bu başarısız koşu saklanıyor.

### CI temiz ortam bağımlılıkları

İkinci GitHub koşusunun gerçek job logları alındı: `safetensors==0.5.3` save yolu NumPy istiyor; workflow'da eksikti. Ayrıca eski pilot testleri import anında VPS'deki CPT transport kodunu arıyordu. CI'a NumPy1.26.4 eklendi; VPS transportu canlı rental/SSH çağrısında lazy yükleniyor. Bütçe/credential-redaction CPU testleri external VPS repo olmadan import edilebilir. Canlı merge payloadı/puanlayıcıları ve dondurulmuş manifest değişmedi.

### Sampler v4 — gerçek insan Türkçe diyalogları

İlk geçerli3M dry run üzerine upstream train bölümünden **1,000 insan tarafından yazılmış Türkçe görev odaklı çok turlu diyalog** eklendi. Upstream test+validation bütünü ve development/final exact/13-word spanları dışlandı.52 ek overlap adayı elendi; önceki QA dahil toplam404 overlap reddi. Kaynak revizyonu `72b45678a1fd22d5138491ad8e8c328d79de429a`; source/collection scope açıkça korunuyor. Bu dar corpus geniş serbest sohbet yeteneği diye adlandırılmıyor.

En son gerçek DataLoader dry run: **3,000,085** shifted target token / **5,525 batch**, max reuse2;53,093 benzersiz prepared örnek. Gerçek source-level sayımda yeni Türkçe diyaloglar **32,246 target token** aldı; yalnız havuzda var diye eğitim exposure'ı varsayılmadı. Altı aile tekrar hedef oranlarına ulaştı; bütün shortage değerleri0. [En son receipt](results/merge_20261019/sampler_dry_run_v4.json), [step exposure CSV](results/merge_20261019/accumulated_step_tokens_v4.csv). Önceki3,000,027-token run geçerli tarihî hazırlık kanıtı olarak korunuyor; en yeni havuz için3,000,085 sayısı kullanılır. Sampler'ın son arayüzü source metadata'sını model.forward'a geçirmeden filtreler;41 CPU test tekrar geçti. Merge dataset/payload/çıktı bütçeleri değişmedi.
