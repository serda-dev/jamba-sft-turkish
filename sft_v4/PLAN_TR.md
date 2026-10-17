# SFT v4 — kaynak metne dayanan doğal Türkçe yönergeler

Durum: veri doğrulandı, GPU eğitimi hazırlanıyor. 17 Ekim 2026 (UTC); çalışma kimliği `20261018`, sürecin gece yarısına uzaması için sabitlenmiştir. Adım günlüğü VPS'de `/home/serda/sft-v4-run-20261018/OPERATIONS_TR.md`.

## 1. Amaç ve bütçe

Kullanıcının “güzel, devam et” onayı üzerine, önceki $0.46 tanısında doğrulanan **BF16 working model + FP32 optimizer ana ağırlıklarını** gerçek bir yönerge paketinde uygula. Başlangıç, korunmuş V2: `serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015`, revision `c8ee3888c486fa49d8d9e798a632c9f7ce664d7a`, `checkpoints/final`. V3 üzerinden devam edilmez.

Tek A100 SXM 80 GB. **Toplam harcama tavanı $5, kiralamadan itibaren süre tavanı 3 saat**, saatlik teklif <=$1.35. Hesap bakiyesi farkı izlenir; $4 veya son 20 dakikada güvenli export istenir, $5/3 saatte durdurulur. Bağımsız watchdog 2 dakikada bir ana izleyici ve bütçeyi kontrol eder. Model ve loglar özel HF reposunda boyut/hash doğrulanmadan instance silinmez. Hata durumunda GPU durdurulur ve korunur.

## 2. Veri: ne üretildi, ne üretilmedi?

Yeni bir ücretli Teacher yanıt üretimi yapılmadı. Mevcut doğal Türkçe kaynak yanıtları, **kaynak metin olarak isteme konularak** düzenleme, ayıklama ve biçimleme görevleri oluşturuldu. Böylece kaynak sözcükleri ve istenen format bağımsız olarak denetlenebiliyor. Bu paket serbest bilgi üretimi veya açık uçlu özet kalitesi öğretmeni değildir. Kaynak metnin gerçek dünyadaki tüm iddialarının doğruluğu sertifikalandırılmadı; görev kaynağa sadakattir.

Örnek kaynak incelemesinde bazı Atlas yanıtları kendi eski kısıtlarıyla örtüşmüyordu. Eski istem/yanıt çiftlerini doğrudan doğru etiket olarak almak yerine, yanıt metni açık bağlam yapıldı ve yeni, uyumlu talimatlar oluşturuldu. Eski çok koşullu soru yeni soruya eklenmedi. Kaynaklar:

- [Turkce Atlas Instruct](https://huggingface.co/datasets/AlicanKiraz0/Turkce-Atlas-Instruct), MIT, sabit revision `c387738c5deacbb35667ab4057930adc34e647e8`.
- [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset), Apache-2.0, sabit revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`; yalnızca Türkçe train alt kümesi.
- Önceki V2 **train** verisinden sınırlı doğal yanıt tekrarı; önceki lisans/provenans devralınır. Test splitleri eğitim kaynağı yapılmadı.

| Bileşen | Örnek | Assistant hedef tokenı | Pay |
|---|---:|---:|---:|
| Kaynak temelli yönerge | 2.000 | 299.059 | %69,98 |
| Doğrulanmış aritmetik | 717 | 64.137 | %15,01 |
| Genel doğal tekrar | 365 | 64.152 | %15,01 |
| Toplam | 3.082 | 427.348 | %100 |

Bu hedef-token kütlesidir; mikro batch başına ortalama loss nedeniyle tam gradyan katkısı oranı değildir. Kaynak temelli yönergelerin 1.760'ı Atlas, 240'ı Aya metinlerinden gelir. Konular ve içerikler farklıdır; her kaynak için tek bir yönerge örneği kullanıldı.

16 aile, her birinde 125 örnek: yıldız/boş satır paragrafları, madde/numaralı liste, JSON, Türkçe büyük/küçük harf, kapanış, bölüm başlıkları, satır başına anahtar kelime, iki ayrı metin, kelime sınırı, tırnak, virgül dönüşümü, konuşma takibi ve kaynak açıklamasını sunma. Parametre ve talimat ifadeleri çeşitlidir. Yaklaşık yarısı system mesajı içerir (992/2.000).

## 3. Split ve veri kontrolleri

- **128 yeni geliştirme örneği**, 16 aileden 8'er tane; eğitimde kullanılmayan kaynak soru ve yanıt grupları. Kaynak soru kimlikleri önceki V1/V2 train/dev tam istem kimliklerinden de elendi.
- Yeni **64 aritmetik + 64 okuma** gelişim/korunum kontrolü. Bunlar sabit dış benchmarkın yerine geçen genel başarı ölçüleri değildir.
- Önceki 32/128 tanı örnekleri yeni veriye eklenmedi.
- Sabit 2.320 benchmark koşuluyla normalize istem hashleri ve ardışık 13 kelime eşleşmesi engellendi. Bu, semantik sızıntının yokluğunu ispatlamaz.
- Kaynak soru hashleri ve benzersiz kaynak yanıtları splitler arasında ayrılır. Genel tekrarla dev kaynak yanıtı tam eşleşmesi ayrıca kontrol edildi: **0**.
- Tüm 2.000 yönerge ve 128 dev hedefinde biçim + kaynak içerik sadakati doğrulandı. Matematik cevapları tamsayı işlemleriyle hesaplandı.
- Bütün assistant etiketleri decode edilerek metin + `<|im_end|>\n` ile bire bir karşılaştırıldı; EOS sayıları kontrol edildi. **Eğitim hedefi kırpılmadı.**
- Gerçek en uzun eğitim örneği **1.063 token**, izin verilen üst sınır 1.536. Geliştirme yanıt üretimi üst sınırı 512 yeni token; hedefler <=384 token.
- JSON duplicate key, yanlış tip/içerik; paragraf ayırıcı; satır sayısı/kelime; math doğruluğu negatif testleri eklendi. Önceki FP32 birikim ve büyük-matris indeks regresyonları da çalıştırılır.

## 4. Eğitim ve pilot kapısı

Full weight, BF16 working model, ayrı FP32 optimizer ana ağırlıkları, AdamW8bit. Microbatch **1**, accumulation **8**, effective batch **8**. İki epoch, yaklaşık **771 optimizer adımı**. Maksimum LR **1e-5**, 10 adımlık ısınma ve cosine decay. Tüm 3B parametreler eğitilir; katman dondurma veya LoRA yok.

Önce V2 ile 128 doğal dev + yeni 64 math/64 reading baseline ölçülür. Gerçek en uzun örnekte LR=0 ile optimizer durumları ve aktivasyonlar tahsis edilerek VRAM/finite loss kontrol edilir; bu kontrolün optimizer durumları sıfırlanır. Sonra asıl eğitim başlar.

**100 adım sonunda otomatik devam kapısı:**

1. Dev strict doğruda en az **13/128 artış** (yaklaşık +10 puan); baseline zaten yüksekse >=122/128 ve kötüleşmeme.
2. Yeni math ve reading kontrollerinde ayrı ayrı en fazla **4/64** gerileme.
3. Dev uzunluk kesilmesi baseline'dan en fazla 2 fazla.
4. Aynı CPT heldout manifestinde, token sayıları eşit olmak üzere Türkçe NLL V2'den en fazla **%2**, İngilizce NLL en fazla **%3** yüksek.
5. Tüm sonuçlar tamamlanmış ve finite olmalı.

Kapı geçmezse daha fazla epoch yapılmaz. 100-adım modeli yine özel HF'ye doğrulanarak kaydedilir. Kapı geçerse iki epocha devam edilir. Dev/korunum kontrolleri her 200 adımda ve sonunda tekrar edilir. Seçim yeni dev strict başarı ve kesilme ile yapılır; math/reading gerileme sınırını geçmeyen adaylar seçilebilir. Eşitlikte erken checkpoint korunur. Son seçilen aday dil NLL korunumunu geçmezse, **sabit benchmarkı çalıştırmadan önce**, korunumdan geçmiş 100-adım adayına dönülür.

100-adım ve periyodik model yedekleri GPUdan doğrudan HF'ye gider. FP32 master/optimizer durumlarının bit düzeyinde resumable checkpointi bu kısa işte yayımlanmaz; dışa aktarılan model BF16 working ağırlıklardır. Tamamlanmış eğitimden seçilen adım, eğitimde çalışılmış toplam adımla ayrı raporlanır.

## 5. Son bağımsız değerlendirme

Yalnızca yeni geliştirme verisiyle seçilmiş son adayda, önceki protokol değişmeden tek tam benchmark: MMLU 900, Belebele 900, GSM8K 200, GSM8K 8-shot 200, IFEval 120. Bu test ağırlık veya LR seçmek için kullanılmaz.

Kabul tabanları: IFEval **>=60/120**, kesilen IFEval **<=12**; GSM8K >=68/200; 8-shot >=48/200; Belebele >=435/900; MMLU >=208/900; aynı TR/EN NLL korunum sınırları. Bütün kapılar ve eğitim tamamlanması gerekir. Başarısız aday da kaydedilir ve açıkça reddedilir. Üretim V2 otomatik değiştirilmez.

## 6. Çalıştırma ve kanıtlar

CPU hazırlık: `python sft_v4/prepare.py`. Denetleyici: `python sft_v4/supervisor.py`. Servis: `linguai-sft-v4-supervisor.service`; watchdog timer: `linguai-sft-v4-watchdog.timer`.

VPS çalışma dizini `/home/serda/sft-v4-run-20261018`; GPU `/mnt/home_extra/sft`. Özel model hedefi `serda-dev/Jamba2-3B-Turkish-SFT-v4-20261018`, final yolu `checkpoints/final`. Manifest, veri auditi, ham mesajlar, kod, eğitim metrikleri, pilot kapısı, acceptance ve final/log receiptleri tutulur. VPSye büyük model ağırlıkları indirilmez.

Kaynak okuma: Dataset Viewer split/preview; Aya filtre endpointi HTTP 500 verdi, sabit parquet yerel okunup `language_code=tur` filtresi uygulandı. Atlas sabit JSONL kaynağı indirildi. Tüm adımlar ve son durum çalışma günlüğüne eklenir.


## Başlangıç kaydı

15 CPU testi geçti. Dataset özel Hub reposuna kaydedildi: `serda-dev/turkish-grounded-instructions-v4-20261018`, revision `c80ca20d712dc24b5a9f2c7a456efa5a103a9f28`; 6 dosyanın boyut/hash denetimi geçti. Parquet roundtrip bütün satırlarda doğrulandı. İlk dataset upload README lisans bağlantısı için HTTPS URI istedi; bağlantı düzeltildi ve upload tamamlandı.

A100 80 GB instance **56468842**, teklif **36994163**, saatlik **$1.07963**, 70 GB disk, kiralama **17 Ekim 23:55:18 UTC**. İlk 100 GB disk araması ucuz teklifi dışladığı için GPU kiralamadan durdu; ölçülen disk gereksinimi 70 GB içine sığdığından düzeltildi. Aynı $5 / 3 saat sınırı korunuyor. Kod, veri, etiket auditi ve protokol kiralamadan önce özel model reposuna yedeklendi.


### Eğitim öncesi tokenizer tutarlılığı

CPUda doğrulandı: fast tokenizer ile padded generation, backend padding durumunu ve mevcut fingerprint fonksiyonunun hashini değiştiriyor. V4 dil heldout kontrolü, generationdan geçmiş tokenizer yerine sabit dosyalardan taze tokenizer kullanır. Checkpoint exportlarında da sabit tokenizer dosyaları bire bir kopyalanır. Eğitim/baseline ölçümü başlamadan iki başlangıç güncellemesi yapıldı; indirme cacheden sürdü, başlangıç logları korundu. Cache tokenları ve benchmark protokolü değiştirilmedi.
