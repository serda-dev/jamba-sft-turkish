# SFT v3 — Türkçe yönerge takibi için ek veri ve kısa eğitim planı

Tarih: 17 Ekim 2026. Bu dosya hazırlık, eğitim ve sonuç kontrolünün ana günlüğüdür.
Durumlar ölçüm tamamlandıkça güncellenir; planlanan değerler gerçekleşmiş sonuç değildir.

## 1. Başlangıç ve amaç

Başlangıç: `serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015`, revizyon
`c8ee3888c486fa49d8d9e798a632c9f7ce664d7a`, alt klasör `checkpoints/final`.
6.782 adım / iki epoch SFT tamamlandı. Dosyaların HF kimlikleri yeniden doğrulandı.
Önceki A100 instance `56099238` silindi. Son doğrulanan kredi yaklaşık **20,11 USD**.

| Sabit test | V2 sonucu | Bu koşunun kabul sınırı |
|---|---:|---:|
| Türkçe yönerge strict | 41/120 (%34,2) | en az 60/120 (%50) |
| Yönergede token sınırına takılma | 34/120 (%28,3) | en fazla 12/120 (%10) |
| Matematik kısa | 73/200 | en az 68/200 |
| Matematik 8 örnekli | 53/200 | en az 48/200 |
| Okuduğunu anlama | 453/900 | en az 435/900 |
| MMLU | 226/900 | en az 208/900 |

Amaç, kullanıcının biçim/uzunluk koşullarını sağlayan tamamlanmış yanıtlar üretmek;
v2 matematik kazanımlarını korumak. %50 eşiği bir **hedef ve kabul sınırıdır**;
bu küçük eğitimle ulaşılacağı garanti edilmez. Bu test genel asistan kalitesinin
tamamını ölçmez. Yalnızca düşük loss, yayımlama kararı için yeterli değildir.

## 2. Hata incelemesi — tamamlandı

Kaynak: `/home/serda/sft-v2-run-20261015/final_benchmark.jsonl`.
Yeni çalışmanın `v2_instruction_errors.json` dosyasında her kuralın başarı/toplam/
kesilme sayısı bulunur. Öncelikler: paragraf 0/8, çoklu bölüm 0/2, kapanış 3/13,
yer tutucu 3/12, tırnak 1/10. Örneklerde tekrar döngüleri ve içerik hataları da var.
Tek nedenin EOS veya yalnızca biçim olduğu varsayılmıyor.

Bu analizde benchmark yanıtları eğitim hedefi olarak kopyalanmadı. Gözlenen görev
aileleri yeni kurallı veriyi şekillendirdiği için aynı testin sonraki skorunu
tamamen bağımsız, hiç görülmemiş bir değerlendirme gibi sunmayacağız.

## 3. Ek veri karışımı

| Bileşen | Eğitim hedefi | Geliştirme hedefi | İşlev |
|---|---:|---:|---|
| Yeni Türkçe, kayıt içeriğine bağlı görevler | 12.000 | 480 | Birden fazla koşulu birlikte uygulamak |
| Ai2 Tulu persona yönerge verisi, İngilizce | 1.000 | 100 | Daha doğal ve çeşitli isteklerde biçim aktarımı |
| V2'nin matematik eğitim verisinden tekrar | 2.000 | 0 | Matematik kazanımını korumak |
| V2'nin diğer eğitim verisinden tekrar | 2.000 | 0 | Genel yardım, okuma ve kod çeşitliliğini korumak |
| **Toplam hedef** | **17.000** | **580** | Gerçek sayılar manifestte |

Türkçe görevler 17 aile içerir: paragraf, kapanış, yer tutucu, tırnak, kelime
sıklığı, maddeler, bölümler, iki yanıt, büyük/küçük harf, JSON, CSV, kelime bütçesi,
yasak kelimeler, konuşmada hatırlama, içerikten hesaplanan JSON ve 100–120 kelimelik
iki paragraflı açıklama. Uzun yanıt isteyen örnekler de vardır; amaç her isteği
kısaltmak değildir. Kısa yanıt istendiğinde durmak ve gerektiğinde belirtilen
uzunluğa ulaşmak birlikte öğretilir.

Yanıtlar verilmiş kayıtlardaki kişi/kod/sayılarla ilişkilidir. Program sadece
biçimi değil gerekli içerik alanlarını da doğrular. Yeni sayı/kişi uydurmak,
yanlış JSON ve fazladan açıklama uygun ailelerde reddedilir. Bunlar **deterministik
sentetik kayıtlar**; doğal sohbet veya kapsamlı doğruluk öğretiminin yerine geçmez.
Eğitim ve geliştirme aynı görev kalıplarını paylaşır. Farklı değer/kimlik/ön söz
kullanılması kalıptan bağımsız genelleme kanıtı sayılmaz.

### Dış kaynak ve seçim gerekçesi

Kaynak: [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following).
Sabit revizyon: `fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e`.
Dosya: `data/train-00000-of-00001.parquet`. Yayıncı Ai2; lisans ODC-BY-1.0.
Kaynak kartı `tulu_source_card.md` olarak saklanır; atıf çıktı veri kartında da kalır.
Kaynak toplam 29.980 İngilizce sentetik istek içerir; burada tamamı kullanılmaz.

Ön incelemede bazı kaynak yanıtlarının kendi koşullarını sağlamadığı görüldü.
Bu yüzden **yalnızca tek beyan edilmiş koşulu olan ve yerel denetleyicinin açıkça
geçirdiği** kayıtlar seçilir. Desteklenmeyen/çıkarılamayan koşullar otomatik
reddedilir. JSON, virgülsüzlük, harf biçimi, tırnak, başlık ve çıkarılabilen
kapanış ifadeleri kontrol edilir. Bu denetim her örtük isteği ve olgusal
doğruluğu kanıtlamaz. Rastgele kaynak kullanımına kıyasla dar ve denetlenebilir
bir alt küme elde edilir. Kaynağın test soruları eğitimde kullanılmaz.

Lisansı belirtilmeyen `kilicai/turkish-sft-instruction-following-10k` bu koşuda
kullanılmıyor. Daha önce kullanılan SmolTalk tekrar yeni veri gibi sayılmıyor.
Tekrar örneklerinin özgün kaynak/revizyon/lisans bilgileri v2 manifestinden devralınır.

## 4. Hazırlık ve veri bütünlüğü

1. `data.py`: görevleri ve bağımsız biçim/içerik denetleyicilerini üret.
2. `prepare.py`: kaynak revizyonunu sabitle, eski eğitim kimliklerini çıkar,
   kaynakları filtrele ve yeni karışımı oluştur.
3. Donmuş 2.320 test koşulunun normalize prompt hashlerini ve kesintisiz 13
   kelimelik eşleşmelerini dışla. Yeni örnekleri eski v1/v2 eğitim/geliştirme
   promptlarıyla da karşılaştır. Tekrar verisi yalnızca eski **train** splitinden.
   Anlamsal veya çeviri düzeyinde sızıntının tamamı dışlanmış sayılmaz.
4. Yeni Türkçe hedeflerin tamamında programatik doğrulama yap. Dış kaynakta
   desteklenmeyen kuralı sessizce geçirme. Şekil/rol bozukluğunu ve tekrarı ele.
5. En fazla 2.048 input token ve yeni hedefte en fazla 384 token kullan.
   Uzun kayıtları **kesme; reddet**. Assistant-only loss ve assistant sonu EOS
   etiketi zorunlu. 100–120 kelimelik aile için gerçekte kabul edilen sayıyı raporla.
6. Kaynak/grup/prompt kimliğini sakla. Eğitim ve geliştirmede kimlik/grup kesişimi
   sıfır olmalı. Eski train tekrarlarının yeni geliştirmeye geçmesine izin verme.
7. Okunabilir mesajlar, token/label dosyaları, kaynak kartı, örnek inceleme listesi,
   token sayıları ve tüm dosyaların SHA256 değerlerini sakla.

Hazırlama komutları, repo kökünden:

```bash
/home/serda/.local/share/linguai-training/venv/bin/python -u sft_v3/prepare.py > /home/serda/sft-v3-run-20261017/prepare.log 2>&1
/home/serda/.local/share/linguai-training/venv/bin/python -m pytest -q sft_v3/test_data.py
```

Çalışma dizini: `/home/serda/sft-v3-run-20261017`.
`config.json` hazırlık sonrası **PREPARED_NOT_TRAINED** durumunu taşır.
Gerçek sayı/hash/denetim sonuçları `manifest.json` ve `data_audit.json` içindedir.

## 5. Eğitim planı — henüz başlatılmadı

- Başlangıç doğrudan doğrulanmış **v2 final**. İkinci CPT veya iki başlangıçlı
  pilot gerekmiyor. Yeni optimizer ile ayrı bir SFT koşusu.
- Tek A100 SXM4 80 GB, en fazla $1,35/saat. Hazır Docker digest:
  `serdadev/jamba2-cpt@sha256:3222508c69957f28d1ce79200686c0bf665416cb74d5ddb162bf7f99377f5b4b`.
- BF16 **tüm ağırlıklar**, AdamW 8-bit optimizer durumları, microbatch 2 ×
  accumulation 16 = etkin batch 32. Bir epoch; 17 bin örnekte yaklaşık 532 adım.
  LR `2e-6`, %5 warmup, cosine. Düşük LR ve tekrar verisi mevcut becerileri koruma
  tedbiridir; korunduğu ayrıca test edilir.
- 2.048 eğitim uzunluğu bu kısa hedefli karışımın filtresidir; modelin yapılandırılmış
  bağlam kapasitesi değiştirilmez. Tamamlanmamış yanıtla eğitim yapılmaz.
- Dinamik padding, BF16, gradient checkpointing kapalı. Başta en uzun gerçek
  örneklerle iki optimizer adımlık bellek kontrolü; probe ağırlıkları atılır.
- Her 200 adımda geliştirme loss ve checkpoint; en düşük geliştirme loss ile
  seçim. Sonraki test skoruyla checkpoint seçme veya test başına tekrar eğitim yok.
- `absl-py==2.3.1`, accelerate ve mevcut scorer bağımlılıkları baştan kurulur.
  Gerçek import ve bir üretim örneği, uzun koşudan önce doğrulanır.
- GPU'nun gerçek ilk 50 adım süresiyle kalan süre yeniden hesaplanır ve bu
  dosyaya yazılır. V2'nin 108 bin örnek × iki epoch / 7,28 saat sonucu yalnızca
  referanstır; sıra uzunluğu/indirme/test farkları nedeniyle doğrudan hız garantisi değildir.

**Bütçe:** eğitim + kurulum + test için başlangıç tahmini **1–3 saat / yaklaşık
$2–5**; ölçülmüş v3 süresi değildir. Sert iş bütçesi $5 ve 3 saattir; kalan
kredi $15 civarına geldiğinde güvenli export, $12'de compute durdurma. Depolama
ve veri aktarımı ücretleri de bütçe kontrolünde hesaba katılır. Ek kredi yükleme
ve başarısız eşik sonrası otomatik uzun eğitim yok.

## 6. Son değerlendirme ve modelin hazır sayılması

1. Donmuş aynı 2.320 koşulu **aynı üretim/puanlama ayarlarıyla** çalıştır.
   Yönerge skorunu artırmak için üretim limitini veya kuralları değiştirme.
2. Eğitimde kullanılmamış geliştirme görevlerinde yapısal hataları, tekrarları
   ve bitişleri raporla; bunları genel kalite puanı olarak sunma.
3. Aynı yaklaşık 2M token TR/EN retention setini v2 ile karşılaştır. Ortalama
   NLL artışı TR en fazla %2, EN en fazla %3. Donmuş manifest hashleri eşit olmalı.
4. Girişteki tüm kabul sınırlarını kontrol et. Tek yönerge artışı, matematikte
   ciddi düşüşü örtmez. 100–120 kelimelik yanıt ve JSON/multi-turn örneklerini
   ayrıca gözle incele. Örnekler ve inceleme sonucu dokümana eklenir.
5. Özel aday `serda-dev/Jamba2-3B-Turkish-SFT-v3-20261017` altında model,
   tokenizer, kaynak kartı, provenance, ham test yanıtları ve son raporu sakla.
   Model dosyalarını ve kayıt arşivini HF revizyonu ve SHA ile doğrula.
6. Doğrulamadan sonra instance'ı sil ve CLI listesinde bulunmadığını kaydet.
   Hata durumunda GPU'yu durdurup diski kurtarma için tut; durdurulan diskin de
   ücretlendiğini raporla. Mevcut v2 erişilebilir kalır.
7. Bütün eşikler geçerse adayı “hazır” olarak işaretle; mevcut herkese açık modelin
   değiştirilmesini ve v3 yayınını ayrı, açıkça kayıtlı yayın adımı olarak yap.
   Eşikler geçmezse gerçeği raporla, v2'yi yeni modelle değiştirme.

## 7. Adım günlüğü

| Adım | Durum | Kanıt |
|---|---|---|
| V2 çıktısı / GPU / kredi kontrolü | Tamamlandı | V2 `final_audit.json`, CLI: açık instance yok |
| Kural bazında hata çıkarımı | Tamamlandı | `v2_instruction_errors.json` |
| Kaynak metadata ve lisans kontrolü | Tamamlandı | `tulu_splits.json`, `tulu_first-rows.json`, kaynak kartı |
| Yeni hedef üreticisi ve negatif testler | Tamamlandı | `sft_v3/data.py`, `test_data.py` |
| Kaynak hedef kalite kontrolü | Tamamlandı | Bazı örnekler ihlal içerdi; dar tek-kural filtresi eklendi |
| CPU veri hazırlığı | Tamamlandı | 17.000 train + 580 development; `prepare.log`, `manifest.json` |
| Veri/maske/hash/grup denetimi | Tamamlandı | Bütün 17.580 kayıt yeniden kodlandı; `data_audit.json` |
| Agent ile örnek incelemesi | Tamamlandı | 26 örnek; `reviewed_examples.json`, `review_notes.json` |
| GPU kiralama / bellek kontrolü | Başlatılmadı | Kiralama ID ve gerçek ölçüm sonra eklenecek |
| Eğitim / checkpoint / gerçek maliyet | Başlatılmadı | Adım, token, loss, süre, ücret sonra eklenecek |
| Son testler / kabul / HF doğrulama | Başlatılmadı | Ham yanıt, skor, dosya hashleri sonra eklenecek |
| GPU silme / adayın hazır olması | Başlatılmadı | CLI sonucu ve karar sonra eklenecek |

Bu günlüğün sonraki her aşamasında kullanılan komut, başlangıç/bitiş zamanı,
elde edilen ölçüm, hata varsa düzeltmesi ve çıktı dosyası buraya eklenir.

+
+## 8. Gerçekleşen hazırlık sonuçları — 17 Ekim 2026 UTC
+
+- Son veri: **17.000 eğitim + 580 geliştirme** kaydı. Eğitimde 12.000 yeni
+  Türkçe, 1.000 dış kaynak İngilizce, 2.000 matematik ve 2.000 diğer tekrar var.
+- Eğitim: **5.185.245 input token**, **1.804.048 assistant hedef token**.
+  Geliştirme: 141.152 input / 39.258 assistant token. Bunlar bir epoch değerleri.
+- Bütün token/label kayıtları okunabilir mesajlarından bağımsız geçişte yeniden
+  kodlandı ve birebir eşleşti. Prompt ve grup split kesişimi: **0**.
+  Eski eğitim/geliştirmede görülmüş yeni geliştirme promptu: **0**.
+- 17 Türkçe ailenin tamamı her iki splitte var. 100–120 kelimelik aile de
+  kabul edilmiş hedefler içeriyor. Yeni hedeflerin en uzunu 384 token.
+- 27.757 dış kaynak adayı desteklenmeyen/çoklu veya başarısız kural nedeniyle
+  elendi; ayrıca 468 uzun hedef ve 33 uzun input/EOS uygunsuzluğu reddedildi.
+  Bu sayılar kaliteyi kusursuz yapmaz; uygulanan elemenin kaydıdır.
+- 26 örnek agent tarafından incelendi: 17 görev ailesi + 5 dış kaynak + 4 tekrar.
+  Örneklerin kimlikleri ve bulgular `review_notes.json` dosyasında.
+- İnceleme düzeltmeleri: eski SmolTalk constraint tekrarlarının tamamı çıkarıldı;
+  kelime sıklığı yanıtından kayıtta verilmeyen aşama tamamlama iddiası kaldırıldı;
+  98 kelimelik ilk uzun yanıt taslağı kurala uymadığı için düzeltilip veri yeniden
+  üretildi. Son veri bu düzeltmelerden sonra tekrar denetlendi.
+- `sft_v3/test_data.py` beş negatif/pozitif testi geçti. Yedi mevcut recovery
+  testiyle birlikte toplam **12 test geçti**.
+- Son manifest SHA256:
+  `f6aa634ec3e336eb5baa68b0ae4eab86e845d8e17658c1c7d36c46dc3d3bafb6`.
+- 18:14:30 UTC canlı Vast kontrolü: team 733063 kredisi **$20,11094**.
+  A100 SXM4 80GB teklif 46608736: 100GB disk dahil yaklaşık **$1,09444/saat**;
+  ayrıca ilan edilen upload/download maliyetleri sırasıyla $0,03906 / $0,02604
+  / GB. Teklif kiralanmadı ve uygunluğu değişebilir. `offer_check.json` saklandı.
+- Yeni model eğitimi **başlatılmadı**. Hazır olan çıktı ek veri, denetim ve eğitim planıdır.
+
+Özel veri yedeği hedefi:
+`serda-dev/LinguAI-Turkish-Instruction-v3-20261017`.
+`publish.py`, gerçek `datasets` JSON okuyucusuyla mesaj splitlerini açar;
+başarılı denetim ve sabit manifestten sonra yükler. Bütün uzak dosyaların
+hash/boyut kontrolünden sonra `dataset_hub_receipt.json` oluşturur.
+
## 9. Yedekleme ve okuyucu kontrolü — tamamlandı

- Özel HF veri reposu: [LinguAI-Turkish-Instruction-v3-20261017](https://huggingface.co/datasets/serda-dev/LinguAI-Turkish-Instruction-v3-20261017).
- Doğrulanmış sabit revizyon: `25fdf7ee0d5abdcc183c30def40e340eca4e27a2`; 24 dosyanın boyut ve hashleri karşılaştırıldı.
- `messages` ve `tokenized` configleri var. `datasets` JSON okuyucusu messages train/dev splitlerini 17.000/580 kayıtla açtı.
- İlk okuyucu denemesi, ilk bölümde boş olan kaynak kural listelerinin Arrow tarafından `null` sayılması nedeniyle başarısız oldu. Dağıtım şemasında `checks_json` ve `source_constraints_json` metin alanlarına geçildi. Okuyucu kontrolü tekrar başarıyla geçti; hedefler değiştirilmedi.
- Kanıt: `dataset_hub_receipt.json`, `publish.log`. Tamamlanan aşamalar veri hazırlığı, denetim, örnek incelemesi ve özel veri yedeğidir. GPU kiralama, v3 eğitimi ve v3 model yayını henüz yapılmadı.


## 10. Kullanıcı devam talimatı ve yürütme — 17 Ekim 2026

Kullanıcı “devam et” diyerek bu hazırlıktan eğitime geçilmesini istedi.

- Yeni yürütücü: `sft_v3/runtime.py`; ayrı `preflight`, `train`, `finalize`
  modları. Başlangıç modelinin bütün dosyaları v2 receipt ile SHA256 doğrulanır.
- İki gerçek, maksimum uzunluklu AdamW8bit adımı; scratch ağırlıklar atılır.
  Eğitim, kimliği doğrulanmış orijinal v2 modelini yeniden yükler.
- 200/400. adımlara ek olarak son 532. adımda da geliştirme değerlendirmesi ve
  checkpoint istenir. Seçim yalnızca geliştirme loss'u ile yapılır.
- Sonra 34 geliştirme üretimi (17 aile × 2), aynı 2.320 dış test ve aynı TR/EN
  retention seti çalışır. Geliştirme üretimleri kalıp paylaştığı için bağımsız
  genel başarı ölçümü sayılmaz. Önceden yazılan kabul sınırları aynen korunur.
- `sft_v3/supervisor.py` ana takipçisi 60 saniyede bir ilerleme ve krediyi
  denetler; her beş dakikada kayıtları HF'ye doğrulayarak yedekler.
- `$5` toplam kredi tüketimi ve 3 saat sınırı, kiralama anından itibaren hesaplanır.
  `$4` tüketim veya 2 saat 40 dakikada güvenli export istenir; $15 kalan kredi
  eşiği de korunur. Ücretlendirme/API gecikmeleri nedeniyle limitler fatura
  kuruşu garantisi değil, otomatik durdurma eşikleridir.
- `sft_v3/watchdog.py` ayrı systemd timer ile her iki dakikada ana servisi ve
  sert bütçeyi denetler. Ana servis ölürse yalnızca bu koşunun etiketli GPU'sunu
  durdurur. Hata durumunda disk korunur ve depolama ücretlenmeye devam edebilir.
- Terminal `DONE` veya güvenli export sonrasında model ve kayıt receipt'leri
  doğrulanır, instance silinir, CLI listesinden kaybolduğu kontrol edilir.
- Çalışan servislere ait unit dosyaları `sft_v3/systemd/` içinde saklandı.
- Saf kabul/bütçe kuralları ve veri kontrolleri: **15 test geçti**.

Kiralama: teklif **34475480**, instance **56434707**, tek **A100 SXM4 80 GB**,
100 GB disk dahil yaklaşık **$1,22/saat**. Aktarım maliyetleri de teklif seçiminde
hesaba katıldı: 3 saat compute + 65 GB upload + 15 GB download üst tahmini $5'in
altında olmalı. İlk kredi **$20,11094**. Sabit Docker digest bölüm 5'tekiyle aynı.
Hazırlık ayrıca özel model reposuna `prepared/` olarak, revizyon
`65841d4807dbb08c1836f834afd971501abafe74` ile yedeklendi.

### Canlı adım günlüğü

Her ilerleme, bellek sonucu, hata, export ve silme olayı zaman damgasıyla
`/home/serda/sft-v3-run-20261017/OPERATIONS_TR.md` dosyasına eklenir. Ham GPU olayları
`events.jsonl`, eğitim ölçümleri `metrics.jsonl`, durum `supervisor.json` içindedir.
Model reposunda `run/logs.tar.gz` ve sonrasında `run/OPERATIONS_TR.md` saklanır.
Bu bölümün yazıldığı anda GPU **BOOTING** durumundaydı; eğitim başarısı veya
kabul kararı henüz mevcut değildi.

Başlatma komutu:

```bash
systemctl --user enable --now linguai-sft-v3-watchdog.timer linguai-sft-v3-supervisor.service
```

İzleme:

```bash
systemctl --user status linguai-sft-v3-supervisor.service linguai-sft-v3-watchdog.timer
cat /home/serda/sft-v3-run-20261017/supervisor.json
```
