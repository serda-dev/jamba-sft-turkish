# Kapsamlı SFT v5 — 18 Ekim 2026

Kullanıcı veri hazırlığı, GPU kiralama ve eğitimi başlatmayı onayladı. Bu belge koşu öncesi protokoldür; son durum otomatik çalışma günlüğüne işlenir: `/home/serda/sft-v5-run-20261018/OPERATIONS_TR.md`.

## Veri ve kapsam

Başlangıç korunmuş V2: `serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015`, revision `c8ee3888c486fa49d8d9e798a632c9f7ce664d7a`, `checkpoints/final`. Reddedilen V3/V4 üzerinden devam edilmiyor.

**63.896 konuşma**, 39.468.325 giriş ve 19.884.747 assistant hedef tokenı; 3.125 çok turlu konuşma. 61.896 örnek V2 train tekrarından, 2.000 kaynak metne sadık görev V4 hazırlığından gelir. Bunlar 64 bin yeni Teacher yanıtı değildir. V2'ye maruz kalmış veriler açıkça tekrar sayılıyor. Bir epoch, 1.997 optimizer adımı; ikinci epoch otomatik yapılmaz.

| Veri | Örnek |
|---|---:|
| Atlas / merve / Aya genel Türkçe | 21.500 |
| UltraChat, İngilizce çok turlu yardım | 3.000 |
| GSM8K Türkçe / İngilizce train | 7.396 / 3.000 |
| Türkçe web reasoning / science | 14.000 / 3.000 |
| Türkçe kod görevleri | 5.000 |
| Smol constraints / summarize | 2.500 / 2.500 |
| Kaynak temelli Türkçe yönergeler | 2.000 |

Kaynak revizyon ve lisansları `manifest.json` içinde, eski V2/V4 manifestleriyle ilişkilidir. Yayıncının `verified` etiketi her referansın bağımsız olarak doğru olduğu anlamına gelmez; doğal kaynakların olgusal doğruluğu tüm satırlarda sertifikalandırılmadı. Eski basit aritmetik/okuma şablonları yeni train karışımına eklenmedi.

Ham assistant tokenlarının %54,57'si reasoning, %1,17'si okumaydı. Bu nedenle yalnız örnek sayısı dengelemesi kullanılmıyor. `balance_objective.py` şu hedeflere göre bileşen ağırlığı yazar: **%25 genel TR, %25 matematik/reasoning, %20 yönerge, %15 okuma, %10 kod, %5 genel EN**. Ağırlık = hedef pay / ham hedef-token payı. Her optimizer batchinde ağırlıklı assistant tokenı üzerinden normalizasyon yapılır. Veri metni/etiketi değişmez; veri kütlesi payı tam gradyan katkısı değildir. Okuma/constraint ve matematik bileşenlerinde de İngilizce vardır; %5, toplam İngilizce oranı değildir. Önceki önerinin %10–15 toplam EN hedefi garanti edilmez; mevcut denetlenmiş kaynaklarla kapsam korunur.

63.896 train ve 870 validation kaydının **tüm assistant etiketleri decode edilip kaynak mesaj + EOS ile bire bir karşılaştırıldı**, etiket konumları ve EOS sayısı kontrol edildi. En uzun örnek 3.071 token, limit 3.072. Hedefler kırpılmadı. Hazırlık ilk denemede bir doğal özetin sonda iki boş satırını çözümleyici silince durdu; çözümleyici boş satırları koruyacak şekilde düzeltildi ve tüm audit yeniden geçti. Kaynak hedef değiştirilmedi.

## Geliştirme ve test ayrımı

- 128 kaynak koruma/biçim görevi: V4'ün V2 traininden ayrılmış dev grupları, bu koşunun trainine alınmaz. Bu görülmüş bir geliştirme setidir, yeni bağımsız final test diye sunulmaz.
- **143 gerçek çok adımlı matematik problemi:** V2 validationdan 72 TR + 71 EN. V2 traininden tam istem kimliğiyle ayrıdır; çeviri/semantik eşleşmenin tamamen yokluğu garanti edilmez. Küçük aritmetik kontrolünün yerine bu kullanılır.
- 870 eski validation kaydı train dışı tutulur; her aileden sabit en fazla 64 örnekle assistant token ağırlıklı loss takibi yapılır. Kod/okuma/reference loss anlam doğruluğu veya yürütülmüş kod testi yerine geçmez.
- 128 ek Türkçe GSM8K sorusu: eski 2.320 testle 13 kelime çakışması olmayan, eski eğitimlerden tüm test splitini dışlama kuralıyla korunmuş örnekler. **Yalnız model seçimi bittikten sonra** V2 ve son adayda ölçülür; eğitim/dev seçimine kullanılmaz. Bu yeni yönerge testinin yerine geçmez.
- Eski 2.320 koşul artık görülmüş bir regresyon raporudur, tamamen görülmemiş nihai başarı tahmini diye sunulmaz. Test üzerinden checkpoint/LR seçilmez.
- Tüm GSM8K-TR test splitine ve eski sabit teste normalize tam hash / ardışık 13 kelime kontrolü; train/validation tam hash ayrımı. Kaynak gerçekliği, semantik ve çeviri sızıntısı bu kontrollerden daha geniş problemlerdir.

## Eğitim

Tek A100 SXM 80 GB. **QLoRA yok**: tüm BF16 çalışma ağırlıkları + ayrı FP32 optimizer ana ağırlıkları, AdamW8bit optimizer durumları. LR **7,5e-7**, 50 adım warmup, cosine decay, microbatch **1**, accumulation **32**, effective batch **32** (son batch daha küçük), bir epoch. Assistant loss görev bileşenlerine ve hedef tokenlarına göre ağırlıklandırılır; user/system/padding maskelenir. FP32 mastersız eski V2 öğrenme oranı aynen taşınmaz.

İlk olarak V2 dev, doğal math, aile lossları ve sabit 2M-token TR/EN NLL ölçülür. Ardından gerçek en uzun örnekte LR=0 optimizer adımıyla bellek kontrolü; optimizer durumları sıfırlanır. Çalışma ağırlıkları değişmez. Sığmıyorsa eğitim başlamaz; otomatik kalite/bağlam kısması yapılmaz.

**100-adım pilot kapısı, sonuçlar görülmeden sabitlenmiştir:**

1. 128 yönerge devinde V2'ye göre en az **4 ek strict doğru**.
2. 143 doğal math probleminde en fazla **4 doğru kayıp**.
3. Dev kesilmesi baseline + **2** sınırında.
4. Her validation ailesinin assistant lossu baseline × **1,05** sınırında; eşit tokenlar ve tam sonuç gerekir.
5. Sabit TR NLL V2 × **1,02**, EN × **1,03** sınırında; manifest/token kimliği eşit.

Başarısız pilot tam eğitime geçmez; model/loglar kaydedilip instance silinir. Geçerse aynı optimizer ve scheduler ile 1.997 adıma devam edilir. Her 400 adımda ve sonunda aynı dev/math/aile lossları ölçülür. Korunum ve durdurma sınırını geçen adaylar arasında strict yönerge başarısı, sonra düşük kesilme, sonra aile lossları ile seçim yapılır. Son dil korunumunu geçmeyen geliştirme adayının raporu korunur; dış test açılmadan, korunumdan geçmiş pilot100'e dönülür. Seçilen adım ile toplam çalışılmış adım ayrı kaydedilir.

Tam epoch bitmediyse model kabul edilmez. Eski kabul tabanları: IFEval ≥60/120 ve kesilme ≤12; GSM ≥68/200, 8-shot ≥48/200; Belebele ≥435/900, MMLU ≥208/900; aynı NLL sınırları. Ek math final testinde V2'ye göre en fazla 4/128 kayıp. Bunlar pratik eşiklerdir; istatistiksel üstünlük iddiası değildir. Üretim V2 otomatik değiştirilmez.

## Bütçe, kayıt ve otomasyon

Başlangıçta Team bakiyesi yaklaşık **$16,64**. Bu koşunun ayrı üst sınırı **$8 / kiralamadan itibaren 6 saat**, teklif ≤$1,35/saat. $7 harcamada veya son 20 dakikada güvenli export istenir; $8/6 saat ya da bakiye $5'te GPU durdurulur. Normal yumuşak kredi tabanı $7. Aday seçilirken transfer ücretleri de hesaba katılır. Gerçek pilot hızı loglara yazılır; süre tahmini ölçüm gelmeden garanti edilmez.

Denetleyici `sft_v5/supervisor.py`, servis `linguai-sft-v5-supervisor.service`; bağımsız watchdog 2 dakikada bir. Hazırlık modelin özel Hub reposuna kiralamadan önce yedeklenir. Koşu kodu/hashleri, veri manifesti ve tüm metrikler saklanır. Periyodik model/log yedekleri GPUdan HF'ye gider. VPS'ye büyük ağırlık indirilmez. FP32 optimizer master durumları tam yeniden başlatılabilir checkpoint olarak dışa aktarılmıyor; model BF16 exporttur.

Normal DONE durumunda **model ve log uzak boyut/hash doğrulanınca** yalnız bu koşunun etiketli instance'ı silinir. Hata/bütçe sert duruşunda disk kaybını önlemek için GPU stop edilip disk tutulur; bu durumda depolama ücretinin tamamen bittiği iddia edilmez. Kimlik/SSH/token dosyaları log arşivine alınmaz.

Özel hedef: `serda-dev/Jamba2-3B-Turkish-SFT-v5-20261018`, final `checkpoints/final`. Son durum, instance kimliği, saatlik fiyat, pilot ve kapanış olayları operasyon günlüğüne yazılır. CPUda gerçek math/korunum kapıları, token ağırlıklı accumulation ve LR ısınması için dört odaklı test geçti; GPU çalışma/bellek kontrolü henüz koşu başlangıcında yapılacaktır.


## Başlatma doğrulaması

18 Ekim 2026, yaklaşık 14:30 UTC: instance 56565846 üzerinde ilk 10/1.997 optimizer adımı doğrulandı. [Başlatma kaydı ve ilk hız ölçümü](LAUNCH_TR.md). 100-adım pilot kapısı henüz bekleniyor.


## Pilot sonrası sonuç

100. adımda kapı geçilmedi; tam epoch yapılmadı. Model/log doğrulandı ve GPU silindi. [Sonuç incelemesi](RESULTS_TR.md).
