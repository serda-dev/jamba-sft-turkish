# V2 üzerinden kapsamlı SFT önerisi — 18 Ekim 2026

Durum: kullanıcıyla değerlendirilen öneri. Dataset henüz hazırlanmadı; bu plan için GPU kiralanmadı ve yeni eğitim başlatılmadı. Önceki reddedilen koşuların kabul ölçütleri değiştirilmedi.

## Dayanak

V1: 63.083 eğitim konuşması, iki epoch. V2: 108.483 konuşma, epoch başına yaklaşık 59,9 milyon giriş / 30,6 milyon assistant tokenı, iki epoch, 6.782 tam eğitim adımı. V2, 7,28 saatlik eğitim sonrasında matematikte 45→73/200 ve 8-shot 25→53/200 yükseldi. Yönergede 38→41/120 sınırlı artış oldu. Bunlar tek koşu gözlemleri; veri bileşenlerinin nedensel katkısı ayrıştırılmadı.

Son V4 paketi 3.082 örnekti; assistant tokenlarının yaklaşık %70'i kaynak kopyalama/biçimleme görevleriydi. Daha dar dağılım, gerçek problem çözmeyi koruyamadı. V4b yönergede 57/120'ye çıkarken matematikte 43/200'e düştü. [Tam sonuçlar](RESULTS_TR.md).

V2 ve V4 tam ağırlık eğitimi kullandı; QLoRA kullanılmadı. AdamW8bit ifadesi optimizer durumlarını anlatır, modelin 4-bit QLoRA ile eğitildiğini göstermez. V4'ün FP32 master düzeltmesiyle küçük güncellemeler artık etkili olduğundan V2 öğrenme oranı aynen kopyalanmamalı.

## Önerilen veri kapsamı

V2'den başlayarak 60–80 bin seçilmiş konuşma; bu V4'ten kapsamlıdır, V2'nin eski 108 bin örneğinden büyük bir veri iddiası değildir. V2 train kaynakları korunum tekrarı olarak kullanılabilir. Gerçek yeni içerikle tekrarlar manifestte ayrı sayılır.

Başlangıç hedefleri **assistant hedef tokenı** üzerinden:

| Aile | Hedef pay |
|---|---:|
| Genel Türkçe yardım, açıklama ve çok turlu konuşma | %25 |
| Çok adımlı matematik ve doğrulanabilir problem çözme | %25 |
| Doğal yönerge, birleşik kısıtlar, JSON ve çıktı biçimi | %20 |
| Okuduğunu anlama, özetleme ve bağlamdan çıkarım | %15 |
| Kod, hata düzeltme ve teknik görevler | %10 |
| İngilizce genel yardım korunum tekrarı | %5 |

Oranlar hazırlanmış verinin ölçümü değildir. Diğer ailelerdeki İngilizce örnekler dahil toplam İngilizce payı ayrıca raporlanır; başlangıç hedefi %10–15. Diller bir tablo ailesi ile eş anlamlı değildir. Mikro batch başına ortalama kayıp bu token paylarını tam gradient ağırlığına dönüştürmez.

İngilizce çok turlu kaynak: eski UltraChat train tekrarı. Türkçe yardım: Atlas/Aya/merve kaynaklarının kalite süzgecinden geçmiş train örnekleri. Matematik: eski V2 GSM8K train kaynakları ve denetlenebilir ek problemler; aynı problemin Türkçe/İngilizce sürümleri aynı grupta tutulmalı. Kod/doğal reasoning kaynaklarında yayıncının `verified` etiketi bağımsız doğruluk kanıtı sayılmaz. Kaynak revizyon/lisansları eski manifestten sabitlenmeli ve tekrar kontrol edilmeli.

Kaynak metni düzenleyen V4 görevleri küçük bir alt bileşen olarak kalabilir. Doğal görevleri yalnızca sayısal şablonları çoğaltarak ikame etme. Sözcük sınırı, paragraf, kapanış, JSON, çok turlu koşul birleştirme ve doğru durma örnekleri çeşitli olsun. Çok koşullu etiketler açıkça denetlensin. Tam cevap/EOS korunmalı; sığmayan örneklerde hedef yanıt kırpılmamalı.

## Eğitim önerisi

- Başlangıç: sabit V2 final revision `c8ee3888c486fa49d8d9e798a632c9f7ce664d7a`, `checkpoints/final`.
- Öncelikli yöntem: quantization olmadan BF16 çalışma ağırlıkları, FP32 optimizer ana ağırlıkları ile full SFT. 3B için kanıtlanmış mevcut kod yolu kullanılır.
- Başlangıç LR adayı **5e-7–1e-6**; bu bir pilot aralığıdır, başarı garantisi veya tamamlanmış karşılaştırma değildir. FP32 mastersız V2 LR'siyle eşdeğer etki varsayılmaz.
- Effective batch **32**; mikro batch gerçek uzunluk dağılımında GPU ön kontrolüyle seçilir. Accumulation buna göre ayarlanır.
- Bağlam üst sınırı ilk hedef **3.072 token**. Daha uzun örnekler varsa sayıları raporlanıp ayrıca karar verilir; bağlam sınırı artırımı otomatik öğrenme kazancı değildir.
- Önce **bir epoch**; ikinci epoch yalnızca doğal geliştirme kontrolleri ve korunum iyileşiyorsa. Veri V2'ye önceden gösterildiğinden tekrarlar toplam maruziyete dahil raporlanır.
- Tek A100 80 GB, ancak FP32 masters + 3.072 token için gerçek optimizer adımlı bellek kontrolü gerekir. V4'ün 1.063 tokenda 39,74 GiB sonucu daha uzun dizide garantili sığma kanıtı değildir.

QLoRA, 4-bit taban ağırlıkları üzerinde LoRA eğitir. Daha az sıkıştırma isteniyorsa BF16 tabanla çalışılabilir. LoRA rankını düşürmek güncellemenin temsil kapasitesini azaltır; bunu kalite artırımı olarak önermiyoruz. BF16 LoRA ayrı bir seçenek olabilir, ancak Jamba'nın Mamba ve attention katmanlarının gerçek modül kapsamı/gradient akışı doğrulanmadan yalnızca q/v projeksiyonlu bir reçete uygulanmamalı. Şimdilik mevcut full SFT yolu daha az entegrasyon değişikliği gerektiriyor.

Resmi teknik kaynaklar:
- https://huggingface.co/docs/peft/main/developer_guides/quantization
- https://huggingface.co/docs/peft/main/package_reference/lora

## Kiralamadan önce tamamlanacaklar

1. Kaynak/konuşma grupları ve çeviri eşleri bazında ayır; V1/V2'de görülmüş train kayıtlarını yeni dev/teste koyma. Eski train tekrarını yeni veri diye raporlama.
2. Yeni, doğal ve aile bazında raporlanan geliştirme paketi oluştur: çok adımlı matematik, gerçek okuma/özetleme, birleşik yönergeler, konuşma ve kod. Kolay aritmetik kontrolünü yalnızca smoke test say.
3. Kod görevlerinde uygulanabilir örneklerde sandbox testleri; matematikte referans/işlem zinciri denetimi; açık uçlu görevlerde nitel inceleme ve kısıt kontrolleri yap. Otomatik biçim kontrolünü anlam doğruluğu yerine koyma.
4. Kaynak bazlı token payı, kopya, şablon yoğunluğu, dil, uzunluk, hedef maskesi ve EOS auditini tamamla. Eski testlerle tam/13 kelime çakışmasını ele; bunun semantik sızıntıyı bütünüyle dışlamadığını belirt.
5. Yeni V2 baseline ve kısa pilotu aynı geliştirme protokolüyle karşılaştır. Matematik/okuma/konuşma gerilerse tam koşuyu başlatma. Eşikleri baseline ölçümünden sonra, aday sonuçları görülmeden kaydet.
6. Görülmüş eski 2.320 benchmarkı regresyon raporu olarak koru. Yeni nihai test kaynak/grup bazında ayrılıp eğitim/ayar seçimine açılmasın. TR/EN NLL kontrolü de sürsün; tek başına beceri korunum kanıtı değildir.
7. GPU bütçe/süre sınırını gerçek dataset token sayısı ve pilot hızından hesapla. Eski V2'nin 7,28 saatini veya V4 hızını yeni koşu için doğrudan garanti sayma. Model/log hash doğrulaması sonrası otomatik silme korunur.

Öneri: daha kapsamlı ve dengeli veri + etkili güncelleme boyutunu sınırlayan full SFT. Daha çok epoch, daha yüksek LR veya daha büyük batch kendi başına daha iyi model garantisi değildir.
