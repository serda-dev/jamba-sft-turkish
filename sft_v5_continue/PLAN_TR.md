# V5 aynı veriyle devam — 18 Ekim 2026

Kullanıcı, matematik/dil gerilemesi yokken çok hassas erken eşiklerin tam eğitimi durdurmasına itiraz ederek **aynı datasetle devam edilmesini** açıkça istedi. Yeni veri toplanmıyor. Önceki pilotun başarısız kararı değiştirilmedi; bu belge kullanıcı talimatıyla değişen devam politikasını kaydeder.

## Nereden ve nasıl devam?

- Kaydedilmiş V5 100-adım modeli: `serda-dev/Jamba2-3B-Turkish-SFT-v5-20261018`, revision `0b37b2644e8b04051ea0a523e81af93a830e4f38`, `checkpoints/final`.
- BF16 model ağırlıkları mevcut. Optimizer durumları ve FP32 ana ağırlıklar dışa aktarılmamıştı. **Yeni optimizer ve FP32 ana ağırlıklar oluşturulacak; birebir optimizer resume iddiası yok.** Yeni optimizer ilk 20 adımda yumuşak LR artışı kullanır.
- Datasetin sıkıştırılmış eğitim dosyası dahil byte/hash kimliği aynı. 63.896 örnek, seed `20261018`, aynı shuffle sırası. İlk 100 × 32 = **3.200 örnek atlanır**. **101–1.997 adımları**, kalan **60.696 örnek / 1.897 optimizer güncellemesi** işlenir. Son küçük batch korunur; geçmiş örnekler baştan okutulmaz.
- BF16 tam ağırlık eğitimi + FP32 optimizer ana ağırlıkları; AdamW8bit, LR tepe `7.5e-7`, effective batch 32, 3.072-token sınırı, aynı görev bileşen ağırlıkları. LoRA/QLoRA eklenmedi.
- V2 karşılaştırma ölçümleri ve 100-adım raporları sabit hashlerle yeniden kullanılır; geçmiş baselineı tekrar ölçmek için GPU zamanı harcanmaz. V2 ve başlangıç V5 modellerinin gerçek dosya hashleri GPUda doğrulanır.

## Kullanıcı talimatıyla değişen kontrol

Önceki V5 pilotu yönergede 2→5/128, doğal matematikte 71→71/143; TR/EN NLL korunmuştu. Konuşma lossu 0,003313→0,003785 olduğu için göreli +%5 kapısı aşılmıştı. Bu yüzde, doğruluk yüzdesi değildir.

1. Erken 100-adımda zorunlu +4 strict yönerge şartı **devam engeli olmaktan çıkarıldı**. Eski test sonucu olduğu gibi korunur.
2. Aile loss sınırı artık `max(baseline × 1.05, baseline + 0.01)`; sıfıra yakın lossun küçük mutlak değişimi tek başına eğitimi kesmez. Bu kural aday sonucunu görmeden seçilmiş eski kural gibi sunulmaz: kullanıcı itirazından sonra değiştirilmiştir.
3. 400/800/1.200/1.600/1.997 adımlarında aynı yönerge, 143 doğal matematik ve aile lossları izlenir. Korunum ölçütleri checkpoint seçimini sınırlar; küçük dalgalanmalarda yeni erken pilot duruşu yaratılmaz. Nonfinite değer, süreç hatası ve bütçe sınırları işi durdurmaya devam eder.
4. Checkpoint seçimi, matematikte V2'ye göre en fazla 4/143 kayıp, dev kesilmesinde en fazla +2 ve düzeltilmiş aile-loss sınırını geçen adaylar arasında strict yönerge başarısına dayanır. Son dil korunumunu geçmeyen aday yerine bilinen 100-adım modeli korunabilir. Bu, son adımın koşulsuz daha iyi kabul edilmesi değildir.
5. Finaldeki eski regresyon testi, ayrı tutulmuş 128 math testi ve TR/EN NLL ölçümleri korunur. Eski final kabul eşikleri değiştirilmedi; üretim V2 otomatik değiştirilmez. Eğitimin tamamlanması modelin kabul edilmesiyle aynı şey değildir.

`original_pilot_gate.json` eski sonucu tutar. Yeni `pilot_gate.json` açıkça **kullanıcı onaylı devam politikasını** ve orijinal pilotun geçmediğini kaydeder; orijinal pilot geçmiş gibi gösterilmez.

## Süre, maliyet ve kayıt

- Eski V5 harcaması (~$0,61) dahil **birleşik üst sınır $8**. Harcama başlangıç kredisi $16,640970'e göre ölçülür. Yeni kiralama için süre sınırı 6 saat; eski koşuyla aradaki GPU kapalı bekleme süresi GPU çalışma süresi sayılmaz.
- Tek A100 SXM 80 GB; teklif ≤$1,35/saat. Aynı Docker image digest, yedekleme ve watchdog kullanılır. İlk ölçüm önceki 6,2 sn/adım civarındaysa kalan optimizer işi ~3,3 saat; ara değerlendirme/aktarım/final testleri ek süre gerektirir.
- Kök: `/home/serda/sft-v5-cont-run-20261018`. Servisler: `linguai-sft-v5-cont-supervisor.service`, `linguai-sft-v5-cont-watchdog.timer`.
- Yeni özel çıktı: `serda-dev/Jamba2-3B-Turkish-SFT-v5-cont-20261018`; eski pilot/model/değerlendirme değişmez.
- Hazırlık/kod/veri HF'ye kaydedilip hash doğrulandıktan sonra kiralama yapılır. Model/log hashleri doğrulanınca normal bitişte instance otomatik silinir. Hata/sert bütçe duruşunda GPU stop edilir, kurtarma için disk tutulur.
- Atlanan örneklerin tekrarlanmaması, son batch, hassas loss düzeltmesi, gerçek math gerilemesinin seçimden dışlanması ve yeni optimizer LR rampı için dört odaklı CPU testi geçti.

Tüm olaylar `OPERATIONS_TR.md` dosyasına ve özel Hub reposunun `run/` alanına yazılır. Yeni GPU kimliği ve ilk devam adımları başlatma kaydına eklenecek.

## Başlatma doğrulaması

18 Ekim yaklaşık 16:54 UTC: instance 56585829 üzerinde 110/1997 adım doğrulandı; yeni koşuda 10 güncelleme tamamlandı. [Başlatma kaydı](LAUNCH_TR.md).

## 19:30 ara kontrol ve final seçim düzeltmesi

Gerçek konuşma kontrolünde V2 ve adım1600 15/15. NLL ölçütünün faydalı checkpointleri dışlamaması ve son ağırlıkların kaybolmaması için optimizer adımları tamamlandıktan sonra final incelemesi eklendi. [Ölçümler ve değişikliklerin ayrıntısı](PROGRESS_REVIEW_TR.md).
