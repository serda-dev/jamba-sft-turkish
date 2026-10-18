# SFT v4 — tamamlanan denemeler, 18 Ekim 2026

## Karar

**Yeni model kabul edilmedi. V2 referans olarak korunuyor.** İki deneme ve bağımsız değerlendirme tamamlandı. Bu sonuç, modelin hazır olduğu anlamına gelmez. Yeni GPU kiralanmadı; tamamlanan işlerin servis ve watchdog timerı kapatıldı.

## Yapılanlar

1. FP32 optimizer ana ağırlıklarıyla küçük güncellemelerin kaybolması sorunu giderildi. BF16 çalışma ağırlıkları ve tüm 3B parametrelerin eğitimi korundu.
2. Sabit kaynaklardan 2.000 doğal Türkçe kaynak metne dayanan yönerge üretildi; 717 aritmetik ve 365 genel tekrar örneği eklendi. Toplam 3.082 örnek; assistant token payları yaklaşık %70/%15/%15. Kaynak koruma görevleri yeni serbest Teacher yanıtları değildir.
3. Etiket, EOS, split, kaynak içerik ve benchmark çakışması kontrolleri yapıldı. 15 CPU testi geçti. Dataset özel Hub reposuna kaydedildi; ayrıntı ve revision [planda](PLAN_TR.md).
4. A100 SXM 80 GB üzerinde LR=1e-5 ile A denemesi yapıldı. 100. adımda geliştirme başarısı 2/128 → 37/128 yükseldi. Ancak kesilme 0 → 4 ve Türkçe heldout NLL artışı %14,73 oldu. Pilot kapısı reddetti, kalan eğitim ve dış benchmark yapılmadı.
5. Aynı veri, V2, kabul eşikleri ve toplam $5/3 saat sınırıyla B denemesi LR=2e-6 olarak çalıştırıldı. Önceki 256 baseline yanıtı hash ve yeniden puanlama ile doğrulanarak kullanıldı; ayrıca V2 dil baselineı aynı runtime ile yeniden ölçüldü, öncekiyle aynı çıktı.
6. B pilotu geçti: geliştirme 31/128, kesilme 1, aritmetik 63/64, okuma 64/64. 771 adım / iki epoch tamamlandı. Geliştirme doğruları adım 100/200/400/600/771 için 31/35/39/47/44 oldu.
7. Geliştirmeyle seçilen 600. adım dil korunumunu geçmedi. **Dış testten önce** önceden korunumdan geçmiş 100. adıma dönüldü. Dış test yalnızca bu son adayda bir kez çalıştırıldı. Eğitimde tamamlanan 771 adım ile yayımlanan 100. adım farklıdır.
8. Son adaya 2.320 koşullu sabit test uygulandı. Sonuçlar aşağıda. Kabul başarısız: matematik, yönerge başarı tabanı ve yanıt durdurma eşikleri karşılanmadı.
9. Model/loglar HF metadata boyut ve hashleriyle doğrulandıktan sonra iki GPU silindi. Sonradan API kontrolünde açık instance yoktu. VPS'ye model ağırlığı indirilmedi.
10. CPU'da 1.536 geliştirme ve 2.320 benchmark yanıtı yeniden puanlandı: **0 fark**. Sabit test kimlikleri ve girdileri de bire bir doğrulandı. [Tekrarlanabilir denetim](audit_results.py).

## Sabit dış test

| Test | V2 | B, seçilen adım 100 | Kabul tabanı |
|---|---:|---:|---:|
| Yönerge, strict | 41/120 | 57/120 | ≥60 |
| Kesilen yönerge yanıtı | 34/120 | 24/120 | ≤12 |
| GSM8K Türkçe | 73/200 | 43/200 | ≥68 |
| GSM8K Türkçe, 8-shot | 53/200 | 34/200 | ≥48 |
| Belebele Türkçe | 453/900 | 447/900 | ≥435 |
| Türkçe MMLU | 226/900 | 245/900 | ≥208 |

Türkçe NLL 1,737054 → 1,768095 (**+%1,787**), İngilizce 1,837996 → 1,839725 (**+%0,094**). Tokenlar eşit: TR 1.596.332, EN 402.691; sabit manifest aynı. NLL korunumunun geçilmesi matematik becerisinin korunduğunu göstermedi.

Eşleşmiş GSM8K sorularında önce doğru olan **40** cevap kaybedildi, **10** yeni doğru kazanıldı. 8-shot için 28 kayıp/9 kazanım. Yönergede 8 kayıp/24 kazanım. Dolayısıyla gerileme yalnızca toplu yüzde yuvarlaması değildir.

## Veri ve seçim kontrolü hakkında çıkarım

Yeni küçük aritmetik kontrolü, eğitimdeki basit işlem ailesine yakındı. 19/64 → 63/64 yükselmesi çok adımlı metin problemlerine genellenmedi. Basit okuma kontrolü de V2'de zaten 64/64 olduğu için doyuma ulaşmıştı. Bunlar ölçüm hatası değil, korunum kontrollerinin kapsam eksikliğidir.

Kaynak metni kopyalama/biçimleme ağırlıklı paket, yönerge puanını artırırken mevcut problem çözme becerisini koruyamadı. Bunun tek nedeninin veri dağılımı olduğu kanıtlanmadı; tam ağırlık güncellemesi ve öğrenme oranının etkisi de olabilir. FP32 güncellemelerin çalışması genel model kalitesini tek başına garanti etmez.

## Sonraki deneme öncesi gerekli düzeltme

1. **Bu iki adayın üzerinden devam etme; V2'den başla.** Sabit dış test değişmeden kalsın. Dış testteki kaybedilen sorular eğitim örneğine çevrilmesin. Bu test artık görülmüştür; sonraki geliştirmelerin tamamen bağımsız nihai testi sayılmamalıdır.
2. Kaynağa sadık düzenleme görevlerini koru; fakat yalnızca bunlara yüklenme. Genel soru-cevap, kısa doğal cevap ve çok turlu konuşma tekrarını artır. Başlangıç hedefi assistant tokenlarında %40 yönerge / %30 gerçek çok adımlı problem / %30 genel tekrar; bu henüz hazırlanmış veya doğrulanmış veri değildir.
3. Basit aritmetiğin yanına eğitimden ayrı kaynak/grup bazlı **128 çok adımlı problem** kontrolü koy. İşlem zincirleri ve son cevap programla doğrulanabilen görevleri tercih et. Hazır doğal kaynak kullanılırsa referans çözümü denetlenmeden doğru kabul etme. 64 doğal okuma kontrolü de ekle; mevcut kolay kontroller yalnızca smoke test olarak kalsın.
4. Bu kontrolleri V2 ile ölç; V2'nin sıfıra veya tavana yakın olduğu kontrolleri tek korunum kanıtı yapma. İstem, hedef ve token uzunluklarını raporla; sabit benchmarkla tam/13 kelime çakışmasını ele. Cevapların kesilmesi ve tekrar oranını ayrıca ölç.
5. CPU hazırlık/audit tamamlanınca kısa V2 pilotu çalıştır. Gerçek problem kontrolünde bozulma varsa tam eğitime geçme. LR düşürmeyi veya parametre verimli eğitimi ayrı deney olarak değerlendir; şimdiki sonuçlar LoRA'nın çözüm olduğunu kanıtlamıyor. Her denemenin yeni bütçesini ve başlangıcını kaydet; eski süresi dolmuş işi yeniden başlatma.

Bu maddeler sonraki işin planıdır; hazır model veya başlatılmış üçüncü eğitim iddiası değildir. Kabul koşulları bu sonuçları geçerli göstermek için gevşetilmedi.

## Maliyet, yedekler ve sınırlamalar

- A: instance `56468842`, 17 Ekim 23:55:18 → 18 Ekim 00:13:26 UTC.
- B: instance `56470731`, GPU işi 18 Ekim 00:16:13 UTC’de başlatıldı; instance 01:09:42 UTC’de silindi.
- İki denemenin başlangıç bakiyesi $18,453537; sonraki kontrol $16,640970; **bakiye farkı $1,812567**. Son silme anı ölçümü $1,701798 idi; daha sonraki faturalama gözlemi ayrı kaydedildi. Bu bir fatura kalemi ayrıştırması değildir. Önceki $0,4566 tanı çalışması bu tutara dahil değildir.
- Son model: özel `serda-dev/Jamba2-3B-Turkish-SFT-v4b-20261018`, revision `7acde389be357602af459076a02054db56e7ae09`, `checkpoints/final`. **Reddedilen araştırma adayıdır.**
- B log arşivi: revision `c5b9d37f78de73ff8137a54c41fdb12bf7b39121`, `run/logs.tar.gz`, SHA256 `a41870fe9d68ceccfd520444f1f04cbd56667df016392852d67b9962b4e87841`.
- 10 final dosyasının uzak metadata boyut/hashleri ve log arşivi yeniden doğrulandı. Bu, ağırlıkları VPS'ye indirip model çalıştırma testi değildir; inference GPU'da yapıldı.
- Çalışma günlüğü: `/home/serda/sft-v4-run-20261018/OPERATIONS_TR.md`; özet/receiptler [results_20261018](results_20261018).
- `training_provenance.json` eğitim sonunda geliştirme seçimini (600) yazar; son fallback seçimi (100) `acceptance.json` ve final modelin `sft_provenance.json` içindedir. Önceki runtime 600. adım NLL raporunu fallback sırasında aynı dosyaya yazıp değiştirdi; sayısal reddedilen NLL burada mevcut olmadığı için uydurulmadı. Olay/provenans dil kapısının başarısız olduğunu kaydetti.
- **İş bittikten sonraki kod düzeltmesi:** reddedilen geliştirme adayının NLL raporunu ayrı dosyada koruma ve `selected_model_provenance.json` yazma eklendi. Bu düzeltme tamamlanmış A/B koşularında yoktu; gelecekteki çalışmaları etkiler. Sözdizimi doğrulandı.

Arşivlenen `attempt_b_config.json` hazırlık anının bire bir kopyasıdır; `status=PREPARED_NOT_TRAINED` çalışma sonucu değildir. Son karar `acceptance.json` içindedir. Runtime ısınmayı sabit 10 adım yapar; configteki eski `warmup_ratio` alanı burada kullanılmaz.
