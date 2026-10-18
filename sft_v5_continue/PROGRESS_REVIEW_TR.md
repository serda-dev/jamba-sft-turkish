# Ara kontrol ve checkpoint seçimi düzeltmesi — 18 Ekim 2026, 19:30–19:40 UTC

Eğitim çalışıyor; bu kayıt nihai başarı raporu değildir. 19:30 kontrolünde 1.600/1.997 adım tamamlanmıştı. Sonraki kontrolde 1.650 adımı geçti, optimizer süreci yeniden başlatılmadı. Bütçe/izleyici hata bildirmedi.

## Ölçümler

| Geliştirme adımı | Kaynak koruma/yönerge strict | Kesilme | Tekrar | Doğal math |
|---|---:|---:|---:|---:|
| V2 baseline | 2/128 | 0 | 4 | 71/143 |
| 100 | 5/128 | 1 | 4 | 71/143 |
| 400 | 15/128 | 2 | 4 | 64/143 |
| 800 | 18/128 | 1 | 1 | 67/143 |
| 1.200 | 24/128 | 0 | 0 | 68/143 |
| 1.600 | 28/128 | 0 | 0 | 69/143 |

Bu 128 örneklik geliştirme seti kaynak koruma/biçim görevleridir; 120 örneklik dış IFEval testi veya genel model doğruluğu değildir. Matematikte son ölçüm baseline'dan 2/143 düşüktür; sıfır gerileme diye sunulmaz. Kod, okuma, yönerge ve reasoning reference lossları azaldı. Bu loss düşüşü tek başına gerçek anlam/kod doğruluğu kanıtı değildir.

## Bulunan seçim sorunu

Devam kuralındaki konuşma lossunun `baseline+0.01` sınırı da 800/1.200 civarında aşıldı. GPUdaki eski seçicinin `best` klasörü hâlâ 100. adımdaydı. Bu, eğitim durmamış olsa bile bütün ilerlemenin final ağırlık seçiminde dışlanması riskini yaratıyordu.

15 mevcut prosedürel proje-kodu hatırlama kontrolünde **gerçek yanıt üretimi** yapıldı: **V2 15/15, 1.600. adım 15/15**. JSON nesnesi tam karşılaştırılır; fazladan açıklama veya duplicate key kabul edilmez. Bu, yalnızca bu 15 kontrolün sonucu olup doğal diyalog genelinde %100 başarı iddiası değildir.

1.600. adım modeli ayrı klasöre kopyalandı ve 10 dosyasının boyut/SHA256 değerleri özel Hub revision `e88878d7ce512720636bfbab19246c9a80033527` receiptine karşı doğrulandı. Dolayısıyla konuşma testi, checkpoint yazılırken oluşmuş karışık ağırlıklarla yapılmadı.

## Eğitim kesilmeden uygulanan düzeltme

1. **GPU optimizer süreci, veri ve optimizer durumu değiştirilmedi.** Yalnız VPS gözlemcisi kısa süre yeniden başlatıldı; watchdog tekrar aktif.
2. Son **1.997 optimizer güncellemesi tamamlanmadan** hiçbir finalizasyon duraklaması yapılmaz. Bekleyici, tam `dev_1997` raporunu (128 örnek) gördükten sonra sonlandırma isteği verir. Böylece asıl runner son ham modeli `stopped/` yoluna kaydeder; 1.997. adım ağırlıkları yanlış eski seçim yüzünden kaybolmaz.
3. Bu duraklama, tüm optimizer adımları tamamlandıktan sonraki **finalizasyon devridir**. Yeni optimizer/reset veya kalan veriyi tekrar eğitme yok. Asıl runnerın ham `stopped`/acceptance kayıtları değiştirilmeden ayrıca korunur.
4. İzleyici son modeli silmeden `final_review.py` aşamasını başlatır. Bu aşama 1.997. adımın eksik ara ölçümlerini tamamlar; korunmuş 1.600 ve 100. adımlarla yalnız geliştirme verisi üzerinden karşılaştırır.
5. Konuşma korunumunda gerçek üretilmiş JSON yanıt doğruluğu baseline'dan aşağı düşmemeli. Bu küçük grubun NLL'si raporlanmaya devam eder; tek başına modelin dışlanmasına yol açmaz. Diğer aile-loss, math, kesilme ve TR/EN NLL sınırları korunur.
6. Geliştirmeyle seçilen son adayda ayrı tutulmuş 128 math ve eski 2.320 regresyon koşulu uygulanır. **Dış test sonuçları checkpoint seçmek için kullanılmaz.** Eski nihai kabul eşikleri gevşetilmedi; üretim V2 otomatik değiştirilmez.
7. Final model ve log hashleri doğrulandıktan sonra otomatik destroy. Yeni GPU kiralanmadı. Aynı $8 birleşik bütçe / 6 saatlik kiralama süresi sınırı sürüyor. Yeni bir durdurma isteği veya bütçe soft-stopu finalizasyon incelemesi tarafından geçersiz kılınmaz.

Üç odaklı regresyon testi geçti: erken/eksik raporda finalizasyonu duraklatmama, gerçek konuşma yanlışlarını lossla örtmeme ve gerçek math gerilemesini aday seçiminden dışlama. GPUya gönderilen ek dosyaların SHA256 değerleri `review_deployment.json` içinde; politika `final_review_policy.json` içinde tutulur.

**Bu politika değişikliği 1.600. adım geliştirme kontrolü sonrasında yapıldı**, sıfırdan kör değerlendirme kuralı diye sunulmaz. Kullanıcının küçük loss oynamaları yüzünden faydalı eğitimin dışlanmaması talimatını, gerçek yanıt testiyle uygular. Nihai testler henüz açılmadı.

19:30 civarı hesap bakiyesi $12,82; ilk V5 pilotu dahil toplam bakiye farkı yaklaşık $3,82. Sonraki harcama otomatik izleniyor. İlk kalan süre tahmini: optimizer için 30–40 dakika, ara/final kontrollerle toplam yaklaşık 1–2 saat; garanti değildir.
