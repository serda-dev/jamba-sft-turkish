# SFT v5 başlatma doğrulaması — 18 Ekim 2026, yaklaşık 14:30 UTC

**GPU kiralandı ve gerçek optimizer adımları doğrulandı: 10 / 1.997.** Bu, tamamlanmış eğitim veya kabul edilmiş model değildir. Güncel ilerleme çalışma günlüğü ve servis state dosyasında devam eder.

- Instance **56565846**, A100 SXM4 80 GB, **$1,077963/saat**. Kiralama 14:18:12 UTC; GPU işi (baseline dahil) 14:20:11 UTC.
- İlk $0,82 teklif kiralamadan önce kalktı. İlk RAM filtre sorgusu CLI birimi nedeniyle sonuçsuzdu; GPU kiralanmadan düzeltildi. Sonraki teklif aynı $8 / 6 saat sınırındadır.
- Hazırlık özel Hub reposuna kaydedildi ve dosya boyut/hashleri doğrulandı: `serda-dev/Jamba2-3B-Turkish-SFT-v5-20261018`, revision `f9be2634f118476ccf9b880dab30cb6f7b29f229`, `prepared/`.
- GPUda 20 kod dosyasının hashleri ve V2 model ağırlıkları doğrulandı. CUDA/Mamba preflight geçti.
- Yönerge baselineı 2/128; önceki V2 ölçümüyle aynı. Doğal matematik 71/143. TR/EN NLL baselineı aynı sabit manifest ve token sayılarında önceki V2 ile uyumlu.
- 3.071 tokenlık gerçek en uzun örnekte LR=0 bellek kontrolü geçti: **39,74 GiB** ayrılmış VRAM. İlk 10 gerçek adımda gözlenen tepe **45,72 GiB**. Warmup başladı; loss finite, FP32 master güncellemeleri dosyayla doğrulandı.
- İlk 10 adım 62,0 saniye; yalnız optimizer işi için kalan kaba tahmin **3,42 saat**. Dev kontrolleri, yedekler ve final testleri eklenince ilk toplam kalan tahmin **4–5 saat**. Bu erken ölçümdür; garanti veya son kalite sonucu değildir.
- Supervisor ve iki dakikalık bağımsız watchdog aktif; `Linger=yes`, oturum kapanınca iş sürer. 100-adım kapısı henüz değerlendirilmedi. Geçerse aynı koşu tek epocha devam eder; geçmezse aday kaydedilip kapatılır.
- Normal bitişte model ve log uzak hashleri doğrulandıktan sonra otomatik destroy. Hata/sert bütçe kesmesinde GPU durdurulup disk korunur. Üretim V2 değiştirilmedi.

Protokol: [PLAN_TR.md](PLAN_TR.md). Sayısal başlatma kaydı: [launch_receipt.json](launch_receipt.json).

Canlı çalışma dizini: `/home/serda/sft-v5-run-20261018`. Servisler: `linguai-sft-v5-supervisor.service`, `linguai-sft-v5-watchdog.timer`. Operasyon günlüğü `OPERATIONS_TR.md`; snapshotlar özel Hub reposundaki `run/` altında güncellenir. Büyük model ağırlıkları VPSye indirilmedi.
