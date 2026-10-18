# V5 devam başlatma kaydı — 18 Ekim 2026, yaklaşık 16:54 UTC

**Kayıtlı 100-adım modelinden sonra gerçek eğitim adımları doğrulandı: en az 110/1.997.** Yeni koşuda 10 optimizer adımı tamamlandı, loss sonlu, FP32 güncellemeleri mevcut. Bu kabul edilmiş nihai model veya tamamlanmış epoch değildir.

- GPU **56585829**, A100 SXM4 80 GB, **$1,07963/saat**. Aynı Docker ortamı; 20 kod dosyası ve iki başlangıç/referans modelin dosya hashleri doğrulandı.
- Dataset byte/hash kimliği aynı; yeni veri toplanmadı. İlk 3.200 eğitim örneği atlandı, adım 101'den başlandı.
- Optimizer ve FP32 ana ağırlıklar sıfırdan oluşturuldu; 20 adımlık yeni optimizer ısınması var. Birebir optimizer resume değildir.
- 3.071-token bellek kontrolü geçti: 39,74 GiB. İlk devam adımlarındaki gerçek tepe yaklaşık 44,55 GiB.
- İlk 10 yeni adım yaklaşık 58,5 saniye: kalan optimizer işi kaba 3,1 saat. Kontrol/aktarım/final testleriyle ilk toplam kalan tahmin 4–5 saat, garanti değil.
- Erken +4 yönerge kapısı kaldırıldı. Düşük loss için mutlak farkı da dikkate alan devam kuralı kayda geçirildi; orijinal pilotun başarısız sonucu korunuyor. Ara kontroller checkpoint seçimi ve raporlama için sürer. Nihai kabul eşikleri değiştirilmedi.
- Önceki V5 harcaması dahil $8 sınırı, bu kiralama için 6 saat, model/log hashleri doğrulanınca otomatik silme ve bağımsız watchdog aktif. Üretim V2 değişmedi.

Protokol: [PLAN_TR.md](PLAN_TR.md). Sayısal başlatma kanıtı: [launch_receipt.json](launch_receipt.json). Canlı kayıtlar `/home/serda/sft-v5-cont-run-20261018/OPERATIONS_TR.md`, özel çıktı `serda-dev/Jamba2-3B-Turkish-SFT-v5-cont-20261018`.
