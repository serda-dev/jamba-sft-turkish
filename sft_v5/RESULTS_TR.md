# SFT v5 pilot sonucu — 18 Ekim 2026

**Tam eğitim tamamlanmadı.** 100/1.997 adımda önceden konmuş pilot kapısı geçilmedi; kalan adımlar ve dış testler çalıştırılmadı. `COMPLETE`, denetleyicinin yedekleme/silme işinin tamamlanmasıdır; model kabulü değildir. V2 korunuyor.

| Kontrol | V2 baseline | Pilot100 | Karar |
|---|---:|---:|---|
| Kaynak temelli yönerge strict | 2/128 | 5/128 | En az +4 gerekiyordu; +3 oldu |
| Doğal çok adımlı matematik | 71/143 | 71/143 | Geçti |
| Yönerge kesilmesi | 0 | 1 | Geçti |
| Türkçe heldout NLL | 1,737054 | 1,736667 | Geçti |
| İngilizce heldout NLL | 1,837996 | 1,837976 | Geçti |
| Küçük konuşma grubunda loss | 0,003313 | 0,003785 | Göreli +%14,24; +%5 sınırını aştı |

Kod, yönerge, matematik, okuma ve reasoning aile lossları azaldı. Başarısız aile kontrolü 15 eski prosedürel konuşmadan oluşuyordu. Artışın mutlak değeri **0,000472 loss/token**; **%14 konuşma doğruluğu kaybı anlamına gelmez**. Bu grupta üretilmiş konuşma yanıtları ayrıca değerlendirilmediği için gerçek konuşma başarısının düştüğü veya korunduğu kesin söylenemez.

Bu koşu V4'teki belirgin matematik gerilemesini göstermedi. Ancak 100 adımlık gelişme, kapsamlı eğitimin başarılı olacağını kanıtlamaz. Pilot kapısı planlandığı gibi çalıştı; göreli loss sınırının sıfıra yakın ve kolay bir grupta aşırı hassas olabileceği ortaya çıktı. Eşikler sonuçları kabul edilmiş göstermek için değiştirilmedi; yeni GPU kiralanmadı.

Sonraki protokolde konuşma korunumunu doğal çok turlu üretilmiş yanıtlarla doğrudan ölçmek; çok düşük NLL için mutlak farkı da dikkate alan bir kuralı **yeni denemeden önce** sabitlemek gerekiyor. Yönerge gelişimi açısından 100 adımın erkenliği de değerlendirilmeli. Bu öneriler mevcut adayın kabulü veya başlatılmış yeni eğitim değildir.

## Kapanış

- GPU `56565846` **14:51:04 UTC** itibarıyla otomatik silindi; 16:05 UTC civarı Vast API kontrolünde instance yok.
- Model/log yedekleri otomatik boyut/hash denetimini geçti. Özel model final revision: `0b37b2644e8b04051ea0a523e81af93a830e4f38`, `checkpoints/final`.
- Başlangıç bakiye $16,640970; son gözlem $16,032863; **bakiye farkı yaklaşık $0,61**. Bu, fatura kalemi ayrıştırması değildir.
- Tamamlanmış supervisor ve watchdog timer devre dışı bırakıldı. Açık GPU yok.
- Ham sonuçlar ve receiptler özel model reposunda `run/` altında; yerel kayıtlar `/home/serda/sft-v5-run-20261018/` içinde korunuyor.
