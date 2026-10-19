# Yetkili ilk GPU pilotu — 19 Ekim 2026

Kullanıcının «yetki verdim, devam et» yanıtıyla önceki önerilen **$2 toplam /1 saat** pilot sınırı yetkilendirildi. Yeni büyük SFT, DPO veya production değişikliği bu yetkinin kapsamı değildir.

## Sabit deney

V2 `c8ee3888c486fa49d8d9e798a632c9f7ce664d7a`, `checkpoints/final`. Bütün indirilen ağırlık payloadları SHA256 ile doğrulanır. Aynı128 örnek/seed/sıra/objective ağırlığı, micro1/accum32,4 update, LR7.5e-7, AdamW8bit. Mevcut BF16 gradient toplamı ile FP32 mikro toplamı karşılaştırılır; forward/kernel BF16 kalır. İlk update aynı ağırlıkta doğrudan karşılaştırmadır; sonraki updateler farklılaşan yörüngelerdir.

İki kolun son BF16 ağırlıkları ayrı private araştırma reposuna yedeklenir; boyut/hashleri GPUda ve VPSden bağımsız doğrulanır. FP32 master/optimizer/scheduler resume durumu kaydedilmez; bu4-adımlık disposable tanı için **birebir optimizer resume desteklenmiyor**. Production checkpoint olarak seçilmez.

Ardından değişmemiş V2 üzerinde math/reasoning/instruction/reading/code/English retention için3'er ayrı stored grup ile gradient normları ve örneklenmiş cosine ölçülür. Doğal TR dialogue eksikliği korunur;6-aile kısmi tanı olduğu açıkça raporlanır. Negatif cosine tek başına gerilemenin nedeni sayılmaz. Eski/yeni final performans benchmarkı bu pilotta açılmaz.

## Kaynak ve bütçe

Başlangıç kredi **$11,387318**. Güncel aramada A100 SXM4 80GB teklif `54677825`:35GB disk dahil **$0,996389/saat**, up$0,00390625/GB, down$0,00260417/GB. Teklifin müsaitliği kiralama anında değişebilir; rental sonucu çalışma günlüğüne yazılır. Önceden çalışan digest-pinned `serdadev/jamba2-cpt` imajı kullanılır.

Bağımsız30sn watchdog, gecikmeli faturaya karşı30sent rezervle zaman/fiyat tabanlı harcamayı ve gerçek kredi farkını birlikte denetler. Güvenli kapanış55dk veya$1,80 konservatif eşikte başlar; user service ayrıca3500sn ile sınırlıdır. Ana izleyici çalışmazsa watchdog yalnız tam pilot labelı eşleşen instanceı siler. Sert sınırda sonuç yedeklemeyi sonsuza kadar beklemek yerine harcama sınırı önceliklidir. Normal bitişte checkpoint receiptleri ve sonuç/log arşivi doğrulanıp instance silinir.

Özel yedek: `serda-dev/Jamba2-3B-Capability-Pilot-20261019`. VPS çalışma dizini: `/home/serda/capability-pilot-20261019`. HF token mevcut özel oturumdan600 izinle iletilir; Vast anahtarı VPSden çıkarılmaz. Ham private örnekler Git'e alınmaz. Kod/data manifesti payload içindedir; kaynak helperlar mevcut CPT kodundan alias adlarıyla paketlenir.

Launch öncesi **27 CPU testi** ve systemd unit doğrulaması geçti. Bu belge launch reçetesidir; ölçülmüş GPU sonucu değildir. Tamamlanma ve maliyet sonucu ayrıca yazılacak.
