# Checkpoint birleştirme raporu

## Doğrulanan

V2, V5-cont1600, V4-B ve CPT sabit revisionları erişilebilir. V2/V5/V4-B:462 saklanmış tensor,3.039.823.232 eleman, BF16, aynı isim/şekil/sıra; vocab69.632. Embedding/output head configte tied; safetensors yalnız embeddingi saklıyor. Çalışma anındaki ortak pointer GPU yükleyicide ayrıca zorunlu denetleniyor; bu son kontrol henüz çalışmadı.

Token ID sözleşmesi ve chat template aynı. V2 tokenizer dosyasındaki kayıtlı left-padding objesi diğerlerinde `null`; byte hashleri farklı, padding/truncation dışındaki **tüm** tokenizer semantiği aynı. Birleştirmede token ID/normalizer/model/added-token farkı kabul edilmez. Generation padding açıkça left; hedefler kesilmez.

Orijinal AI21 vocab65.536; embedding şekli ve serialization farklı. Doğrudan merge dışında tutuldu. Safetensors saklanmış eleman sayısı tied headin tekrar saklanmasından etkilenebilir; onu benzersiz parametre sayısıyla karıştırmayın.

[checkpoints.json](results/checkpoints.json): küçük dosyaların gerçek SHA256'ları; uzak LFS payload SHA256 beyanları; her shardın byte-range ile gerçek isim/şekil/dtype/offset/boyut kontrolü. Tam weight payload hashleri indirilerek yeniden ölçülmedi; rapor bunu ayrı alanla gösterir.

## Dört aday

`θ=(1−α)θV2+αθV5`, α=0.15/0.30/0.50/0.70. FP32 hesaplama, BF16 inference/export. İsimle eşleme, shard başına sınırlı bellek, nonfinite kontrolü, yeni çıktı dizini, ebeveyn koruma ve hash receipt: [merge.py](merge.py). Parametre siparişi aynı olsa da `zip(state_dict.values())` kullanılmaz.

Gerçek uzak V2/V5 tensorlarından grup başına en fazla2048 koordinatta dört katsayı uygulandı; tüm sonuçlar finite. [merge_probe.json](results/merge_probe.json). İlk embedding kesitinde α.15'in%90,14'ü V2 ile aynı BF16 değere yuvarlandı. Bu küçük alpha'nın tüm modelde etkisiz olduğunu kanıtlamaz; BF16 exportun etkisini ayrıca ölçmek gerekir. İlk koordinatlar rastgele örneklem değildir. Tied output head ayrı bir bağımsız örneklem değildir.

**Tam3B adaylar henüz oluşturulmadı veya inference ile ölçülmedi.** [merge_candidates.json](results/merge_candidates.json) sabit, uyumlu tariflerdir; model ağırlığı diye sunulmaz. VPS başlangıcında22GB boş disk vardı; iki ebeveyn ve dört yoğun çıktı yaklaşık36,5GB ister. Bu yüzden dört tam modeli VPS'ye yığmak yerine GPU entrypointi her adayı sırayla tek model belleğinde oluşturup ölçebilir. Kod local checkpoint exportunu da destekler. Yalnız byte dosyası yazmak yetenek deneyinin yerine geçmez.

## Sonraki küçük deney

Önce yeni devin kapsamını tamamla. Aynı runtime/bütçelerle iki ebeveyn+4 aday; görev bazlı paired fark ve completion/NLL. Başarılı merge yoksa V4-B, layer-selective veya task-vector denemesi açılmaz. Nonfinite tensor, tokenizer/shape farkı veya ciddi ara ağırlık bozulmasında reddet. Hangi alpha'nın daha iyi olduğu şu an bilinmiyor.
