# Baseline, geliştirme ve bağımsız test protokolü

## Tarihsel test korunumu

Eski900 MMLU+900 Belebele+200 math kısa+aynı200 math8shot+120 IFEval toplam2320 koşul değişmedi. Sorular, istemler, batch sözleşmesi, decoding ve scorer dosyaları bu PRda değiştirilmedi. Yeni `history.py` yalnız kaydedilmiş yanıtların Wilson aralıklarını, paired farklarını ve tekrar oranını raporlar. Bu eski test çok kez geri bildirim verdi; checkpoint/hyperparameter seçimi için kullanılamaz. Yeni selection kodu `split=dev` dışındaki raporu reddeder.

V2 referansı226/453/73/53/41, V5-cont242/450/68/53/51; paydalar900/900/200/200/120. Yeni GPU baseline sonucu değildir. [Tarihsel ayrıntılar](HISTORICAL_AUDIT.md).

## Yeni havuz ve kayıt

`prepare_eval.py` sabit revisionlı TQuAD, MATH-500, sanitized MBPP ve ARC-Challenge kaynaklarını indirir. Doğal Türkçe yönerge/çoklu koşul/sonlandırma/math/özet/bağlam/diyalog ve İngilizce kontroller için25 ayrı proje-yazımı görev ekler; bunlar insan katılımcılardan toplanmış gerçek konuşmalar veya öğretmen doğrulaması diye sunulmaz. Doğal görev çeşitliliğini destekleyen küçük smoke dilimidir; genel başarı testi değildir. Cevapları belirsiz örnekler manuel inceleme bekler.

Kaynak JSONdaki447 tekrarlanan kimlik önce bulunup düzeltildi; soru kimliği bağlam+normalize soru hashine bağlandı. Eğitim ve test sızıntısı için255.544 eski train kaydı ve2320 tarihi koşul taranır. Tam metin/13-kelime eşleşmesi ve paylaşılan source-document grupları konservatif olarak birleştirilir/elenir. Bu yöntem semantik/çeviri sızıntısını ispatlı dışlamaz; ön eğitim kontaminasyonu da bilinmiyor.

Kaynak/paragraf/çeviri eşlerini aynı grup içinde tutmak zorunlu. Aynı bağlamdaki farklı sorular farklı splitlere gitmez. Kaynak repo adının farklı olması metnin bağımsızlığını garanti etmez. Büyük ortak boilerplate paragrafları grupları gereğinden fazla birleştirebilir; grupların büyüklüğü ayrıca incelenmeli. Final sınırlaması kusursuz bir güvenlik sınırı değildir; araştırma iş akışı korumasıdır.

Ham dev/final dosyaları `/home/serda/capability-recovery-artifacts/` altında0700 klasör/0600 dosya olarak tutulur; Git'e ve public modele yüklenmez. Public raporda yalnız kaynaklar, hashler ve toplu sayılar vardır. Final dosyasının kimlik/sayı/hashleri hazırlanırken bilinir; aday yanıtı veya final skoru henüz yoktur. `gpu_experiments evaluate --split final`, manifestte `final_ready=true` ve aynı model/manifest için dondurulmuş dev seçim receiptini zorunlu kılar. Şimdilik `final_ready=false`.

## Ölçüm ve scorer kapsamı

- Wilson%95 başarı aralığı; aynı sorularda gained/lost ve source-group paired bootstrap.
- Gerçek token sonu, token-cap,8-gram tekrar, format, kelime/karakter/token uzunluk dağılımı ve hata kategorisi.
- İçerik ve biçim ayrı: formatın geçmesi anlam doğruluğu demek değildir. Açık uçlu görevlerde skor`null`; eksik puan otomatik başarı veya yanlış diye sayılmaz.
- Math numeric/boxed exact scorerı yalnız son yanıtı ölçer. İç çözümün geçerliliği`null`; simgesel eşdeğerlik çözücüsü yok. MATH-500 karmaşık boxed ifadelerinde false-negative olabilir. Bu yüzden henüz final için yeterli değildir.
- Kod: network kapalı, read-only, yetkisiz kullanıcı, CPU/RAM/PID kotalı, digest-pinned hazır Docker imajında örnek testleri; timeoutta yalnız oluşturulan container temizlenir. Image çekilmez. Testleri geçmek tüm girdilerde doğruluk ispatı değildir; adversarial grader güvenliği iddiası yoktur.
- NLL: `gpu_experiments nll` aynı hashlenmiş assistant-label diliminde token-ağırlıklı aile kaybı üretir. TR/EN korunumu ayrıca aynı sabit heldout manifest/token sayısıyla ölçülür. Eski2M-token NLL seti tekrar kullanılmış diagnostiktir; yeni bağımsız görev testi yerine geçmez.
- Erken/çok kısa bitirme ve gereksiz uzunluk, istenen içeriğe göre manuel rubric ister. Sırf kısa yanıt daha iyi sayılmaz. Bu etiketler henüz tamamlanmadı.

## Aday sonuçlarından önce sabitlenen ölçütler

[policy.json](policy.json) her aile için en az64 puanlanmış örnek/32 kaynak grup; accuracy düşüşü≤3pp ve paired CI alt sınırı≥−8pp; completion gerilememesi; TR NLL≤+%2, EN≤+%3. Bu pratik araştırma sınırları istatistiksel güç garantisi değildir. Birçok ailede az sayı varsa seçim engellenir. Politika, baseline/aday yanıtları görülmeden yazıldı. Sonradan değişirse yeni protokol sürümü gerekir.

Önce korunum kapıları, ardından **Pareto kümesi**; görevler keyfi tek skora toplanmaz. İki ebeveyn için karşılaştırma ayrı ayrı çalıştırılmalı; V2 math ve V5 instruction referansları kaybolmamalı. Yönergede+5pp ve kesilmede%50 göreli azalma yeni hedeflerdir. Tarihi **≥60/120, ≤12 kesilme** dönüm noktası korunur ve yalnız raporlanır. `acceptance.py` dondurulmuş aday, final hazırlığı, aile kapıları, doğru kaynağa bağlı TR/EN NLL ve eksiksiz semantic puanları kontrol eder. GPU sonuçları olmadan henüz kabul kararı üretilemez. Mevcut selection modülü NLLyi doğrulamadığını açıkça raporlar; Pareto üyeliği release kabulü değildir.

Finale dev ile seçilmiş tek aday girer; başarısız finalden sonra diğer adayları deneyerek finali geliştirme setine dönüştürmek yasaktır. Baseline yeni sette henüz ölçülmedi. Açık eksikler [FINAL_RECOMMENDATION.md](FINAL_RECOMMENDATION.md) içinde.
