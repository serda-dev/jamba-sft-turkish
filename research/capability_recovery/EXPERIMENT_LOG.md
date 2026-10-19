# Deney ve uygulama günlüğü

İlk CPU teslimi19 Ekim2026; bu aşamanın harcaması **$0** idi. Sonraki yetkili GPU pilotu aşağıda ayrıca kayıtlıdır. Sayılar ölçüldüğü kapsamla raporlanır.

| Sıra | İş / konfigürasyon | Sonuç / kanıt |
|---|---|---|
|1| Yerel ağaç ve güncel GitHub `main` karşılaştırması | main81e96e…; V1 sonuç JSONu yerelde eksik, uzak blobdan geri getirildi. Uzak feature branch bu main üzerinden açıldı. |
|2| V1–V5 manifest/config/runtime/provenans ve ilgili araştırma denetimi |143 kaynak dosyanın hash envanteri; soy ağacı doğrusal değil. |
|3| V2/V5/V4-B/CPT/original uzak small-file+range header kontrolü |5/5 erişilebilir; V2/V5/V4-B uyumlu. İlk byte-hash kontrolü tokenizerı reddetti; gerçek fark yalnız padding. Tam vocab/tokenizer semantiği denetlendi. Original farklı vocab, merge dışında. |
|4| V1–V5 train/validation streaming sayımı |V1 assistant17.618.670; V5 train63.896 ve19.884.747 target. V5 bütün mask/EOS decode:0 hata. |
|5| V5 aynı seed/batch objective maruziyeti |Reading gerçek katsayı%13,20; hedef%15. Instruction%19,03; hedef%20. Selected1600 hedef15.938.479 token. |
|6| Gerçek V2/V5 tensorlarında4-alpha koordinat probu |6 mimari gruptan mevcut olanlarda en fazla2048 koordinat; finite. Tied output head ayrı olmadığı için bağımsız head probu yok. Tam model/inference sonucu değil. |
|7| CPU gradient accumulation ayrıştırması |32 kontrollü mikro gradient: mevcut1.0, FP32 mikro1.03098297. Jamba/GPU sonucu değil. |
|8| Eski yanıt istatistiği yeniden analiz |İlk analiz IFEvalde `gold` alanı varsayımıyla durdu; opsiyonel alan düzeltildi. Prompt/gold kimliği kontrolü sonrası CI/gained/lost/tekrar üretildi. Yeni inference yok. |
|9| Yeni kaynak havuzu hazırlığı |İlk hazırlık yinelenen TQuAD ID nedeniyle sonlandırıldı;447 kaynak tekrarına deterministik kimlik/dedup eklendi. Başarısız çıktı final kabulüne açılmadı. |
|10| Tarihsel sızıntı taraması |255.544 eski train kayıt ve2320 test koşulu; exact/13kelime+source group. Proje-yazımı görevlerin public-pool kotasında kaybolması düzeltildi; yeni havuz sürümü oluşturuldu. |
|11| Kısa prompt denetimi düzeltmesi |Chat role metinleri exact eşleşmeyi bozabileceğinden her mesaj içeriği ayrıca karşılaştırıldı; regresyon testi ve yeni tam tarama. Python set sırasından kaynaklanan answer-list nondeterminizmi sorted ile giderildi. |
|12| Diagnostik veri/sampler dry-run |128 precision;6 aile×3 gradient örneği. Doğal TR dialogue eksik.3M-token sampler grounded_qa yokluğunda910 tokenda fail-closed durdu; eğitim dataseti gibi sunulmadı. |
|13| Gerçek CPU Docker correctness/timeout testi |Doğru fonksiyon geçti; yanlış fonksiyon reddedildi; sonsuz döngü timeout; kendi containerı temizlendi. Yeni imaj indirilmedi. |
|14| CPU numerik/integrasyon testleri |Test sonucu `results/test_report.json`; GPU gerektiren deneyler koşulmadı. |
|15| Türkçe10 rapor, CLIlar, policy ve branch commitleri |PR araştırma/CPU uygulamasıdır; GPU sonuçları veya hazır V6 iddiası yok. |

Eski ~$5,25 V5 pilot+devam harcaması yeni çalışmanın maliyeti değildir. Önceden silinen GPU yeniden açılmadı. Final havuzunun skoruna bakılarak alpha/LR seçilmedi. Veri kapsamı/kimlik düzeltmeleri aday çıktısı görülmeden yapıldı; kaynak verinin doğruluğu bu nedenle otomatik sertifikalı sayılmaz.

## Sonraki kullanıcı yetkisiyle GPU yürütümü

- Kullanıcı «yetki verdim, devam et»: önce önerilen$2/1 saat precision/gradyan pilotu yetkilendirildi; ayrı `PILOT_LAUNCH_TR.md`.
- Müsaitliğini yitiren ilk teklif iki kez `no_such_ask` ile reddedildi; oluşan instance olmadığı ve kredinin değişmediği doğrulanarak yeni teklife geçildi. CLI exit0/stderr hata JSONu adapterı düzeltildi; credential alanları kayıtlara yazılmaz.
- Instance56737582, teklif46878740, A10080GB, $0,964815/saat;30sn bağımsız watchdog. Kaynak V2 payload hashleri ve Mamba kernel kontrolü geçti.
- İki kolun4'er updatei ve6 aile×3 backward tamamlandı. İki kolun BF16 ağırlıkları private Hubda bağımsız hash doğrulandı. Optimizer/master state kaydedilmedi; birebir optimizer resume yok.
- Raw precision117MB; Git'e değil private24MB log arşivine kondu. Arşiv SHA256 doğrulandı. Sonuç/görsel/receiptler `results/pilot_20261019/` içinde.
-19 Ekim14:06:54 UTC GPU silindi; API boş instance listesi döndürdü.14:09 UTC gözlenen bakiye farkı~$0,17. Biten watchdog kapatıldı; production V2 ve benchmark eşikleri değişmedi.
- [Deney sonuçları ve karar](PILOT_RESULTS_TR.md). Source-group/language sınırlamaları ve timing warmup eksikliği açıklandı. PCGrad veya yeni SFT açılmadı.
