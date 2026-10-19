# Son karar ve tamamlanma sınırı

## Sonraki yetkili GPU pilotu — tamamlandı

[Gerçek precision/gradyan sonuçları](PILOT_RESULTS_TR.md) önceki CPU-only durumu günceller:8 tanı optimizer updatei,6 aile×3 gradient örneği, iki private ağırlık yedeği, doğrulanmış log ve otomatik GPU silme. İlk adım sample gradient farkı%1,349/cosine0,999909. B38,29s/50,63GiB; A26,00s/44,96GiB. Bazı çiftlerde tekrarlanan negatif yön var, instruction–math için tutarlı değil. Genel görev başarısı ölçülmedi. Sonuç **production optimizer veya PCGrad geçişini henüz gerekçelendirmiyor**. Mevcut karar: doğal TR dev/semantic kapsamını tamamla, ardından basit mergeyi değerlendir; büyük SFTyi açma. Aşağıdaki önceki teslim kaydı tarihsel olarak korunur.

## Karar

**Şimdi yeni büyük SFT veya production değişimi yapma.** V2 kabul edilmiş matematik referansı olarak kalır; V5-cont1600 karşılaştırma adayı, V4-B reddedilmiş yönerge referansıdır. V5'in yönerge kazanımı ve matematik kaybı tarihsel testte gerçek gözlemdir; yeni bağımsız genelleme sonucu değildir.

## Tamamlananlar

- V1–V5/continuation geçmişi, soy/optimizer-resume ve veri/token maruziyeti denetimi.
-5 sabit referansın erişimi; gerçek remote tensor headerları, small-file hashleri, semantic tokenizer uyumu ve original modelin dışlanması.
-4-alpha gerçek tensor koordinat probu; full-model merge ve tek-model GPU-memory interpolation kodu.
- Mevcut BF16 gradient birikimi doğrulandı; ayrı FP32 mikro-birikim yolu ve gerçek Jamba A/B entrypointi. Production training değiştirilmedi.
- Kaynak-grup ayırma, exact/13-kelime dışlama, paired CI, format/anlam ayrımı, EOS/tekrar/uzunluk ve dev-only Pareto seçimi; bağımsız final koruması.
- Yeni yerel dev/final **havuzu** ve hash manifestleri; gerçek CPU Docker code-test yolu.
- V1–V5 token audit, batch-normalize katsayı hesabı, reuse-sınırlı token samplerı, precision/gradient diagnostik örnekleri.
-10 Türkçe rapor, testler, sabit policy ve feature branch. Ham/private örnekler commitlenmedi.

## Henüz tamamlanmayanlar

| İş | Engel / gereklilik |
|---|---|
| Gerçek V2/V5/4 merge dev baseline/aday ölçümleri | Yerel CUDA yok; yeni ücretli compute yetkisi yok. Önce doğal TR dev kapsamı tamamlanmalı. |
| Tam3B merge ağırlık dosyaları | Yalnız uyumlu tarif ve koordinat probu hazır. Dört tam çıktı+ebeveynler VPS boş diskini aşar. GPUda sırayla materyalizasyon destekleniyor. |
| Daha geniş precision/gradient ve görev yararı | İlk gerçek A/B ve6-aile ölçümü tamamlandı; doğal dialogue, geniş grup tekrarı ve gerçek görev yararı eksik. CPU toy, GPU kanıtı yerine konmadı. |
| Nitelikli yeni bağımsız final | TR math/yönerge/dialogue/summary/termination örnekleri az; açık uçlu anlam ve reasoning scorerı eksik; kaynak lisans/semantic-contamination incelemesi bekliyor. `final_ready=false`. |
| Yeni dengeli train dataseti | Grounded QA ayrı denetlenmiş değil; doğal TR diyalog ve family/language etiketleri eksik. Eski loss ağırlığını sampler sanmıyoruz. |
| V6 / DPO / uzun bağlam | Önceki sonuçlar henüz bu fazları açmayı gerekçelendirmiyor. DPO çifti veya uzun-context ölçümü hazır değil. |

## İlk CPU tesliminde önerilen pilot — sonradan tamamlandı

İlk öneri V2 üzerinde aynı128 örnekle iki×4 optimizer update precision karşılaştırması ve6-aile gradyan tanısıydı; o CPU teslimi anında yetki yoktu. Sonraki kullanıcı yetkisiyle **en çok$2 ve1 saat** sınırında tamamlandı; gerçek sonuçlar üst bölümde. Bu bütçe tüm araştırmanın fiyatı değildir. Tam doğal dev ve6-model merge karşılaştırması ayrı süre/kapsam hesabı ister.

İzin verildiğinde gerçek A10080GB preflight/throughput ile devam; pozitif sonuç olmadan production precision değişimi veya PCGrad yok. Merge başarılıysa yeni SFTyi atlamak önceliklidir. Başarısızlığı, bütçe veya veri kıtlığını eşik gevşeterek kapatma.

İlk teslim CPU uygulanabilir kısmıydı; sonraki yetkili GPU tanısı yukarıda kayıtlıdır. Henüz daha iyi kabul edilmiş model üretilmedi; dengeli yetenek toparlanması sağlandı iddiası yoktur.


## 19 Ekim — yeni ölçüme dayalı güncel karar

**Bu dört alpha ile yeni merged model önermiyoruz.** Altı gerçek GPU koşusunda V5 yeni kümenin dört ana ölçümünde en iyi/eşit; hiçbir aday önceden belirlenen iki-ebeveyn üstünlük kapısını geçmedi. Alpha0.70 nonregression geçti ama iyileşme kanıtlamadı. V5'in bu kümedeki noktasal üstünlüğü, tarihî kabulü başarısız araştırma checkpointini production kabul edilmiş yapmaz; bütün mevcut checkpointler korunur.

Yeni development altı ailede64/32 şartını karşılıyor. Yönerge=constraint compliance; math=finalnumeric; reading=extractive EM; conversation=gerçek insan diyaloğunda memory. Post-hoc hata denetimi, reading/memory exact-match'in anlam doğru yanıtları da eleyebildiğini gösterdi. Bu nedenle tam sohbet/okuma semantiği veya genel yetenek kaybı iddiası yapılmaz; secondary tokenF1/fact proxy de semantic entailment değildir ve seçimde kullanılmadı. Ayrılmış320 örnekte çoklu kısıt24 olduğundan tam final kapısı kapalı kalır; sonuçlar görülünce64/32 şartı düşürülmedi.

En son gerçek sampler:53,093 prepared örnek,5,197 QA+1,000 insan yazımı Türkçe görev diyaloğu;3,000,085 target token/5,525 batch, max reuse2, source exposure doğrulandı. Bu20M dengeli run veya tam V6 data readiness değildir; reading kapasitesi yaklaşık3.10M toplam exposure ile sınırlı, language/free-dialogue çeşitliliği ve lisans boşlukları sürüyor.

V6/DPO/PCGrad başlatılmadı. Bu olumsuz/kararsız merge sonucu tek başına başka SFT'yi gerekçelendirmez. Önce tespit edilen semantic-vs-format değerlendirme ayrımını yeni deneyden önce dondurmak gerekir; ücretli yeni işlem otomatik başlamaz. [Sonuç raporu](results/merge_20261019/RESULTS_TR.md), [sampler arayüzü](SAMPLER_INTEGRATION_TR.md).
