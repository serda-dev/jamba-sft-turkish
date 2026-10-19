# Son karar ve tamamlanma sınırı

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
| Precision A/B ve gradient cosine | Gerçek GPU gerektiriyor. CPU toy, GPU kanıtı yerine konmadı.6 aile verisi hazır; doğal dialogue eksik. |
| Nitelikli yeni bağımsız final | TR math/yönerge/dialogue/summary/termination örnekleri az; açık uçlu anlam ve reasoning scorerı eksik; kaynak lisans/semantic-contamination incelemesi bekliyor. `final_ready=false`. |
| Yeni dengeli train dataseti | Grounded QA ayrı denetlenmiş değil; doğal TR diyalog ve family/language etiketleri eksik. Eski loss ağırlığını sampler sanmıyoruz. |
| V6 / DPO / uzun bağlam | Önceki sonuçlar henüz bu fazları açmayı gerekçelendirmiyor. DPO çifti veya uzun-context ölçümü hazır değil. |

## En küçük sonraki deney

Önce bütçesiz yapılabilecek veri/rubric eksiklerini tamamla. GPU için ilk aşama tüm SFT değil: V2 üzerinde aynı128 örnekle iki×4 optimizer update precision karşılaştırması,6-aile gradyan tanısı. Önerilen araştırma sınırı **en çok$2 ve1 saat**, ancak **yetki verilmedi** ve güncel teklif alınmadı. Bu bir harcama taahhüdü veya tüm araştırmanın fiyatı değildir. Tam doğal dev ve6-model merge karşılaştırması ayrı süre hesabı ister.

İzin verildiğinde gerçek A10080GB preflight/throughput ile devam; pozitif sonuç olmadan production precision değişimi veya PCGrad yok. Merge başarılıysa yeni SFTyi atlamak önceliklidir. Başarısızlığı, bütçe veya veri kıtlığını eşik gevşeterek kapatma.

Bu teslim araştırmanın CPU uygulanabilir kısmıdır. Henüz daha iyi kabul edilmiş model üretilmedi; dengeli yetenek toparlanması sağlandı iddiası yoktur.
