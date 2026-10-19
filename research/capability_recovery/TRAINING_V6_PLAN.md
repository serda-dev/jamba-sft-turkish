# Koşullu V6 ve tercih eğitimi kararı

**19 Ekim güncellemesi:** [Yetkili ilk GPU tanısı tamamlandı](PILOT_RESULTS_TR.md). İki kolun4'er updatei64,29s, gradient tanısı10,41s; toplam kiralama/yedek/silme486,8s. Aşağıdaki eski hız dayanakları ve koşullu SFT planı korunuyor. Bu yetki büyük V6/DPO eğitimi için değildir.

**Şu an V6 gerekçelendirilmiş veya başlatılmış değildir.** Öncelik ölçüm kapsamı, merge ve sayısal/gradyan tanısı. En yeni checkpoint otomatik başlangıç değildir. Üretim V2 korunur; V5 ve V4-B araştırma referansıdır.

## V6yı açacak kanıt

1. Yeterli doğal TR/EN dev, semantic/scorer doğruluğu ve source-group ayrımı.
2. V2/V5/4 alpha dev sonuçları; basit merge yararlıysa yeni SFT gerekmeyebilir.
3. Aynı örnekli precision A/B ile güncelleme yolu seçimi; production kodu CPU toy sonucu nedeniyle değiştirilmez.
4. Çoklu kaynak gruplarında task-gradient ölçümü; PCGrad ancak tekrarlanan anlamlı çatışma ve dev faydasında.
5. Gerçek aile/dil çeşitliliği, EOS/mask ve reuse tavanı geçmiş eğitim karışımı.
6. Ayrı ücretli compute bütçesi ve süre yetkisi.

## Gerekirse küçük pilot

Seçilen başlangıç V2, V5 veya başarı gösteren merge olabilir. Full BF16 forward, kanıtlanan FP32 update yolu; assistant-only etiketler ve gerçek EOS; mevcut çalışan Jamba runtime. LR adayları5e-7/7.5e-7/1e-6, aynı data/order/batch ve kısa kontrollü pilot; daha yüksek LR/daha büyük batch otomatik iyileşme değildir. İlk epoch1, ikinci epoch otomatik değil. Samplerın actual sampled indices, input/target/weighted tokenları ve kaynak/tekrar sayıları checkpoint provenansına bağlanmalı.

FP32 masters ve optimizer durumu kaydedilmezse restart **exact optimizer resume** diye sunulamaz. Tam resume için scheduler, scaler varsa durumu, RNG, sampler cursor/sıra, master parametreler, optimizer states ve data hashleri korunmalı. Şimdiki araştırma runnerı kısa disposable deney yapar; uzun V6 resume altyapısı diye sunulmaz.

Checkpointler dev korunum kapıları ve Pareto ilişkisiyle seçilir. NLL küçük oynadı diye gerçek yanıtı doğru modeli otomatik eleme; gerçek math/code/reading geriledi diye yönerge artışını gerekçe gösterme. TR/EN NLL yine ayrı guarddır. Finali tekrar tekrar açarak aday seçme veya eşik düşürme yok.

## Ayrı sonlandırma/preference yolu

Kısa doğru yanıt, uzun gerekçeli yanıt, tam JSON, paragraf/son ifade ve doğal konuşma bitişi ayrı örneklenmeli. Gerçek train/dev model hatalarından500–1000 aynı-prompt chosen/rejected çifti ancak bağımsız içerik+koşul denetimiyle hazırlanır. Şimdilik böyle bir hazır çift dataseti veya DPO sonucu yok. Sadece kısa olması chosen nedeni değildir; eksik doğru görünen yanıt reddedilir. Final benchmark yanıtları tercih datasına alınmaz.

Merge/SFT bitişi düzeltmiyorsa küçük DPO pilotu ve no-DPO baseline; math/read/code/dil kaybında reddet. Büyük RL/reward platformu gerekmiyor.

## Compute ve operasyon

V5 gerçek A10080GB kaydı:1890 yeni adım için11.709,6s, yaklaşık6,20s/adım; ara değerlendirmeler dahil. İlk10 adım58,48s, peak44,55GiB; son kayıtta45,80GiB. Eski scalar tanıdaki~0,6s/adım daha kısa sequence içindi; yeni örneklere doğrudan taşınmaz.

İlk araştırma işi tüm SFT değil:128 örnek, iki×4 update precision ve18 mikro backwardlık6-aile gradient kontrolü. V5 adım süresinden salt precision training yaklaşık47–50s tahmin edilir; yükleme/transfer/derleme/diagnostik farkı hariç. Toplam GPU işi birkaç dakikadan uzun olabilir; kesin süre ölçülmedi. Yeni dev6-model üretiminin süresi ayrıca gerçek output-token sayısı ve batch4 throughput ile ölçülmeli; bunu birkaç training adımıyla fiyatlamayın.

İlk CPU planında GPU yetkisi yoktu; sonraki$2/1saat yetkili pilotta A10080GB kullanıldı, gerçek FP32-micro peak50,63GiB ölçüldü. Eski$1,08/saat fiyat güncel teklif değildir. Açık USD/süre tavanı, bağımsız watchdog, hashli persistent yedek ve doğrulanmış kapanış uygulandı. `gpu_experiments` CLIı kendisi kiralama yapmaz; tek yetkili işi `pilot_supervisor.py` yönetti. Bu başarı uzun V6 bütçesinin veya tam eğitim resume altyapısının doğrulandığı anlamına gelmez.

## Daha sonra uzun bağlam

Dengeli kalite sağlandıktan sonra4K/16K/32K/64K; retrieval pozisyonları, uzun/multi-doc evidence, prefill/decode/cache/VRAM ve karakter/sn. Qwen2.5-3B ve Qwen3-4B uygun aynı donanım/bütçe ile;4B boyut farkı açıkça yazılır. Mimari maksimum context, gerçek uzun bağlam anlama kanıtı değildir. Bu faz henüz uygulanmadı ve genel kalite öncesi ertelendi.
