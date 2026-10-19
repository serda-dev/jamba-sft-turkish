# Precision ve gradient accumulation denetimi

**Güncelleme:** [19 Ekim gerçek GPU A/B sonucu](PILOT_RESULTS_TR.md): ilk adım sample L2 farkı%1,349, cosine0,999909; B38,29s/50,63GiB, A26,00s/44,96GiB. Production değişmedi. Aşağıdaki CPU kayıtları önceki aşamanın kanıtıdır.

## Gerçek imported kod yolu

`sft_v4/supervisor.py` ve `sft_v5/supervisor.py`, `research/instruction_following_20261017/pilot_runtime.py` dosyasını GPU payloadına **precision.py** adıyla koyuyor. Yerel arşivdeki `/home/serda/sft-v4-run-20261018/hub_publish/precision.py` de incelendi. Yalnız fonksiyon adına bakılmadı.

V4/V5/continuation çalışma ağırlıkları BF16. Her mikro lossun backwardı `.grad` alanına BF16 topluyor. `copy_gradients_to_master` ancak tüm mikro-batchler bitince `p.grad.detach().float()` çağırıyor. FP32 clipping, AdamW8bit'in FP32 master parametre güncellemesi, ardından BF16 working kopyası var. FP32 masters küçük adımları saklıyor; önceki BF16 mikro toplamını sonradan FP32ye çevirmek kaybolan toplama hassasiyetini geri getirmiyor.

Birbirinden ayrı dört konu: BF16 working parametre; BF16 mikro gradient; FP32 master güncellemeler; BF16 forwarda son yuvarlama. Optimizer durumlarının8bit olması ayrı bir beşinci ayrıntı.

## Uygulanan araştırma referansı

[precision.py](precision.py) mevcut yolu ve `fp32_micro` yolunu içerir. Yeni yolda her mikro backwarddan sonra BF16 gradient FP32 master `.grad` bufferına eklenir; working gradient temizlenir. Tied parametreler `named_parameters()` ile bir kez sayılır. Kullanılmayan gradient `None` kalır. FP32 forward veya Mamba kernel dtype değişimi yapılmaz. Mikro gradyanın hesaplanma hassasiyeti hâlâ BF16dır; full FP32 gradient referansı diye adlandırılmaz.

## Bu tur ölçülen

[cpu_precision_probe.json](results/cpu_precision_probe.json): CPUda tek BF16 parametre, ilk mikro gradient1.0 ve31 adet0.001. Mevcut toplam **1.0**, FP32 mikro toplam **1.0309829711914062**; BF16 mikro değerlerinin FP64 toplamıyla aynı. Bu kontrollü dizide kayıp mekanizmasını yeniden üretir. **Jamba/GPU ölçümü değildir; gerçek eğitim gradyan hatası oranı değildir.** CPU testleri ayrıca master birikimi, final BF16 yuvarlama, tied parametre ve NaN reddini kontrol eder.

## Hazır GPU A/B

`gpu_experiments precision`: aynı hashlenmiş checkpoint,128 aynı örnek, aynı sıra/seed/objective-weight, LR7.5e-7, micro1/accum32,4 adım, AdamW8bit. Her kol sıfırdan yüklenir; clip ve optimizer aynı. Önce gradient norm/sample cosine, sonra FP32 update ve BF16 working değişimi, sıfır update, loss, peak allocated/reserved VRAM ve süre kaydedilir. İlk adım aynı ağırlıkta kontrollü karşılaştırmadır; sonraki adımlar ayrışan yörüngedir. Koordinat örnekleri tam-gradient cosine değildir. Alternatif LRler ancak bu kontrol sonrasında.

İlk CPU tesliminde yeni GPU ölçümü yoktu; sonraki yetkili pilot yukarıda bağlantılıdır. Önceki32train/128 tanı GPU sonucu tarihsel kanıt olarak korundu; yeni mikro-birikim A/B sonucu gibi sunulmadı. Production eğitim runtimeı değiştirilmedi. Geçiş için hâlâ görev korunumu/yararı ölçümü gerekir.
