# Jamba Türkçe yetenek korunumu araştırması

Güncel: **[Yeni Türkçe development seti ve V2/V5 merge çalışması](MERGE_EXECUTION_TR.md).** Altı development ailesi64 örnek/32 grup kapısını geçti; iki ebeveyn ve dört alpha gerçek A100 üzerinde karşılaştırıldı. **[Sonuç: üstün merge gösterilemedi](results/merge_20261019/RESULTS_TR.md)**; instance yedek/hash kontrolünden sonra otomatik silindi. **[Gerçek eğitim sampler arayüzü](SAMPLER_INTEGRATION_TR.md)** 3.000.085 assistant target tokenı5.525 batch üzerinden saydı. Yeni büyük SFT/production değişimi yok.

Önceki **[precision/gradyan pilotu](PILOT_RESULTS_TR.md)** tamamlandı ve o instance silindi; yeni merge kiralaması ayrı5USD/4saat yetkisi ve watchdog ile yürütülür. Tarihî CPU denetimi/karar kayıtları aşağıda korunur.

## Okuma sırası

1. [Geçmiş denetimi](HISTORICAL_AUDIT.md)
2. [Kök neden hipotezleri](ROOT_CAUSE_HYPOTHESES.md)
3. [Baseline ve değerlendirme](BASELINE_AND_EVAL.md)
4. [Birleştirme](MERGING_REPORT.md)
5. [Precision](PRECISION_AUDIT.md)
6. [Gradient çatışmaları](GRADIENT_INTERFERENCE.md)
7. [Veri denetimi](DATA_AUDIT.md)
8. [Koşullu V6/DPO/uzun bağlam](TRAINING_V6_PLAN.md)
9. [Deney günlüğü](EXPERIMENT_LOG.md)
10. [Son karar ve açık işler](FINAL_RECOMMENDATION.md)

## CPU komutları

Repo kökünde mevcut venv ile (`python` yerine `/home/serda/.local/share/linguai-training/venv/bin/python` kullanılabilir):

```bash
python -m pytest -q research/capability_recovery/tests
python -m research.capability_recovery.checkpoints
python -m research.capability_recovery.audit --decode-v5
python -m research.capability_recovery.history
python -m research.capability_recovery.merge_probe
python -m research.capability_recovery.cpu_precision_probe
```

Gerekli CPU bağımlılıkları mevcut ortamda torch2.6.0+cpu, transformers4.56.1, safetensors, huggingface_hub, pandas/pyarrow, pytest. Bunlar GPU runtimeını değiştirme talimatı değildir. GPUda geçmiş çalışmış digest-pinned Jamba imajı ve kendi paket sürüm kanıtı kullanılır; deneyler gerçek sürümleri kaydeder. Tokenlar CLI argümanına veya Git'e yazılmaz; mevcut HF oturumu kullanılır.

Aşağıdaki **eski smoke/diagnostic havuzunu** yeniden üretir;64/32 merge development kabulünü sağlamaz. Geçmiş tanıyı tekrarlamak için **yeni** output-dir kullan:

```bash
python -m research.capability_recovery.prepare_eval \
  --output-dir /home/serda/capability-recovery-artifacts/eval-new \
  --tokenizer /home/serda/sft-v2-run-20261015/tokenizer \
  --historical-cases /home/serda/benchmark-run-20261014/cases.jsonl \
  --training-files /home/serda/sft-run-20261013/train.jsonl.gz \
  /home/serda/sft-v2-run-20261015/train.jsonl.gz \
  /home/serda/sft-v3-run-20261017/train.jsonl.gz \
  /home/serda/sft-v4-run-20261018/train.jsonl.gz \
  /home/serda/sft-v5-run-20261018/train.jsonl.gz
python -m research.capability_recovery.prepare_diagnostics \
  --encoded /home/serda/sft-v5-run-20261018/train.jsonl.gz \
  --messages /home/serda/sft-v5-run-20261018/train_messages.jsonl.gz \
  --output-dir /home/serda/capability-recovery-artifacts/diagnostics-new
```

Yeni manifestin `final_ready=false` olması bilinçlidir; sayı büyük olsa da kalite koşulları eksik. Hazır yerel havuz/diagnostik sürümleri `results/*_manifest.json` ile tanımlıdır. Kaynak lisans/doğruluk boşlukları raporlanır.

## Yeni merge ve training-pool akışı

`prepare_merge_eval` revizyonu dondurulmuş doğal kaynakları indirir, eski eğitim/benchmark overlaplarını çıkarır. `curate_math` pre-output çeviri/aritmetik kalite incelemesini uygular. `validate_merge_eval --prepare-budgets` tüm gerçek skorlayıcıları çalıştırır ve uzun yönergeleri yapay kısa cape sıkıştırmaz. İsim/sayı ikamesiyle binlerce şablon sorusu üretilmez. Tam command argümanları `--help` ve [işlem kaydında](MERGE_EXECUTION_TR.md).

```bash
python -m research.capability_recovery.prepare_merge_eval --help
python -m research.capability_recovery.curate_math --help
python -m research.capability_recovery.validate_merge_eval --help
python -m research.capability_recovery.prepare_training_pool --help
python -m research.capability_recovery.merge_experiment \
  --parent-a /models/v2 --parent-b /models/v5 \
  --manifest /artifacts/merge-eval/manifest.json --output-dir /results/merge --batch-size 16
python -m research.capability_recovery.summarize_merge \
  --results-dir /results/merge --output-dir /results/merge-summary
```

Eski `selection.py` on bir ailelik tam araştırma kapılarını korur; altı ailelik merge feasibility için `merge_experiment` kullanılır. Bunun sonucundan production kabulü çıkmaz. Ayrılmış doğrulama yalnız development seçimi dondurulduktan sonra açılır; çoklu kısıt final64/32 şartını karşılamıyorsa bu eksik saklanmaz. Gerçek train DataLoader/optimizer-window bağlama, kaynak sınıfları ve kapasite sınırları [sampler dokümanında](SAMPLER_INTEGRATION_TR.md).

## GPU entrypointleri — yerel deney komutları kiralama yapmaz

Aşağıdakiler **hazır, indirilmiş ve hash doğrulanmış local checkpoint** yollarıyla ve CUDA ortamında çalışır. Bu komutların yazılması ücretli çalıştırma yetkisi değildir. İlk preflightta mevcut kernel/runtime ve gerçek bellek doğrulanmalı. Bütçe/watchdog/yedek eski supervisorın yeni araştırma konfigürasyonunda ayrıca kurulmalı.

```bash
python -m research.capability_recovery.gpu_experiments precision \
  --model /models/v2 --data /artifacts/precision.jsonl --output /results/precision.json
python -m research.capability_recovery.gpu_experiments interference \
  --model /models/v2 --data /artifacts/interference.jsonl \
  --families math reasoning instruction reading code english_retention \
  --output /results/gradient_partial.json
python -m research.capability_recovery.gpu_experiments evaluate \
  --model /models/v2 --manifest /artifacts/eval/manifest.json --batch-size 4 \
  --output /results/v2_dev.json
python -m research.capability_recovery.gpu_experiments evaluate \
  --model /models/v2 --parent-b /models/v5 --alpha 0.30 \
  --manifest /artifacts/eval/manifest.json --batch-size 4 --output /results/a030_dev.json
python -m research.capability_recovery.gpu_experiments nll \
  --model /models/v2 --data /artifacts/heldout_encoded.jsonl --output /results/v2_nll.json
```

Aynı dev komutu V5 ve diğer alpha tarifleri için uygulanır. Kod testini gerçek çalıştırmak için hazır local Python imajının digestini `--code-image repo@sha256:...` ile ver; yoksa coding sonucu`null` kalır. GPU olmadan entrypoint açık hata verir; CPUda model çalıştırılmış gibi sonuç üretmez.

Tam local merge isteğe bağlı:

```bash
python -m research.capability_recovery.merge --parent-a /models/v2 --parent-b /models/v5 \
  --alpha 0.30 --output /models/research-a030
python -m research.capability_recovery.selection --reference /results/v2_dev.json \
  --candidates /results/a030_dev.json --policy research/capability_recovery/policy.json \
  --output /results/selection_vs_v2.json
```

İki ebeveynle karşılaştırmayı ayrı sakla. Seçilen aday materyalize edildiğinde gerçek export dosya hashleriyle yeniden dev doğrulaması ve NLL yapılmalı. Final için aynı manifest/model receiptini içeren dondurulmuş seçim kaydı gerekir; hazır olmayan final açılmaz. `acceptance.py --help` bağımsız final girişlerini listeler. Bu yerel değerlendirme CLI'ları modeli productiona koymaz, GPU kiralamaz, public ağırlıkları değiştirmez veya otomatik model indirmez. VPS deploymentına özel `merge_supervisor` kiralamayı yalnız dışarıdaki açık yetki/bütçe konfigürasyonu ve bağımsız watchdog ile yürütür; mevcut çalışmanın5USD/4saat yetkisi gelecekte yeni rental yetkisi değildir.
