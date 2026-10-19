# Yetenek korunumu araştırması

[V1–V5 denetimi, değerlendirme, merge ve precision araştırması](research/capability_recovery/README.md): CPU uygulaması ve sonuçları hazır; yeni GPU deneyi veya V6 eğitimi başlatılmadı. [Karar ve açık işler](research/capability_recovery/FINAL_RECOMMENDATION.md).

# Current continuation

[V5 continuation completed: 1,997/1,997 updates and final evaluation](sft_v5_continue/RESULTS_TR.md). Step 1,600 selected; instruction score improved, stopping acceptance targets remain unmet. Model saved, GPU deleted, V2 preserved.

# Jamba2 Turkish SFT after the 1B-token CPT

Current work: [Broad V2-based SFT v5 training protocol](sft_v5/PLAN_TR.md). Pilot stopped at 100/1,997 steps; full epoch was not run. [Results and gate review](sft_v5/RESULTS_TR.md). GPU deleted; V2 preserved.

Previous work: [SFT v4 grounded instruction data and FP32-master training](sft_v4/PLAN_TR.md). Attempts A and B completed and were rejected. V2 remains the reference. [Verified results and next data correction](sft_v4/RESULTS_TR.md).

Previous run: [SFT v3 instruction data, protocol and results](sft_v3/PLAN_TR.md)
(2026-10-17): completed, but rejected because instruction-following and stopping
acceptance gates failed. V2 is preserved. The sections below describe the original SFT run; v3 starts from
the completed v2 model and has its own preparation audit and step journal.

Follow-up: [instruction data diagnosis and proposed remedy](research/instruction_following_20261017/REPORT_TR.md).
Completed: [bounded precision/cache diagnostic and results](research/instruction_following_20261017/PILOT_TR.md) — FP32 master weights resolved small-update loss in the diagnostic; no production checkpoint replaced.

## Purpose

Improve Turkish instruction following, useful concise answers, conversation memory
and exact output formats while retaining the language adaptation gained in CPT.
The original GitHub repository was empty on 2026-10-13; this pipeline uses the
tested CPT Docker environment instead of assuming an earlier SFT implementation.

Base: private `serda-dev/Jamba2-3B-Turkish-CPT-1B-20261007`, immutable revision
`8b1a9c812791fa7930e64b9444b41519c023269e`, subfolder `checkpoints/final`.
All 3,039,823,232 parameters are trained in BF16 with AdamW 8-bit optimizer states.
The weights are not quantized. Tokenizer and ChatML template are preserved.

## Dataset selection

The pinned revisions and quotas are in `config.json`.

| Dataset | Selected target | Purpose | Declared license |
| --- | ---: | --- | --- |
| [Turkce-Atlas-Instruct](https://huggingface.co/datasets/AlicanKiraz0/Turkce-Atlas-Instruct) | 24,000 | Turkish instructions, transformation and formats | MIT |
| [merve/turkish_instructions](https://huggingface.co/datasets/merve/turkish_instructions) | 12,000 | Short instruction-response tasks | Apache-2.0 |
| [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | up to 4,000 Turkish rows | Human-written/edited Turkish instructions | Apache-2.0 |
| [Uunan/turkish-knowledge-sft](https://huggingface.co/datasets/Uunan/turkish-knowledge-sft) | 8,000 | Broad explanations; synthetic, factual correctness not guaranteed | Apache-2.0 |
| [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 8,000 | English and multi-turn retention | MIT |
| Locally generated exact-target cases | up to 8,000 | JSON, arithmetic, extraction and conversation recall | Project-generated |

Sources are sampled with seeded buffered shuffling. Entire conversations over
2,048 tokens are excluded rather than truncated. This first run therefore does
not claim to train long-context abilities. Empty answers, malformed roles, special
token injection, obvious repeated lines and generation boilerplate are rejected.
Prompt hashes remove exact duplicates across sources and assign a separate 1%
validation split. Semantic paraphrase leakage is not comprehensively eliminated.
`manifest.json` records actual counts, input tokens, source revisions and hashes.
`sample-review.json` provides a reproducible spot check. Automated filters do not
establish factual accuracy for every synthetic answer.

Long-response-heavy Turkish-SFT-v1.0 and the large mixed-provenance reasoning/DPO
collection were researched but not selected for this first run. No benchmark
test split is used for training. Eighty independently generated format cases use
disjoint numeric ranges and are excluded from training and validation.

## Training and selection

- Assistant-only loss; system/user/padding labels are masked.
- Full weights, 2 epochs maximum, LR 1e-5, cosine schedule, 3% warmup.
- Single A100 SXM 80GB, microbatch 4, accumulation 8: global batch 32.
- Length grouping and dynamic padding; separate conversations are not packed.
- Validation and complete checkpoint every 300 optimizer steps.
- Best checkpoint chosen by assistant validation loss, then exported.
- Exact-format diagnostics before/after SFT; original frozen 2M-token TR/EN
  language evaluation and twelve CPT diagnostics after SFT.
- Acceptance requires assistant validation loss improvement, no decline in the
  80-case exact-format score, Turkish NLL within +5% of CPT and English within
  +10%. These are diagnostics, not comprehensive capability evaluations.

## Cost and operations

Credit at planning: approximately $41.80. Only offers <=$1.25/hour qualify.
Prepare and check data on the VPS before rental. Expected training is several
hours; an actual first-step measurement replaces this estimate after launch.
Request a graceful stop after 20 hours or credit <=$12; stop GPU at <=$7 if export
cannot be completed. No automatic additional rentals or account top-ups.

Use the known working public CPT image pinned by digest in `supervisor.py`, with
only `accelerate==1.10.1` added. GPU compiler and actual Mamba kernel are checked.
Model/optimizer checkpoints are uploaded to a separate private HF repository and
verified by size/hash. GPU logs, evaluation reports and configs are backed up to
the VPS and private Hub every ten minutes. Credentials are excluded.
The instance is deleted only after final model and log archives verify. Failures
stop the GPU and retain disk for recovery; they do not count as accepted results.

Local files: `/home/serda/sft-run-20261013`. The user systemd service is
`linguai-sft-supervisor.service`. `prepare.py` produces the immutable dataset;
`supervisor.py` rents, launches and monitors; `train.py` executes SFT on the GPU.

## Completed training and benchmarks — 2026-10-14

The final full-weight SFT model is published at
[serda-dev/Jamba2-3B-Turkish](https://huggingface.co/serda-dev/Jamba2-3B-Turkish).
It contains 3,039,823,232 BF16 parameters, not a LoRA adapter. This run used
63,083 training conversations and 680 validation conversations, two epochs and
3,944 optimizer steps after the 1B-token CPT (80% Turkish / 20% English).
The published final export uses checkpoint-3900, selected by validation loss.

All values below are accuracy percentages under the same local evaluation protocol.

| Model | Turkish MMLU (900) | Belebele TR (900) | GSM8K-TR short (200) | GSM8K-TR 8-shot (200) | IFEval-TR subset strict (120) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Jamba Turkish SFT 3B (published)** | **25.9** | **49.9** | **22.5** | **12.5** | **31.7** |
| Qwen2.5-3B-Instruct | 40.8 | 67.0 | 7.5 | 42.0 | 41.7 |
| Qwen3-4B-Instruct-2507 | 54.7 | 78.0 | 69.0 | 70.0 | 75.8 |
| Jamba Turkish CPT 3B | 19.8 | 28.8 | 9.5 | 3.0 | 13.3 |
| Original AI21-Jamba2-3B | 22.8 | 49.0 | 19.0 | 20.0 | 46.7 |

SFT recovers much of the capability lost during CPT. Against the original
already post-trained Jamba, the published model remains weaker on strict
instruction following (46.7 → 31.7) and 8-shot math (20.0 → 12.5).
Small gains on knowledge/reading do not establish an overall improvement.
Qwen3 is larger (4.02B versus 3.04B); Qwen2.5 is approximately the same size.
Both math prompt conditions are retained because results are strongly prompt-sensitive.

On the same RTX 4090, BF16 Transformers SDPA with working Mamba kernels,
512 input / 128 forced output tokens, warm-up excluded, median of three runs:

| Model | Decode tokens/s (batch 1) | First token ms | Peak allocated VRAM GiB | Aggregate tokens/s (batch 4) |
| --- | ---: | ---: | ---: | ---: |
| Jamba Turkish SFT | 48.6 | 30.6 | 5.73 | 190.1 |
| Qwen2.5-3B-Instruct | 51.0 | 30.2 | 5.81 | 199.2 |
| Qwen3-4B-Instruct-2507 | 42.7 | 35.0 | 7.61 | 166.8 |
| Jamba Turkish CPT | 48.6 | 30.3 | 5.73 | 189.3 |
| Original Jamba2 | 48.8 | 30.5 | 5.71 | 189.7 |

These are local measurements, not official leaderboard scores or maximum serving
throughput. MMLU uses the available 900-question subset with 0-shot letter logits;
IFEval uses 120 Turkish-compatible deterministic cases, not the full suite.
GSM8K-TR uses the same seeded 200 questions under short and translated 8-shot prompts.
Generation budgets are 512 / 1,024 / 2,048 tokens for short math / 8-shot math / IFEval.
Original/CPT generative results were rerun after correcting EOS 2 versus chat EOS 519;
the published SFT checkpoint already uses EOS 519. The first incorrect baseline
report is superseded. Exact prompt overlap with SFT data was zero; semantic and
pretraining contamination remain unknown. Long-context quality was not evaluated.
Speed excludes network, tokenization and compilation; tokenizer differences affect
how much text one token represents.

See the [full Turkish report](results/BENCHMARK_20261014.md),
[machine-readable results and 95% confidence intervals](results/BENCHMARK_20261014.json),
[CSV](results/BENCHMARK_20261014.csv) and [reproduction protocol](benchmarks/README.md).
Benchmark artifacts and logs are backed up; all benchmark GPU instances were deleted.

## Targeted SFT v2 — 2026-10-15

A targeted second run has launched from the measured weaknesses above.
It compares short pilots from the existing SFT and CPT checkpoints, then trains
a broader mixture for instruction constraints, multi-step math, reading and
TR/EN retention. The frozen benchmark questions are excluded from training.
See the [v2 data, selection and cost protocol](sft_v2/README.md).
Final v2 results are pending; the published model remains the verified v1.

V2 launch: instance `56099238`, A100 SXM 80GB, ~$1.2635/hour.
Prepared data: 108,483 training / 870 validation conversations, approximately
60M input tokens per epoch. CUDA/Mamba decode preflight passed. The paired
starting-point pilot runs first; final v2 metrics are not available yet.
