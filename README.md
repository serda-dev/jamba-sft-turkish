# Jamba2 Turkish SFT after the 1B-token CPT

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
