# Targeted full-weight SFT v2 — 2026-10-15

## Why another SFT

The frozen 2026-10-14 comparison identifies concrete failures, not an overall IQ score:

- TurkishMMLU: 485/900 predictions are C and only 2/900 are E, versus 163 C and
  215 E gold answers. This is evidence of strong answer-position bias; it does
  not establish that every error is caused by that bias.
- Eight-shot GSM8K-TR: 47/200 outputs have >20% repeated 8-grams; all 200 start
  with `Soru:`. Inspection shows copied demonstrations and unrelated questions.
  Short math accuracy is 45/200; eight-shot accuracy is 25/200.
- IFEval-TR supported subset: 38/120 strict, 51/120 repetitive and 33/120 capped.
  Paragraph count: 0/8; end phrase: 2/13; placeholders: 4/12; lowercasing: 3/14.
- The SFT improved on the CPT checkpoint but did not retain the original
  model's broad instruction following. More repetitions of the previous four
  synthetic templates would not address that gap.

`prepare.py` followed by `finalize_data.py` must complete before rental.

`analyze.py` writes the complete per-rule counts and all three Jamba comparisons.
The raw original evaluation and its scope remain in `../benchmarks/README.md`.

## Data changes

Pinned source revisions, actual accepted counts, source licenses, assistant
and input token counts, and gzip hashes are recorded in the run manifest.

| Source | Maximum selection before final filtering | Purpose |
| --- | ---: | --- |
| Diverse v1 replay (Atlas / merve / Aya / UltraChat) | 26,500 | Retain general TR/EN assistance; excludes old synthetic four-template set |
| `malhajar/gsm8k-tr`, **train only** | 7,472 | Natural Turkish multi-step math with worked solutions |
| `openai/gsm8k`, **train only** | 6,000 | Preserve English arithmetic / worked-solution skill |
| `barandinho/turkish-reasoning-distilled-sft`, `webinstruct_verified` | 26,000 | Varied Turkish worked solutions |
| Same, `openthoughts_science_verified` | 6,000 | Science explanations and reasoning |
| Same, `acecode_verified` | 6,000 | Turkish programming instructions |
| `HuggingFaceTB/smoltalk`, `smol-constraints` train | 9,000 | Varied length, keyword and output constraints |
| Same, `smol-summarize` train | 3,500 | Reading and summarization |
| Procedural Turkish tasks, 28 modes | 40,000 generated, duplicates removed | Balanced 5-choice reading, multi-step math with a demonstration, formats, paragraphs, capitalization, sections, conversation recall |

The distilled source contributes complete user-facing `final_answer` solutions,
not private deliberation traces. `verified` is the publisher's label and is not
an independent factual guarantee. The source's IFEval subset is excluded entirely.
Cards: [distilled Turkish SFT](https://huggingface.co/datasets/barandinho/turkish-reasoning-distilled-sft),
[GSM8K-TR](https://huggingface.co/datasets/malhajar/gsm8k-tr),
[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).
The mixed-provenance research-only 3.4M collection and unlicensed instruction
collection were considered and excluded from this run.

Filters reject malformed roles, embedded chat tokens, empty answers, repeated
6-grams (>12%), duplicate normalized prompts and conversations over 3,072 tokens.
Complete answers are retained; no answer truncation or concatenation of unrelated
conversations. Labels cover only assistant replies and their end token.
Exact normalized hashes and contiguous 13-word matches exclude frozen benchmark
prompts and **the entire GSM8K-TR test split**, including unmeasured questions.
This does not rule out translation or semantic leakage. Prompt groups share one
train/validation assignment. The data audit moves all replay or previously
v1-trained exact prompts into training, so pilot development does not reward
v1 for memorized training examples. Development probes use fresh numbers and different
wording, but share procedural task families and are not a broad capability test.

## Pilot and training

1. Prepare all data on the VPS; there is no paid GPU during preparation.
2. On one A100 SXM 80GB, compare two immutable starting points: published v1 SFT
   `089da0e1c0f95932f7610e1d3fec0ad10e66c33c` and CPT final
   `8b1a9c812791fa7930e64b9444b41519c023269e`.
3. Apply the same 120-step full-weight pilot, same data, seed and 5e-6 LR to each.
   Select lower assistant loss on the same development sample, requiring loss
   improvement and no >5 percentage-point drop on procedural diagnostics.
   If neither qualifies, stop for review.
4. Continue from the selected pilot weights, with a fresh optimizer, on the full
   mixture for at most two epochs. The pilot sample is a training subset and
   will be revisited; pilot weights are not presented as an independent test.
5. BF16 full weights; AdamW 8-bit optimizer states; microbatch 2 × accumulation 16;
   cosine LR 5e-6, 4% warmup; 3,072-token limit. Validation/checkpoint every 400
   steps; select the best checkpoint by development loss. This does not imply
   that increasing epochs or sequence length automatically improves quality.
6. Export and hash-verify a private v2 checkpoint, then run the same frozen 2,320
   benchmark cases for final reporting. Test results do not select checkpoints.
   The published v1 model remains available while v2 is assessed.

After export, the 2M-token frozen TR/EN retention set is checked against v1
(TR NLL within +5%, EN within +10%). A candidate is marked accepted only with
complete final tests, no large regression (3 percentage-point tolerance), and
at least +5pp IFEval or +5pp on one math condition, as well as lower development
loss. These gates are practical thresholds, not statistical significance claims.
They classify the exported candidate after training and do not select weights.

The procedural probe checks prescribed fields exactly and open prose by
structural constraints. It is only a deterministic regression check and does
not judge factual truth or broad helpfulness. Final external-task improvements,
length-cap rates, repetition and answer-position distribution need review before
calling v2 better. Knowledge gains are not promised from SFT alone.

## Operations

VPS root: `/home/serda/sft-v2-run-20261015`.
Private output: `serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015`.
Unit: `linguai-sft-v2-supervisor.service`.

Current initial team credit: $34.50. Rental ceiling $1.35/hour, one A100.
Graceful stop at 18 hours or $12 remaining; hard stop at $7 remaining.
The supervisor checks every five minutes, copies logs to the VPS and verifies
Hub uploads. Final model and logs must verify before instance destruction.
Errors stop compute and retain the instance disk for recovery. Existing published
model and v1 training archives are preserved. No automatic account top-up.

## Prepared dataset audit

- Training: 108483 conversations; validation: 870.
- Input tokens per epoch: 59,921,811.
- Assistant target tokens per epoch: 30,628,366.
- All assistant masks and EOS labels audited. Zero exact overlap between splits
  or between new validation and v1 training. 258 replay validation assignments
  were moved into training before either pilot.
- Five focused target/masking checks passed before rental.


## Recovery on 2026-10-17

The first full run stopped at step 50 with an A100 80GB allocation failure.
The effective example batch remains 32, with microbatch 2 and accumulation 16.
Sequence length 3,072, two epochs, LR 5e-6, and full-weight training are preserved.

The successful paired pilots were retained on instance 56099238. The selected
SFT pilot was backed up and hash-verified on the private v2 Hub repository before
reuse. Full training restarts from those pilot weights with a fresh optimizer;
the failed first 50 full-training steps had no saved optimizer checkpoint.
No exact resume of those 50 steps is claimed.

`memory_preflight.py` performed two real AdamW8bit optimizer steps at padded
length 3,072, microbatch 2 and accumulation 16. Peak allocated VRAM was
50.30 GiB, peak reserved 52.25 GiB, on a 79.25 GiB device. Both losses were finite.
The scratch process discards mutated probe weights; training reloads the
authenticated pilot. `pilot_recovery.py` rejects changed dataset identities,
changed pilot recipes, changed effective batch, or altered preserved weights.

The existing credit floors, periodic verified checkpoint/log uploads, full
2,320-condition external evaluation, TR/EN retention checks and verified final
export before GPU deletion remain active. The published v1 model is preserved.
Recovery files and tested source snapshots are under `recovery/20261017` in
`serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015`.

### Final evaluation recovery, 2026-10-17

Full training completed all 6,782 steps and two epochs in 26,202 seconds.
Development assistant loss fell from 0.76895237 to 0.51659489. These are
development results, not an independent capability score. The final export
was hash verified at HF revision `c8ee3888c486fa49d8d9e798a632c9f7ce664d7a`.
The external scorer then failed before its first case because `absl` was
missing. The supervisor stopped the GPU and retained its disk.

The setup now includes `absl-py==2.3.1`. To resume this completed run without
training again, use `python -u finalize_only.py --baseline-loss 0.7689523696899414`
from its GPU run directory with the original project on `PYTHONPATH`.
This command checks completed steps/epochs, data identity and every exported
model file against the verified receipt before running the same finalization
and acceptance gates as normal training. Keep the original baseline value
from the training log. The normal supervisor must monitor this process using
its new `job.pid`; it verifies the final Hub export and logs before deletion.
