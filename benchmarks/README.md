# Turkish quality and native inference speed comparison

This evaluation runs five immutable BF16 checkpoints on one GPU with Transformers
4.56.1, Torch 2.5.1, SDPA and cached generation. No training, weight quantization,
external tools, API judge or thinking-mode model is used.

Models: Turkish Jamba SFT, Qwen2.5-3B-Instruct, Qwen3-4B-Instruct-2507,
Turkish Jamba CPT and original AI21 Jamba2-3B. Qwen3 has 4B parameters and is
larger than the approximately 3B alternatives. Gated Llama weights were unavailable.

## Common frozen tasks

- [AYueksel TurkishMMLU](https://huggingface.co/datasets/AYueksel/TurkishMMLU):
  all 900 questions in its public `All/test` subset, balanced across nine subjects.
- [Belebele](https://huggingface.co/datasets/facebook/belebele): all 900 Turkish
  `tur_Latn/test` reading-comprehension questions.
- [GSM8K Turkish](https://huggingface.co/datasets/malhajar/gsm8k-tr): seeded
  200-question sample from the 1,318-row `main/test` split. This is translated data.
- [IFEval-TR](https://huggingface.co/datasets/bezir/IFEval_tr): seeded 120-question
  subset containing only supported deterministic constraints. Missing rule
  arguments and unsupported English-specific checks are excluded. The upstream
  Google IFEval source is vendored with its Apache license and content hashes.
  Turkish case checks require Turkish language detection with a fixed seed.

Multiple-choice tasks use zero-shot native chat prompts and choice-letter logits.
This measures choice selection, rather than parsing generated prose. Chance
accuracy is 20% on five-choice TurkishMMLU and 25% on four-choice Belebele.
Generative tasks use greedy decoding, native templates, a batch of four and equal
output budgets: 512 tokens for math, 2,048 for instructions. Outputs hitting the
cap are reported. Math uses the final `####` number or last numeric answer, with
explicit Turkish/English decimal/thousands normalization. IFEval reports both
strict all-constraints-per-prompt and per-instruction accuracy.

These results are a comparison on a shared protocol. They are **not** a reproduced
5-shot TurkishMMLU paper leaderboard or the full IFEval-TR leaderboard metric.
IFEval measures verifiable constraints, not factual truth or answer helpfulness.
Pretraining benchmark contamination and semantic SFT overlap are not exhaustively
audited. Wilson 95% intervals are included; close rankings need caution.

## Speed

Batch-one tests use actual 128, 512 and 2,048 input tokens, 128 forced output
tokens, an excluded warmup and three measured repetitions. Reports include first
token latency, cached decoding tokens/s, end-to-end tokens/s, characters/s and
peak allocated VRAM. A separate batch-four test reports aggregate tokens/s.
Download and compilation time are excluded. Tokenizers differ, so tokens/s and
characters/s must both be considered. Results characterize this GPU/backend,
not optimized vLLM/TensorRT deployments or API providers.

The automatic supervisor checks every five minutes, retains raw responses,
configs, speed trials and logs on the VPS, and verifies Hub hashes before deleting
the GPU. Runtime/cost guards stop the job before consuming the whole team credit.
The output is stored in the private SFT repository under `benchmarks/20261014`.

Preparation: `prepare.py`; scoring: `scoring.py`; GPU execution: `run.py`;
operations: `supervisor.py`. VPS root: `/home/serda/benchmark-run-20261014`.
