"""Explicit, local GPU entrypoints. Does not rent GPUs, spend money or upload artifacts."""
import argparse
from collections import defaultdict
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import time

import torch

from .common import digest, rows, write_json
from .precision import MasterAccumulator, sample, alignment
from .interference import parameter_group, task_matrix
from .evaluation import score_case, summarize


def load(path, seed):
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA GPU required; CPU mocks are not an experiment')
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                attn_implementation='sdpa', local_files_only=True).to('cuda')
    model.config.use_cache = False
    model.config.num_logits_to_keep = 0
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    if bool(model.config.tie_word_embeddings) != tied:
        raise ValueError('Runtime tied-weight storage inconsistent with configuration')
    return model, tok


def batch_loss(model, row):
    batch = {k: torch.tensor([row[k]], device='cuda') for k in ['input_ids', 'labels']}
    batch['attention_mask'] = torch.ones_like(batch['input_ids'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    return loss


def precision_experiment(args):
    import bitsandbytes as bnb
    data = list(rows(args.data))
    rng = random.Random(args.seed)
    rng.shuffle(data)
    needed = args.steps * args.accumulation
    if len(data) < needed:
        raise ValueError('Insufficient unique examples; implicit repetition refused')
    data = data[:needed]
    reports = {}
    for mode in ['existing', 'fp32_micro']:
        model, tok = load(args.model, args.seed)
        model.train()
        accumulator = MasterAccumulator(model, mode)
        parameters = [p for _, p in accumulator.masters]
        optimizer = bnb.optim.AdamW8bit(parameters, lr=args.lr, weight_decay=.01)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        history = []
        for step in range(args.steps):
            accumulator.clear()
            current = data[step*args.accumulation:(step+1)*args.accumulation]
            weights = [sum(y != -100 for y in r['labels'][1:])*r.get('objective_weight', 1.) for r in current]
            denominator = sum(weights)
            if denominator <= 0:
                raise ValueError('Empty training target')
            before_master = {n: sample(p) for n, p in accumulator.masters}
            before_working = {n: sample(p) for n, p in accumulator.working}
            total_loss = 0.
            for row, weight in zip(current, weights):
                loss = batch_loss(model, row)
                (loss*(weight/denominator)).backward()
                total_loss += float(loss.detach())*weight/denominator
                accumulator.after_micro()
            accumulator.finish()
            gradient_samples = {n: sample(p.grad).tolist() for n, p in accumulator.masters if p.grad is not None}
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            accumulator.sync()
            updates = {}
            for (n, master), (_, working) in zip(accumulator.masters, accumulator.working):
                delta_master = sample(master)-before_master[n]
                delta_working = sample(working)-before_working[n]
                active = delta_master != 0
                updates[n] = {'master_delta_l2': float(delta_master.norm()),
                              'negative_gradient_to_update_cosine': alignment(-torch.tensor(gradient_samples[n]),delta_master)['cosine'] if n in gradient_samples else None,
                              'master_zero_fraction': float((delta_master == 0).float().mean()),
                              'working_zero_fraction': float((delta_working == 0).float().mean()),
                              'master_changed_but_working_unchanged_fraction': float((active & (delta_working == 0)).float().mean()),
                              'sample_count': len(delta_master)}
            history.append({'step': step+1, 'loss': total_loss, 'pre_clip_norm': float(norm),
                            'sampled_gradients': gradient_samples, 'sampled_updates': updates})
        torch.cuda.synchronize()
        reports[mode] = {'steps': history, 'seconds': time.monotonic()-started,
                         'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
                         'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3}
        reports[mode]['input_tokens_per_second'] = sum(len(r['input_ids']) for r in data)/reports[mode]['seconds']
        reports[mode]['assistant_tokens_per_second'] = sum(sum(v!=-100 for v in r['labels'][1:]) for r in data)/reports[mode]['seconds']
        checkpoint_root = getattr(args, 'checkpoint_root', None)
        if checkpoint_root:
            directory = Path(checkpoint_root)/('precision_'+mode)
            if directory.exists():
                raise ValueError('Refusing to overwrite diagnostic checkpoint')
            model.save_pretrained(directory, safe_serialization=True, max_shard_size='2GB')
            tok.save_pretrained(directory)
            write_json(directory/'diagnostic_provenance.json', {
                'source_model_files': model_receipt(args.model), 'mode':mode,
                'steps':args.steps, 'data_sha256':digest(args.data), 'seed':args.seed,
                'lr':args.lr,'accumulation':args.accumulation,
                'optimizer_saved':False,'fp32_masters_saved':False,
                'exact_optimizer_resume_supported':False,
                'scope':'Four-step precision diagnostic; not a production SFT candidate'})
            callback = getattr(args,'checkpoint_callback',None)
            if callback:
                reports[mode]['checkpoint_receipt'] = callback(directory)
            write_json(args.output, {'complete':False,'arms':reports})
        del optimizer, parameters, accumulator, model, loss
        gc.collect()
        torch.cuda.empty_cache()
        write_json(args.output, {'complete': False, 'arms': reports})
    comparisons = []
    for a, b in zip(reports['existing']['steps'], reports['fp32_micro']['steps']):
        comparisons.append({n: alignment(torch.tensor(v), torch.tensor(b['sampled_gradients'][n]))
                            for n, v in a['sampled_gradients'].items()})
    write_json(args.output, {'complete': True, 'model_files': model_receipt(args.model),
               'data_sha256': digest(args.data), 'seed': args.seed, 'lr': args.lr,
               'accumulation': args.accumulation, 'optimizer': 'AdamW8bit',
               'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'bitsandbytes']},
               'arms': reports, 'gradient_comparison': comparisons,
               'limitation': 'Only first update isolates accumulation at identical weights; later updates are diverging trajectories. Micro gradients remain BF16. Samples are deterministic, not full-gradient cosine.'})


def model_receipt(path):
    return {p.name: digest(p) for p in sorted(Path(path).glob('*')) if p.suffix in ['.safetensors', '.json', '.jinja']}


def interference_experiment(args):
    model, _ = load(args.model, args.seed)
    model.eval()  # Fixed checkpoint: disable dropout; autograd remains enabled.
    data = list(rows(args.data))
    all_families = ['math', 'reasoning', 'instruction', 'reading', 'dialogue', 'code', 'english_retention']
    families = args.families or all_families
    if not families or any(k not in all_families for k in families):
        raise ValueError('Unsupported diagnostic families')
    grouped = {k: [r for r in data if r['family'] == k] for k in families}
    if any(len(v) < args.repeats for v in grouped.values()):
        raise ValueError('Need representative, source-disjoint batches for every task family')
    groups = [r.get('group') for r in data]
    if None in groups or len(set(groups)) != len(groups):
        raise ValueError('Require explicit unique source groups for diagnostic micro-batches')
    rng = random.Random(args.seed)
    for values in grouped.values():
        rng.shuffle(values)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    matrices, norms, update_alignments = [], [], []
    for repeat in range(args.repeats):
        samples, full_norms = {}, {}
        for family in families:
            model.zero_grad(set_to_none=True)
            loss = batch_loss(model, grouped[family][repeat])
            loss.backward()
            by_group = defaultdict(list)
            norm_group = defaultdict(float)
            for name, p in model.named_parameters():
                if p.grad is not None:
                    group = parameter_group(name)
                    by_group[group].append(sample(p.grad, args.sample_size))
                    norm_group[group] += float(torch.linalg.vector_norm(p.grad.float()))**2
            samples[family] = {k: torch.cat(v) for k, v in by_group.items()}
            full_norms[family] = {k: v**.5 for k, v in norm_group.items()}
        matrices.append(task_matrix(samples))
        alignments = {}
        for family, groups in samples.items():
            alignments[family] = {}
            for key, vector in groups.items():
                average = torch.stack([sample_groups[key] for sample_groups in samples.values() if key in sample_groups]).mean(0)
                alignments[family][key] = alignment(-vector,-average)
        update_alignments.append(alignments)
        norms.append(full_norms)
    torch.cuda.synchronize()
    write_json(args.output, {'complete': True, 'matrices': matrices, 'full_gradient_norms': norms,
               'data_sha256': digest(args.data), 'model_files': model_receipt(args.model),
               'sample_size_per_tensor': args.sample_size, 'seconds': time.monotonic()-started,
               'peak_gib': torch.cuda.max_memory_allocated()/1024**3,
               'scope': 'Sampled coordinate cosines by group, fixed checkpoint, repeated independent tasks. Not proof of long-term regression causality; not an unbiased full-gradient cosine estimate.',
               'included_families': families, 'missing_families': sorted(set(all_families)-set(families)),
               'alignment_to_negative_mean_gradient':update_alignments,
               'alignment_scope':'Sampled SGD-direction proxy, not AdamW update direction',
               'pcgrad_enabled': False})


def evaluate(args):
    from .merge import check_local, interpolate
    from safetensors import safe_open
    model, tok = load(args.model, args.seed)
    receipt = model_receipt(args.model)
    if args.parent_b:
        _, sig, _ = check_local(args.model, args.parent_b)
        named = dict(model.named_parameters())
        if set(named) != set(sig):
            raise ValueError('Serialized and unique runtime parameter keys differ; tied/alias mapping requires review')
        with torch.no_grad():
            for name, p in named.items():
                with safe_open(sig[name][0], framework='pt', device='cpu') as f:
                    other = f.get_tensor(name).to(p.device)
                    p.copy_(interpolate(p, other, args.alpha))
        receipt = {'parent_a': receipt, 'parent_b': model_receipt(args.parent_b), 'alpha': args.alpha}
    manifest = json.loads(Path(args.manifest).read_text())
    if args.split == 'final':
        if not args.selection:
            raise ValueError('Final test requires a frozen development selection receipt')
        selection = json.loads(Path(args.selection).read_text())
        if selection.get('model_receipt') != receipt or selection.get('manifest_sha256') != digest(args.manifest):
            raise ValueError('Selected model/manifest mismatch')
        if not selection.get('development_complete') or not manifest.get('final_ready'):
            raise ValueError('Final test/data quality gates are not ready')
    data_path = Path(args.manifest).parent/manifest['splits'][args.split]['file']
    if digest(data_path) != manifest['splits'][args.split]['sha256']:
        raise ValueError('Dataset hash mismatch')
    model.eval()
    model.config.num_logits_to_keep = 1
    tok.padding_side = 'left'
    records = []
    started = time.monotonic()
    eos_ids = sorted({tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')})
    if None in eos_ids or any(v < 0 for v in eos_ids):
        raise ValueError('Invalid terminal token IDs')
    cases = list(rows(data_path))
    if args.batch_size < 1:
        raise ValueError('Invalid generation batch size')
    for offset in range(0, len(cases), args.batch_size):
        group = cases[offset:offset+args.batch_size]
        budgets = {c.get('max_new_tokens', 768) for c in group}
        if len(budgets) != 1:
            raise ValueError('Group together only cases with identical output budgets')
        budget = budgets.pop()
        prompts = [tok.apply_chat_template(c['messages'], tokenize=False, add_generation_prompt=True) for c in group]
        encoded = tok(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model.generate(**encoded, max_new_tokens=budget,
                                 do_sample=False, use_cache=True, eos_token_id=eos_ids,
                                 pad_token_id=tok.pad_token_id)
        for case, values in zip(group, out[:, encoded['input_ids'].shape[1]:].tolist()):
            ends = [i for i, value in enumerate(values) if value in eos_ids]
            ids = values[:ends[0]+1] if ends else values
            terminal = bool(ends)
            response = tok.decode(ids, skip_special_tokens=True).strip()
            records.append({'id': case['id'], 'group': case['group'], 'family': case['family'],
                            'language': case['language'], 'response': response,
                            'token_ids': ids, 'output_tokens': len(ids),
                            'truncated': not terminal and len(ids) >= budget,
                            'eos_terminated': terminal, **score_case(case, response)})
            if case['scorer'] == 'code' and args.code_image:
                from .code_sandbox import run_tests
                result = run_tests(response, case['tests'], args.code_image, case.get('test_imports', []))
                records[-1]['code_execution'] = result
                records[-1]['correct'] = result.get('passed') if result['scored'] else None
        write_json(args.output, {'complete': False, 'model_receipt': receipt, 'records': records})
    write_json(args.output, {'complete': True, 'model_receipt': receipt,
               'manifest_sha256': digest(args.manifest), 'split': args.split, 'records': records,
               'summary': summarize(records), 'seconds': time.monotonic()-started,
               'batch_size': args.batch_size, 'eos_ids': eos_ids,
               'not_measured': ['premature_semantic_termination', 'reasoning_validity', 'coding_execution', 'per_family_nll', 'TR_EN_language_nll'],
               'production_model_replaced': False})


def nll(args):
    model, _ = load(args.model, args.seed)
    model.eval()
    totals = defaultdict(lambda: {'loss_sum': 0., 'tokens': 0, 'rows': 0})
    with torch.inference_mode():
        for row in rows(args.data):
            n = sum(v != -100 for v in row['labels'][1:])
            if not n:
                raise ValueError('Empty shifted labels')
            loss = batch_loss(model, row)
            key = row.get('family', row.get('language', 'unknown'))
            totals[key]['loss_sum'] += float(loss)*n
            totals[key]['tokens'] += n
            totals[key]['rows'] += 1
    write_json(args.output, {'complete': True, 'data_sha256': digest(args.data),
               'scores': {k: {**v, 'nll': v['loss_sum']/v['tokens']} for k, v in totals.items()},
               'model_files': model_receipt(args.model)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment', choices=['precision', 'interference', 'evaluate', 'nll'])
    p.add_argument('--model', required=True, help='Local hash-verified checkpoint, never a moving Hub branch')
    p.add_argument('--data')
    p.add_argument('--manifest')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--checkpoint-root', help='Optional new directory for BF16 diagnostic weight backups; optimizer resume not supported')
    p.add_argument('--accumulation', type=int, default=32)
    p.add_argument('--lr', type=float, default=7.5e-7)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--families', nargs='+', help='Explicit partial gradient diagnostic; missing families reported')
    p.add_argument('--sample-size', type=int, default=256)
    p.add_argument('--split', choices=['dev', 'final'], default='dev')
    p.add_argument('--selection')
    p.add_argument('--parent-b')
    p.add_argument('--code-image', help='Digest-pinned, already installed Python container; no automatic pull')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--alpha', type=float, default=.3)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: no experiment executed')
    {'precision': precision_experiment, 'interference': interference_experiment,
     'evaluate': evaluate, 'nll': nll}[args.experiment](args)


if __name__ == '__main__':
    main()
