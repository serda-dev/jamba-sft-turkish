"""Bounded gradient diagnostics and optional PCGrad mathematical reference."""
import random
import torch
from .precision import alignment


def parameter_group(name):
    if 'embed' in name:
        return 'embeddings'
    if 'lm_head' in name:
        return 'output_head'
    if 'mamba' in name or any(s in name for s in ['in_proj', 'dt_proj', 'x_proj', 'conv1d', 'A_log', 'D']):
        return 'mamba'
    if any(s in name for s in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn']):
        return 'attention'
    if any(s in name for s in ['feed_forward', 'mlp', 'gate_proj', 'up_proj', 'down_proj']):
        return 'feed_forward'
    return 'other'


def task_matrix(task_samples):
    report = {}
    for a, groups in task_samples.items():
        for b, other in task_samples.items():
            if a >= b:
                continue
            report[a + ':' + b] = {k: alignment(v, other[k]) for k, v in groups.items() if k in other}
    return report


def pcgrad(gradients, seed=20261019):
    """Optional CPU/small-vector reference; never used in production by default."""
    if not gradients or any(g.shape != gradients[0].shape for g in gradients):
        raise ValueError('Gradient vectors must have identical shapes')
    rng = random.Random(seed)
    projected = []
    for i, source in enumerate(gradients):
        current = source.float().clone()
        order = [j for j in range(len(gradients)) if j != i]
        rng.shuffle(order)
        for j in order:
            other = gradients[j].float()
            dot = current.dot(other)
            norm = other.dot(other)
            if dot < 0 and norm > 0:
                current -= dot / norm * other
        projected.append(current)
    return torch.stack(projected).mean(0)
