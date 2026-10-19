"""Research-only FP32 accumulation. Working forward parameters stay BF16."""
import torch


class MasterAccumulator:
    def __init__(self, model, mode):
        if mode not in ('existing', 'fp32_micro'):
            raise ValueError(mode)
        self.mode = mode
        self.working = list(model.named_parameters())  # PyTorch deduplicates tied objects.
        if any(p.dtype != torch.bfloat16 for _, p in self.working):
            raise ValueError('All working parameters must be BF16')
        self.masters = [(n, torch.nn.Parameter(p.detach().float()))
                        for n, p in self.working]
        self.seen = set()

    def clear(self):
        self.seen.clear()
        for (_, p), (_, master) in zip(self.working, self.masters):
            p.grad = None
            master.grad = None

    @torch.no_grad()
    def after_micro(self):
        if self.mode == 'fp32_micro':
            # Clear BF16 leaf grads after each backward. Each micro gradient is still
            # BF16; the sum across micro-batches is FP32. Not full-FP32 differentiation.
            for name, p in self.working:
                if p.grad is not None:
                    self.seen.add(name)
            for (_, p), (_, master) in zip(self.working, self.masters):
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Nonfinite micro gradient')
                if master.grad is None:
                    master.grad = p.grad.detach().float()
                else:
                    master.grad.add_(p.grad.detach().float())
                p.grad = None

    @torch.no_grad()
    def finish(self):
        if self.mode == 'existing':
            for (name, p), (_, master) in zip(self.working, self.masters):
                master.grad = None if p.grad is None else p.grad.detach().float()
                if p.grad is not None:
                    self.seen.add(name)
        for _, p in self.masters:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError('Nonfinite accumulated gradient')

    @torch.no_grad()
    def sync(self):
        for (name, p), (master_name, master) in zip(self.working, self.masters):
            if name != master_name or p.shape != master.shape:
                raise ValueError('Master parameter mismatch')
            p.copy_(master)


def sample(tensor, count=1024):
    flat = tensor.detach().reshape(-1)
    n = min(count, flat.numel())
    idx = torch.arange(n, device=flat.device, dtype=torch.int64) * max(0, flat.numel()-1) // max(1, n-1)
    return flat[idx].float().cpu()


def alignment(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    denom = a.norm() * b.norm()
    return {'norm_a': float(a.norm()), 'norm_b': float(b.norm()),
            'cosine': float(a.dot(b)/denom) if denom > 0 else None,
            'relative_l2_difference': float((a-b).norm()/a.norm()) if a.norm() > 0 else None}
