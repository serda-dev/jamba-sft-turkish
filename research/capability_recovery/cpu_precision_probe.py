"""Controlled CPU numerical demonstration; explicitly not Jamba/GPU evidence."""
import torch
from .common import write_json
from .precision import MasterAccumulator


def main():
    arms = {}
    grads = [1.] + [.001]*31
    for mode in ['existing','fp32_micro']:
        model = torch.nn.Linear(1,1,bias=False).bfloat16()
        with torch.no_grad():
            model.weight.fill_(.1)
        acc = MasterAccumulator(model,mode)
        acc.clear()
        for g in grads:
            (model.weight*g).sum().backward()
            acc.after_micro()
        acc.finish()
        arms[mode] = {'gradient':acc.masters[0][1].grad.item(),
                      'working_gradient_dtype':'BF16', 'master_dtype':'FP32'}
    reference=sum(torch.tensor(g,dtype=torch.bfloat16).item() for g in grads)
    write_json('research/capability_recovery/results/cpu_precision_probe.json',
               {'device':'CPU','model':'one scalar BF16 linear parameter',
                'micro_batches':len(grads),'arms':arms,'sum_of_bf16_micro_gradients_in_fp64':reference,
                'actual_jamba_gpu_experiment':False,
                'conclusion':'Demonstrates cross-micro BF16 accumulation error for a controlled gradient sequence; does not measure its size or capability effect in Jamba.'})


if __name__=='__main__':
    main()
