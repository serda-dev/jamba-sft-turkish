"""Deficit scheduling by actual assistant tokens, without silently unlimited reuse."""
from collections import Counter, defaultdict
import random


TARGETS = {'general': .25, 'math': .20, 'instruction': .20,
           'reading': .15, 'code': .10, 'grounded_qa': .10}


def balanced_schedule(metadata, token_budget, shares=None, max_reuse=2, seed=20261019):
    shares = shares or TARGETS
    if abs(sum(shares.values())-1) > 1e-9 or any(v <= 0 for v in shares.values()):
        raise ValueError('Shares must be positive and sum to 1')
    if max_reuse < 1 or token_budget < 1:
        raise ValueError('Invalid exposure limits')
    pools = defaultdict(list)
    for i, row in enumerate(metadata):
        if row['family'] not in shares or row['target_tokens'] <= 0:
            raise ValueError('Unclassified or empty target')
        pools[row['family']].append(i)
    rng = random.Random(seed)
    for family in shares:
        rng.shuffle(pools[family])
    queues = {k: pools[k] * max_reuse for k in shares}
    cursor = Counter()
    tokens = Counter()
    exposure = Counter()
    selected = []
    # A deficient family halts the schedule: do not silently redirect its quota.
    while sum(tokens.values()) < token_budget:
        family = min(shares, key=lambda k: (tokens[k] / shares[k], k))
        if cursor[family] >= len(queues[family]):
            break
        index = queues[family][cursor[family]]
        cursor[family] += 1
        n = metadata[index]['target_tokens']
        selected.append(index)
        tokens[family] += n
        exposure[index] += 1
    total = sum(tokens.values())
    report = {'requested_tokens': token_budget, 'sampled_tokens': total,
              'complete': total >= token_budget, 'tokens_by_family': dict(tokens),
              'actual_shares': {k: tokens[k]/max(total, 1) for k in shares},
              'unique_by_family': {k: sum(metadata[i]['family'] == k for i in exposure) for k in shares},
              'max_reuse_observed': max(exposure.values(), default=0),
              'shortages': {k: max(0, round(token_budget*v)-sum(metadata[i]['target_tokens'] for i in pools[k])*max_reuse)
                            for k, v in shares.items()},
              'seed': seed, 'max_reuse': max_reuse,
              'note': 'Token exposure schedule; not a claim of equal gradient contributions'}
    return selected, report


class ScheduledDataset:
    """Random-access uncompressed JSONL, safe for DataLoader worker processes."""
    def __init__(self, path, metadata):
        self.path = str(path)
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        import json
        with open(self.path, 'rb') as f:
            f.seek(self.metadata[index]['offset'])
            row = json.loads(f.readline())
        count = sum(v != -100 for v in row['labels'][1:])
        if count != self.metadata[index]['target_tokens']:
            raise ValueError('Dataset metadata/label token mismatch')
        return row


def collate(rows, pad_token_id):
    import torch
    width = max(len(r['input_ids']) for r in rows)
    return {
        'input_ids': torch.tensor([r['input_ids'] + [pad_token_id]*(width-len(r['input_ids'])) for r in rows]),
        'labels': torch.tensor([r['labels'] + [-100]*(width-len(r['labels'])) for r in rows]),
        'attention_mask': torch.tensor([[1]*len(r['input_ids']) + [0]*(width-len(r['input_ids'])) for r in rows]),
        'families': [r['family'] for r in rows],
        'target_tokens': torch.tensor([sum(v != -100 for v in r['labels'][1:]) for r in rows]),
        'sample_ids': [r['sample_id'] for r in rows],
    }


def training_loader(path, metadata, token_budget, pad_token_id, shares=None,
                    max_reuse=2, seed=20261019, micro_batch_size=1, num_workers=0):
    """Returns actual padded train batches plus reproducible exposure receipt.

    Pass ONLY input_ids/labels/attention_mask to the model. For an accumulated
    optimizer step, weight each micro loss by its shifted target-token count
    divided by the accumulated step's count. No objective_weight is applied.
    """
    from functools import partial
    from torch.utils.data import DataLoader
    schedule, report = balanced_schedule(metadata, token_budget, shares, max_reuse, seed)
    if not report['complete']:
        raise ValueError(f'Insufficient unique-family capacity: {report["shortages"]}')
    loader = DataLoader(ScheduledDataset(path, metadata), sampler=schedule,
                        batch_size=micro_batch_size, num_workers=num_workers,
                        collate_fn=partial(collate, pad_token_id=pad_token_id), drop_last=False)
    return loader, report


def optimizer_windows(loader, accumulation_batches=8):
    """Actual model kwargs and token weights, including a short final window.

    For each window: clear gradients once, call the model with each kwargs dict,
    backward(loss * weight), then use the existing audited optimizer/master path.
    Family metadata never accidentally reaches model.forward().
    """
    from itertools import islice
    if accumulation_batches < 1:
        raise ValueError('Positive accumulation size required')
    iterator = iter(loader)
    while True:
        window = list(islice(iterator, accumulation_batches))
        if not window:
            return
        counts = [int(batch['target_tokens'].sum()) for batch in window]
        denominator = sum(counts)
        if denominator <= 0:
            raise ValueError('Empty accumulated supervision')
        yield [({k: batch[k] for k in ['input_ids', 'labels', 'attention_mask']}, n/denominator)
               for batch, n in zip(window, counts)]
