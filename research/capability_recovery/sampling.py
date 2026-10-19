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
