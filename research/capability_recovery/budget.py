"""Pure spending/lifetime limits, independently usable by a watchdog."""


def usage(state, credit, now, max_seconds=13500, max_cost=4.70):
    elapsed = max(0, now-state['started_at'])
    observed = max(0, state['initial_credit']-credit)
    conservative = max(observed, elapsed/3600*state['hourly_rate']+.30)
    return {'elapsed_seconds': elapsed, 'observed_spent': observed,
            'conservative_spent': conservative,
            'hard': elapsed >= max_seconds or conservative >= max_cost}
