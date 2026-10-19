"""Additional deterministic checks for NEW development data only.

The historical benchmark and its supported subset remain unchanged. These
scores measure the declared instruction contracts, never factual correctness.
"""
import re
import sys
from pathlib import Path


class TurkishCapitalWords:
    """Unicode word boundaries, no English Punkt model or runtime downloads."""
    def __init__(self, values):
        self.frequency = values.get('capital_frequency')
        self.relation = values.get('capital_relation')
        if not isinstance(self.frequency, int) or self.frequency < 0 or self.relation not in ['at least', 'less than']:
            raise ValueError('Invalid capital-word constraint')

    def check_following(self, text):
        count = sum(word.isupper() for word in re.findall(r'\b\w+\b', text))
        return count >= self.frequency if self.relation == 'at least' else count < self.frequency


def checks(row):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
    from scoring import checks as historical_checks
    from instruction_following_eval import instructions_registry
    extra = {'detectable_content:postscript', 'length_constraints:nth_paragraph_first_word',
             'combination:repeat_prompt', 'keywords:letter_frequency',
             'change_case:capital_word_frequency'}
    output = []
    for ident, values in zip(row['instruction_id_list'], row['kwargs']):
        if ident == 'change_case:capital_word_frequency':
            output.append(TurkishCapitalWords(values))
            continue
        if ident == 'keywords:letter_frequency':
            letter = values.get('letter')
            if not letter or not re.fullmatch('[a-zA-Z]', letter):
                raise ValueError('Upstream letter checker randomizes non-ASCII letters')
        if ident in extra:
            rule = instructions_registry.INSTRUCTION_DICT[ident](ident)
            keys = rule.get_instruction_args_keys() or []
            args = {k: values.get(k) for k in keys}
            if ident == 'combination:repeat_prompt':
                args['prompt_to_repeat'] = row['prompt']
            if any(v is None for v in args.values()):
                raise ValueError('Random upstream checker argument refused')
            rule.build_description(**args)
            output.append(rule)
        else:
            output.extend(historical_checks({'instruction_id_list': [ident], 'kwargs': [values]}))
    if not output or len(output) != len(row['kwargs']):
        raise ValueError('Empty or misaligned constraints')
    return output


def score(row, response):
    return [bool(response.strip()) and bool(rule.check_following(response)) for rule in checks(row)]


def output_budget(row):
    minimum = max([float(kw.get('num_words') or 0) for kw in row['kwargs']
                   if kw.get('relation') == 'at least'] or [0])
    # A Turkish word may take multiple tokens. Never force a requested 800-word
    # answer into the default 768-token cap and call that a model stopping bug.
    return 4096 if minimum > 512 else 2048 if minimum > 164 else 768
