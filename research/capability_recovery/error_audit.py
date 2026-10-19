"""Post-hoc content/format diagnostics; never replace frozen selection gates.

Unicode token-overlap F1 is inspired by extractive-QA evaluation. It is a lexical
proxy, NOT an entailment verifier or a general conversational quality score.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

from .common import digest, rows, write_json
from .evaluation import normalize


def tokens(text):
    return re.findall(r'\w+', normalize(text), flags=re.UNICODE)


def overlap_f1(response, reference):
    a, b = tokens(response), tokens(reference)
    overlap = sum((Counter(a) & Counter(b)).values())
    return 2*overlap/(len(a)+len(b)) if a or b else 1.


def punctuation_em(response, reference):
    def strip(text):
        return normalize(''.join(c if not unicodedata.category(c).startswith('P') else ' ' for c in text))
    return strip(response) == strip(reference)


def memory_fact_proxy(response, case):
    """Conservative lexical recall proxy, separately from requested output form."""
    response = normalize(response)
    if re.search(r'\b(değil|bilmiyorum|hatırlamıyorum|söylemediniz|belirtmediniz)\b', response):
        return False
    answer = case['answers'][0]
    if answer.isdigit():
        numbers = set(re.findall(r'\b\d+\b', response))
        return numbers == {answer}  # Never ignore another contradictory number.
    needle, haystack = tokens(answer), tokens(response)
    return any(haystack[i:i+len(needle)] == needle for i in range(len(haystack)-len(needle)+1))


def audit(root, manifest):
    spec = json.loads(manifest.read_text())['splits']['dev']
    file = manifest.parent/spec['file']
    if digest(file) != spec['sha256']:
        raise ValueError('Frozen data checksum mismatch')
    cases = {c['id']: c for c in rows(file)}
    result = {'post_hoc': True, 'used_for_selection': False, 'manifest_sha256': digest(manifest),
              'scope': 'Punctuation equivalence and lexical content proxies; not independently verified semantic correctness',
              'reference': 'https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/squad/compute_score.py',
              'deviation': 'Unicode Turkish tokens; no English article removal. Frozen strict scores remain unchanged.', 'models': {}}
    for name in ['v2','v5','alpha_0.15','alpha_0.30','alpha_0.50','alpha_0.70']:
        data = json.loads((root/(name+'.json')).read_text())
        if not data['complete'] or data['manifest_sha256'] != digest(manifest):
            raise ValueError('Incomplete or incompatible generation')
        families = {}
        for family in ['reading', 'dialogue']:
            selected = [r for r in data['records'] if r['family'] == family]
            punct, f1, fact = 0, 0., 0
            for r in selected:
                c = cases[r['id']]
                punct += any(punctuation_em(r['response'], ref) for ref in c['answers'])
                f1 += max(overlap_f1(r['response'], ref) for ref in c['answers'])
                if family == 'dialogue':
                    fact += memory_fact_proxy(r['response'], c)
            families[family] = {'n': len(selected), 'frozen_strict_correct': sum(r['correct'] is True for r in selected),
                'punctuation_normalized_em': punct, 'mean_unicode_token_f1': f1/len(selected),
                'literal_memory_fact_proxy': fact if family == 'dialogue' else None}
        math = [r for r in data['records'] if r['family'] == 'math']
        families['math_format'] = {'n': len(math), 'declared_final_answer_marker_present': sum(
            bool(re.search(r'####\s*-?\d+(?:[.,]\d+)?\s*$', r['response'])) for r in math)}
        result['models'][name] = families
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    write_json(a.output, audit(a.results_dir, a.manifest))
