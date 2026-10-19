"""Independent final assessment. Never used to select a runner-up checkpoint."""
import argparse
import json
from pathlib import Path

from .common import digest, write_json
from .selection import compare


def language_gates(reference, candidate, policy):
    same = (reference.get('data_sha256') == candidate.get('data_sha256')
            and reference.get('data_sha256') is not None
            and reference.get('complete') and candidate.get('complete'))
    result = {}
    for language, key in [('turkish','turkish_nll_max_ratio'),('english','english_nll_max_ratio')]:
        a = reference.get('scores',{}).get(language,{})
        b = candidate.get('scores',{}).get(language,{})
        result[language] = bool(same and a.get('tokens',0)>0 and a.get('tokens')==b.get('tokens')
                                and 'nll' in a and 'nll' in b and a['nll']>0
                                and 0<=b['nll']<=a['nll']*policy[key])
    return result


def assess(reference, candidate, selection, manifest, policy, language_reference, language_candidate):
    if reference.get('split') != 'final' or candidate.get('split') != 'final':
        raise ValueError('Final assessment requires final reports')
    if candidate.get('model_receipt') != selection.get('model_receipt'):
        raise ValueError('Final model differs from frozen development selection')
    if candidate.get('manifest_sha256') != selection.get('manifest_sha256'):
        raise ValueError('Final manifest differs from selection')
    if not manifest.get('final_ready') or not selection.get('development_complete'):
        raise ValueError('Unqualified final data or incomplete development selection')
    # Reuse numerical paired checks, preserving separate split guard at this entrypoint.
    config = {**policy['development'], **policy['final']}
    result = compare({**reference,'split':'dev'}, {**candidate,'split':'dev'}, config)
    languages = language_gates(language_reference,language_candidate,policy['development'])
    language_identity = (language_reference.get('model_files') == reference.get('model_receipt')
                         and language_candidate.get('model_files') == candidate.get('model_receipt')
                         and bool(candidate.get('model_receipt')) and bool(reference.get('model_receipt')))
    instruction = result['families']['instruction']['difference']
    n_a, n_b = len(reference['records']),len(candidate['records'])
    capped_a = sum(r['truncated'] for r in reference['records'])/max(n_a,1)
    capped_b = sum(r['truncated'] for r in candidate['records'])/max(n_b,1)
    complete_scoring = all(r['correct'] is not None for r in candidate['records'])
    gates = {'family_nonregression':result['eligible_for_selection'],
             'language_retention':all(languages.values()) and language_identity, 'all_semantic_scores_available':complete_scoring,
             'instruction_improvement':instruction is not None and instruction>=policy['final']['instruction_gain_target'],
             'truncation_reduction':capped_b<=capped_a*(1-policy['final']['truncation_relative_reduction_target'])}
    return {'accepted':all(gates.values()),'gates':gates,'paired':result,'languages':languages,
            'production_replacement_authorized':False,
            'historical_milestones_report_only':policy['historical_milestones'],
            'scope':'One development-frozen candidate; failure does not authorize testing runner-up models on final'}


def main():
    p=argparse.ArgumentParser()
    for field in ['reference','candidate','selection','manifest','policy','language-reference','language-candidate','output']:
        p.add_argument('--'+field,required=True)
    args=p.parse_args()
    load=lambda key:json.loads(Path(getattr(args,key)).read_text())
    result=assess(*[load(k) for k in ['reference','candidate','selection','manifest','policy','language_reference','language_candidate']])
    result['policy_sha256']=digest(args.policy)
    write_json(args.output,result)


if __name__=='__main__':
    main()
