import json

import pytest
import torch

from research.capability_recovery.sampling import training_loader, TARGETS
from research.capability_recovery.evaluation import score_case, summarize
from research.capability_recovery.prepare_training_pool import task_family, evidence
from research.capability_recovery.prepare_merge_eval import coverage


def fixture_rows(tmp_path, absent=None):
    p = tmp_path/'data.jsonl'
    meta = []
    with p.open('wb') as f:
        for family in TARGETS:
            if family == absent:
                continue
            for i in range(8):
                row = {'input_ids': [1, 4] + [4]*(i % 3) + [5, 519],
                       'labels': [-100, -100] + [-100]*(i % 3) + [5, 519],
                       'family': family, 'sample_id': f'{family}/{i}'}
                meta.append({'offset': f.tell(), 'family': family, 'target_tokens': 2})
                f.write((json.dumps(row)+'\n').encode())
    return p, meta


def test_real_training_loader_reaches_budget_and_matches_shifted_labels(tmp_path):
    path, meta = fixture_rows(tmp_path)
    loader, report = training_loader(path, meta, 40, 0, micro_batch_size=3)
    observed = 0
    padded = 0
    order = []
    for batch in loader:
        assert torch.equal(batch['labels'][batch['attention_mask'] == 0],
                           torch.full_like(batch['labels'][batch['attention_mask'] == 0], -100))
        padded += int((batch['attention_mask'] == 0).sum())
        observed += int((batch['labels'][:, 1:] != -100).sum())
        assert int((batch['labels'][:, 1:] != -100).sum()) == int(batch['target_tokens'].sum())
        order.extend(batch['sample_ids'])
    assert observed == report['sampled_tokens'] >= 40
    assert padded > 0
    again, _ = training_loader(path, meta, 40, 0, micro_batch_size=3)
    assert order == [i for b in again for i in b['sample_ids']]


def test_training_loader_refuses_missing_family_before_partial_training(tmp_path):
    path, meta = fixture_rows(tmp_path, 'grounded_qa')
    with pytest.raises(ValueError, match='Insufficient'):
        training_loader(path, meta, 40, 0)


def test_loader_rechecks_actual_labels_against_metadata(tmp_path):
    path, meta = fixture_rows(tmp_path)
    meta[0]['target_tokens'] = 100
    from research.capability_recovery.sampling import ScheduledDataset
    with pytest.raises(ValueError, match='mismatch'):
        ScheduledDataset(path, meta)[0]


def test_optimizer_windows_accept_real_model_kwargs_and_keep_partial_tail(tmp_path):
    from research.capability_recovery.sampling import optimizer_windows
    path, meta = fixture_rows(tmp_path)
    loader, _ = training_loader(path, meta, 40, 0, micro_batch_size=3)
    windows = list(optimizer_windows(loader, accumulation_batches=4))
    assert len(windows[-1]) < 4
    for window in windows:
        assert sum(weight for _, weight in window) == pytest.approx(1.)
        for kwargs, _ in window:
            assert set(kwargs) == {'input_ids', 'labels', 'attention_mask'}
            assert kwargs['input_ids'].shape == kwargs['labels'].shape


def test_classification_does_not_turn_distilled_reasoning_or_editing_into_qa():
    assert task_family({'family': 'reasoning'}) is None
    assert task_family({'family': 'grounded_instruction'}) == 'instruction'
    assert task_family({'family': 'grounded_qa'}) == 'grounded_qa'
    assert evidence('İlk cümle. Ankara başkenttir. Son cümle.', 'Ankara', 11) == 'Ankara başkenttir.'


def test_instruction_score_is_explicitly_contract_only_and_not_semantic():
    c = {'scorer': 'ifeval_contract', 'rules': {'prompt': 'Virgül kullanma.',
         'instruction_id_list': ['punctuation:no_comma'], 'kwargs': [{}]},
         'semantic_scope': 'contracts only'}
    assert score_case(c, 'Yanlış bir iddia')['correct'] is True
    assert score_case(c, 'Doğru, ama virgüllü')['correct'] is False
    assert score_case(c, '')['correct'] is False
    assert score_case(c, 'Yanlış bir iddia')['semantic_scope'] == 'contracts only'


def test_turkish_capital_word_check_needs_no_english_punkt_download():
    from research.capability_recovery.instruction_scoring import score
    rule = {'instruction_id_list': ['change_case:capital_word_frequency'],
            'kwargs': [{'capital_frequency': 2, 'capital_relation': 'at least'}]}
    assert score(rule, 'ANKARA ve İZMİR.') == [True]
    assert score(rule, 'Ankara ve İzmir.') == [False]


def test_long_instruction_is_not_forced_into_a_short_output_cap():
    from research.capability_recovery.instruction_scoring import output_budget
    assert output_budget({'kwargs': [{'num_words': 800, 'relation': 'at least'}]}) == 4096
    assert output_budget({'kwargs': [{'num_words': 300, 'relation': 'at least'}]}) == 2048
    assert output_budget({'kwargs': [{'num_words': 300, 'relation': 'less than'}]}) == 768


def test_multiconstraint_coverage_requires_real_case_and_group_counts():
    data = [{'group': 'same', 'analysis_families': ['instruction', 'multiconstraint']} for _ in range(64)]
    assert not coverage(data)['multiconstraint']['ready']
    for i, row in enumerate(data):
        row['group'] = str(i)
    assert coverage(data)['multiconstraint']['ready']


def test_overlapping_family_summary_does_not_drop_multiconstraint():
    record = {'family': 'instruction', 'analysis_families': ['instruction', 'multiconstraint', 'termination'],
              'correct': True, 'group': 'one', 'output_tokens': 2, 'truncated': False,
              'eos_terminated': True, 'repetition_8gram': 0}
    result = summarize([record])
    assert set(result) == {'instruction', 'multiconstraint', 'termination'}
    assert all(r['scored_n'] == 1 for r in result.values())


def test_merge_guard_stops_before_approved_time_or_spending_ceiling():
    from research.capability_recovery.budget import usage
    s = {'started_at': 100, 'initial_credit': 11.2, 'hourly_rate': 1.08}
    assert not usage(s, 11.2, 100)['hard']
    assert usage(s, 11.2, 13600)['hard']
    assert usage(s, 6.4, 110)['hard']
    # Delayed provider accounting cannot conceal the elapsed hourly cost.
    assert usage(s, 11.2, 15000)['conservative_spent'] > 4.7


def test_number_scorer_requires_complete_declared_final_answer():
    c = {'scorer': 'number', 'answer': '12'}
    assert score_case(c, '2! × 3! = 12.\n#### 12')['correct']
    assert not score_case(c, 'Biraz düşünelim, belki 12')['correct']
    assert not score_case(c, '#### 12 sonra devam edelim 9')['correct']


def test_equal_models_do_not_establish_merge_superiority():
    from research.capability_recovery.merge_experiment import compare_results, FAMILIES
    records = [{'id': f'{family}/{i}', 'group': f'{family}/{i}', 'family': family,
                'correct': bool(i % 2), 'eos_terminated': True, 'repetition_8gram': 0}
               for family in FAMILIES for i in range(64)]
    r = {'records': records, 'manifest_sha256': 'same', 'complete': True}
    result = compare_results({'v2': r, 'v5': r, 'alpha_0.30': r})
    assert result['comparisons']['alpha_0.30']['nonregression_both']
    assert not result['superior']
    assert result['decision'] == 'no_demonstrated_superior_merge'


def test_secondary_content_audit_does_not_call_overlap_semantic_correctness():
    from research.capability_recovery.error_audit import memory_fact_proxy, overlap_f1, punctuation_em
    assert punctuation_em('Mehmed Ali Paşa.', 'Mehmed Ali Paşa')
    assert memory_fact_proxy('6 kişilik.', {'answers': ['6']})
    assert not memory_fact_proxy('16 kişilik.', {'answers': ['6']})
    assert not memory_fact_proxy('6 değil.', {'answers': ['6']})
    assert not memory_fact_proxy('6 veya 8 kişi.', {'answers': ['6']})
    # Lexical overlap can be positive for a contradiction; never promote it to
    # an entailment/reasoning score or replace the frozen acceptance gates.
    assert overlap_f1('Ankara değil', 'Ankara') > 0
    assert not memory_fact_proxy('Ankara değil', {'answers': ['Ankara']})
