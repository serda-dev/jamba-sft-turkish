import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from safetensors import safe_open

from research.capability_recovery.checkpoints import tokenizer_identity
from research.capability_recovery.evaluation import (
    constraint_score, group_split, paired_difference, pareto_front, score_case, wilson)
from research.capability_recovery.interference import pcgrad, parameter_group
from research.capability_recovery.merge import merge, interpolate
from research.capability_recovery.precision import MasterAccumulator, sample
from research.capability_recovery.sampling import balanced_schedule
from research.capability_recovery.selection import compare


def test_bf16_micro_accumulation_loses_small_gradients_but_fp32_sum_retains_them():
    values = [1.] + [.001]*31
    results = {}
    for mode in ['existing', 'fp32_micro']:
        model = torch.nn.Linear(1, 1, bias=False).bfloat16()
        acc = MasterAccumulator(model, mode)
        acc.clear()
        for value in values:
            (model.weight*value).sum().backward()
            acc.after_micro()
        acc.finish()
        results[mode] = acc.masters[0][1].grad.item()
    expected = sum(torch.tensor(v, dtype=torch.bfloat16).item() for v in values)
    assert results['fp32_micro'] == expected
    assert results['existing'] != expected


def test_master_updates_accumulate_before_working_bf16_rounding():
    model = torch.nn.Linear(1, 1, bias=False).bfloat16()
    with torch.no_grad():
        model.weight.fill_(.1)
    acc = MasterAccumulator(model, 'fp32_micro')
    before = model.weight.item()
    for _ in range(200):
        with torch.no_grad():
            acc.masters[0][1].sub_(2e-6)
        acc.sync()
    assert model.weight.item() != before
    assert acc.masters[0][1].dtype == torch.float32
    assert model.weight.dtype == torch.bfloat16


def test_master_deduplicates_actual_tied_parameter_and_clears_unused_gradient():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    model.b = model.a
    acc = MasterAccumulator(model, 'fp32_micro')
    assert len(acc.masters) == 1
    model.a.sum().backward()
    acc.after_micro()
    assert model.a.grad is None
    assert torch.equal(acc.masters[0][1].grad, torch.ones(2))
    acc.clear()
    assert acc.masters[0][1].grad is None


def test_nonfinite_gradient_fails_closed():
    model = torch.nn.Linear(1, 1, bias=False).bfloat16()
    acc = MasterAccumulator(model, 'fp32_micro')
    (model.weight*float('nan')).sum().backward()
    with pytest.raises(FloatingPointError):
        acc.after_micro()


def test_sample_indices_are_int64_and_bounded():
    x = torch.arange(5)
    assert sample(x, 1024).tolist() == list(range(5))


def test_tokenizer_padding_is_not_vocab_identity():
    a = {'model': {'vocab': {'A': 0}}, 'padding': {'direction': 'Left'}, 'added_tokens': []}
    b = {**a, 'padding': None}
    assert tokenizer_identity(a) == tokenizer_identity(b)
    b['model'] = {'vocab': {'A': 1}}
    assert tokenizer_identity(a) != tokenizer_identity(b)


def test_sampler_controls_tokens_and_stops_on_shortage():
    data = [{'family':'a', 'target_tokens':10}, {'family':'b', 'target_tokens':100}]
    selected, report = balanced_schedule(data, 1000, {'a':.5,'b':.5}, max_reuse=2)
    assert not report['complete']
    assert report['max_reuse_observed'] <= 2
    assert report['shortages']['a'] == 480
    assert selected == balanced_schedule(data, 1000, {'a':.5,'b':.5}, max_reuse=2)[0]


def test_sampler_rejects_unclassified_reasoning_as_verified_math():
    with pytest.raises(ValueError):
        balanced_schedule([{'family':'reasoning', 'target_tokens':50}], 50)


def case(identifier, group, text):
    return {'id': identifier, 'group': group, 'messages':[{'role':'user','content':text}]}


def test_group_split_is_order_independent_and_groups_paraphrases():
    data = [case('a','same','bir'), case('b','same','iki'), case('c','other','üç')]
    a = group_split(data)
    b = group_split(list(reversed(data)))
    assert a == b
    assert a['a'] == a['b']


def test_group_split_joins_exact_and_long_near_duplicate_prompts():
    common = ' '.join('word'+str(i) for i in range(20))
    data = [case('a','one',common+' a'), case('b','two',common+' b'), case('c','three',common+' b')]
    split = group_split(data)
    assert split['a'] == split['b'] == split['c']


def test_paired_counts_and_missing_ids():
    a = [{'id': str(i), 'group':str(i), 'correct':v} for i,v in enumerate([True,False,True])]
    b = copy.deepcopy(a)
    b[0]['correct'], b[1]['correct'] = False, True
    r = paired_difference(a,b)
    assert r['gained'] == r['lost'] == 1
    assert r['difference'] == 0
    with pytest.raises(ValueError):
        paired_difference(a,b[:-1])


def test_unscored_semantic_task_is_not_counted_as_success():
    c = {'scorer':'manual','contracts':[{'kind':'endswith','value':'son'}]}
    out = score_case(c, 'yanlış içerik son')
    assert out['format_compliant'] is True
    assert out['correct'] is None


def test_all_constraints_required_and_no_extra_json_text():
    assert constraint_score([{'kind':'json','value':{'a':1}}], '{"a":1}')
    assert not constraint_score([{'kind':'json','value':{'a':1}}], 'İşte: {"a":1}')
    assert not constraint_score([{'kind':'endswith','value':'son'},{'kind':'lines','value':2}], 'son')
    assert not constraint_score([{'kind':'json','value':{'a':1}}], '{"a":0,"a":1}')
    assert not constraint_score([{'kind':'json','value':{'a':True}}], '{"a":1}')
    assert not constraint_score([{'kind':'json','value':{'a':1}}], '{"a":NaN}')


def test_math_numeric_equivalence_is_not_string_identity():
    assert score_case({'scorer':'number','answer':'1.5'}, 'Çözüm: #### 1,50')['correct']


def test_wilson_handles_zero_and_perfect_without_false_certainty():
    assert wilson(0,0) is None
    assert wilson(10,10)[0] < 1
    assert wilson(0,10)[1] > 0


def test_pareto_preserves_tradeoffs_not_arbitrary_sum():
    assert set(pareto_front({'a':{'math':1.,'instruction':0.}, 'b':{'math':0.,'instruction':1.},
                             'c':{'math':0.,'instruction':0.}})) == {'a','b'}


def test_pcgrad_projects_conflicts_and_handles_zero_vectors():
    assert torch.equal(pcgrad([torch.tensor([1.,0.]),torch.tensor([-1.,0.])]), torch.zeros(2))
    assert torch.isfinite(pcgrad([torch.zeros(2),torch.ones(2)])).all()


def test_interpolation_fp32_and_nonfinite_parent():
    a = torch.tensor([.1,.2],dtype=torch.bfloat16)
    b = torch.tensor([.2,.3],dtype=torch.bfloat16)
    assert torch.equal(interpolate(a,b,.3), ((1-.3)*a.float()+.3*b.float()).bfloat16())
    with pytest.raises(ValueError):
        interpolate(a,torch.tensor([float('nan'),0],dtype=a.dtype),.3)


def parent(path, value, vocab=1):
    path.mkdir()
    (path/'config.json').write_text(json.dumps({'vocab_size':vocab,'tie_word_embeddings':False}))
    (path/'tokenizer.json').write_text(json.dumps({'model':{'vocab':{'a':0}},'padding':None}))
    save_file({'weight':torch.full((4,2), value, dtype=torch.bfloat16)}, str(path/'model.safetensors'))


def test_real_safetensors_merge_roundtrip_and_parent_preservation(tmp_path):
    a,b,out = [tmp_path/p for p in ['a','b','out']]
    parent(a,1.)
    parent(b,3.)
    before = (a/'model.safetensors').read_bytes()
    merge(a,b,out,.3,shard_bytes=8)
    index = json.loads((out/'model.safetensors.index.json').read_text())
    with safe_open(out/index['weight_map']['weight'], framework='pt') as f:
        assert torch.equal(f.get_tensor('weight'), torch.full((4,2),1.6,dtype=torch.bfloat16))
    assert (a/'model.safetensors').read_bytes() == before
    with pytest.raises(ValueError):
        merge(a,b,out,.5)


def test_incompatible_merge_rejected_before_output_creation(tmp_path):
    a,b,out = [tmp_path/p for p in ['a','b','out']]
    parent(a,1.)
    parent(b,3.,vocab=2)
    with pytest.raises(ValueError):
        merge(a,b,out,.5)
    assert not out.exists()


def test_selection_rejects_historical_and_final_outputs():
    policy = {'families':['math'],'minimum_scored_per_family':1,'minimum_groups_per_family':1,
              'maximum_accuracy_drop':.03,'maximum_ci_lower_drop':.08}
    base = {'split':'final','complete':True,'manifest_sha256':'a','records':[]}
    with pytest.raises(ValueError):
        compare(base,base,policy)


def test_scarce_dev_does_not_silently_pass_acceptance():
    policy = {'families':['math'],'minimum_scored_per_family':64,'minimum_groups_per_family':32,
              'maximum_accuracy_drop':.03,'maximum_ci_lower_drop':.08}
    base = {'split':'dev','complete':True,'manifest_sha256':'a',
            'records':[{'id':'x','group':'one','family':'math','correct':True,'truncated':False}]}
    result = compare(base,base,policy)
    assert result['eligible_for_selection'] is False


def test_language_gate_requires_identical_data_and_token_counts():
    from research.capability_recovery.acceptance import language_gates
    a={'complete':True,'data_sha256':'same','scores':{
        'turkish':{'tokens':100,'nll':1.7},'english':{'tokens':100,'nll':1.8}}}
    b=copy.deepcopy(a)
    policy={'turkish_nll_max_ratio':1.02,'english_nll_max_ratio':1.03}
    assert all(language_gates(a,b,policy).values())
    b['scores']['turkish']['tokens']=99
    assert not language_gates(a,b,policy)['turkish']
    b['data_sha256']='different'
    assert not any(language_gates(a,b,policy).values())


def test_existing_accumulator_matches_actual_imported_historical_helper():
    from research.instruction_following_20261017.pilot_runtime import make_master_parameters,copy_gradients_to_master
    model=torch.nn.Linear(2,1,bias=False).bfloat16()
    acc=MasterAccumulator(model,'existing')
    for g in [.1,.003,.006]:
        (model.weight*g).sum().backward()
    historical=make_master_parameters(list(model.named_parameters()))
    copy_gradients_to_master(list(model.named_parameters()),historical)
    acc.finish()
    assert torch.equal(historical[0][1].grad,acc.masters[0][1].grad)


def test_final_assessment_refuses_unready_data_and_wrong_model():
    from research.capability_recovery.acceptance import assess
    a={'split':'final','model_receipt':{'w':'a'},'manifest_sha256':'f'}
    selection={'model_receipt':{'w':'b'},'manifest_sha256':'f','development_complete':True}
    with pytest.raises(ValueError):
        assess(a,a,selection,{'final_ready':True},{},{},{})
    selection['model_receipt']=a['model_receipt']
    with pytest.raises(ValueError):
        assess(a,a,selection,{'final_ready':False},{},{},{})


def test_leakage_exact_short_prompt_inside_chat_roles_is_removed(tmp_path):
    from research.capability_recovery.prepare_eval import exclude_historical
    path=tmp_path/'train.jsonl'
    path.write_text(json.dumps({'input_ids':[1]})+'\n')
    class Tokenizer:
        def decode(self,*args,**kwargs):
            return '<|im_start|>user\nKısa soru?\n<|im_end|>\n<|im_start|>assistant\nCevap.<|im_end|>'
    data=[case('a','g','Kısa soru?'),case('b','h','Başka soru?')]
    kept,report=exclude_historical(data,[path],Tokenizer())
    assert [r['id'] for r in kept]==['b']
    assert report['groups_removed']==1


def test_pilot_budget_guards_cost_time_and_delayed_billing():
    from research.capability_recovery.pilot_supervisor import limits
    state={'started_at':100.,'initial_credit':11.,'hourly_rate':1.}
    assert not limits(state,11.,now=200.)['hard']
    assert limits(state,11.,now=3400.)['hard']
    assert limits(state,9.1,now=200.)['hard']
    assert limits(state,20.,now=3400.)['hard']
