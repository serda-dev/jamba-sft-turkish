import copy
from sft_v3.data import generated, verify, external_check, FAMILIES


def test_all_families_and_targets_satisfy_executable_constraints():
    rows=list(generated(320,123))
    assert {r['family'] for r in rows} == set(FAMILIES)
    assert all(verify(r,r['messages'][-1]['content']) for r in rows)


def test_formatting_alone_does_not_hide_wrong_facts():
    row=next(r for r in generated(100,55) if r['family']=='json')
    assert not verify(row,'{}')
    row=next(r for r in generated(100,55) if r['family']=='paragraphs')
    assert not verify(row,'Yanlış bilgi.\n\nİkinci paragraf.')


def test_extra_commentary_and_repetition_are_rejected():
    for row in generated(64,66):
        if row['family'] in ['json','grounded_json','csv','quotes','ending','placeholders']:
            assert not verify(row,row['messages'][-1]['content']+'\nEk açıklama.')
    row=next(r for r in generated(100,55) if r['family']=='frequency')
    assert not verify(row,row['messages'][-1]['content']*2)


def test_external_source_is_filtered_instead_of_trusted_blindly():
    row={'constraints':['punctuation:use no comma'],'prompt':'No commas.',
         'messages':[{'role':'assistant','content':'Wrong, answer.'}]}
    assert external_check(row) is False
    row['messages'][0]['content']='A valid short answer.'
    assert external_check(row) is True
    row['constraints']=['unknown']
    assert external_check(row) is None
    row['constraints']=['punctuation:use no comma','unknown']
    assert external_check(row) is None


def test_development_identifiers_and_prompts_are_disjoint():
    train=list(generated(128,11));dev=list(generated(128,12,True))
    assert not ({r['group'] for r in train}&{r['group'] for r in dev})
    assert not ({r['messages'][1]['content'] for r in train}&{r['messages'][1]['content'] for r in dev})
