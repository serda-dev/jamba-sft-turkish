import pytest
from research.instruction_following_20261017.contracts import check


def test_explicit_paragraph_separator_is_not_interchangeable():
    stars=[{'kind':'paragraphs','count':2,'separator':'***'}]
    blank=[{'kind':'paragraphs','count':2,'separator':'\n\n'}]
    assert check(stars,'Birinci paragraf.\n***\nİkinci paragraf.')
    assert not check(stars,'Birinci paragraf.\n\nİkinci paragraf.')
    assert check(blank,'Birinci paragraf.\n\nİkinci paragraf.')
    assert not check(blank,'Birinci paragraf.\n***\nİkinci paragraf.')


def test_total_keyword_frequency_cannot_replace_per_line_requirements():
    rules=[{'kind':'keyword_lines','count':2,'keyword':'kültür'}]
    assert check(rules,'kültür kaydı hazır.\nkültür notu tamam.')
    assert not check(rules,'kültür kültür')


def test_two_identical_answers_are_rejected():
    rules=[{'kind':'two_answers','separator':'******'}]
    assert check(rules,'İlk yanıt.\n******\nBaşka yanıt.')
    assert not check(rules,'Aynı yanıt.\n******\nAynı yanıt.')


def test_boolean_is_not_accepted_instead_of_integer():
    rules=[{'kind':'json_exact','value':{'sonuc':1}}]
    assert check(rules,'{"sonuc": 1}')
    assert not check(rules,'{"sonuc": true}')


def test_all_declared_conditions_apply_and_conflicts_fail_closed():
    rules=[{'kind':'paragraphs','count':2,'separator':'***'},{'kind':'ending','value':'Tamamlandı.'}]
    assert check(rules,'İlk bilgi.\n***\nİkinci bilgi. Tamamlandı.')
    assert not check(rules,'İlk bilgi.\n***\nİkinci bilgi.')
    with pytest.raises(ValueError,match='Conflicting'):
        check([{'kind':'paragraphs','count':2,'separator':'***'},{'kind':'paragraphs','count':3,'separator':'***'}],'Örnek')
    with pytest.raises(ValueError,match='Unsupported'):check([{'kind':'unimplemented'}],'Örnek')


def test_required_fact_does_not_match_inside_identifier():
    rules=[{'kind':'required','values':['23']}]
    assert check(rules,'Stok 23 adet.')
    assert not check(rules,'İşlem P8123.')


def test_each_answer_must_contain_requested_code():
    rules=[{'kind':'two_answers','separator':'******','each_required':['P81']}]
    assert check(rules,'P81 hazır.******P81 işlemi tamam.')
    assert not check(rules,'P81 hazır.******İşlem tamam.')


def test_pilot_targets_and_split_separation():
    from research.instruction_following_20261017.prepare_pilot import cases
    train,dev=cases(32,'train'),cases(128,'dev')
    assert len(train)==32 and len(dev)==128
    assert len({r['family'] for r in train})==8
    assert not ({r['messages'][0]['content'] for r in train}&{r['messages'][0]['content'] for r in dev})
    assert all(check(r['contracts'],r['expected']) for r in train+dev)
