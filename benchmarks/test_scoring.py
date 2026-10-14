from scoring import score,checks
import pytest


def test_turkish_case_does_not_require_english():
    row={'instruction_id_list':['change_case:turkish_lowercase'],'kwargs':[{}]}
    assert score(row,'bugün istanbul şehrinde hava güzel. insanlar sokaklarda yürüyüş yapıyor ve türkçe konuşuyor.')[0]
    assert not score(row,'Today the weather is beautiful in London.')[0]


def test_json_and_bullets_use_actual_constraints():
    row={'instruction_id_list':['detectable_format:json_format'],'kwargs':[{}]}
    assert score(row,'{"sonuc": 3}')==[True]
    assert score(row,'Cevap: {"sonuc": 3}')==[False]
    row={'instruction_id_list':['detectable_format:number_bullet_lists'],'kwargs':[{'num_bullets':2}]}
    assert score(row,'* Bir\n* İki')==[True]
    assert score(row,'* Bir')==[False]


def test_missing_arguments_cannot_randomize_the_evaluation():
    with pytest.raises(ValueError):checks({'instruction_id_list':['length_constraints:number_words'],'kwargs':[{'num_words':None,'relation':None}]})
