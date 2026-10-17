import copy
import json
from pathlib import Path
from sft_v3.runtime import acceptance
from sft_v3.supervisor import budget

CFG=json.loads((Path(__file__).parent/'prepared_config.json').read_text())


def fixture():
    report={k:{'n':n,'correct':n,'truncated':0} for k,n in [('turkish_mmlu',900),('belebele_tr',900),('gsm8k_tr',200),('gsm8k_tr_cot8',200),('ifeval_tr_subset',120)]}
    language={'heldout_manifest_sha256':'same','scores':{k:{'mean_nll':1.8} for k in ['turkish','english']}}
    provenance={'training_steps':532,'expected_steps':532,'final_loss':.5,'baseline_loss':.6}
    return report,language,provenance


def test_acceptance_requires_instruction_gain_stopping_and_skill_retention():
    report,language,p=fixture()
    assert acceptance(CFG,report,language,language,p,False)['accepted']
    for task,field,value in [('ifeval_tr_subset','correct',59),('ifeval_tr_subset','truncated',13),('gsm8k_tr','correct',67)]:
        changed=copy.deepcopy(report);changed[task][field]=value
        assert not acceptance(CFG,changed,language,language,p,False)['accepted']
    assert not acceptance(CFG,report,language,language,p,True)['accepted']
    report['belebele_tr']['n']=899
    assert not acceptance(CFG,report,language,language,p,False)['accepted']


def test_incomplete_training_or_changed_retention_set_is_rejected():
    report,language,p=fixture();p['training_steps']=400
    assert not acceptance(CFG,report,language,language,p,False)['accepted']
    p['training_steps']=532;new=copy.deepcopy(language);new['heldout_manifest_sha256']='different'
    assert not acceptance(CFG,report,language,new,p,False)['accepted']
    new=copy.deepcopy(language);new['scores']['turkish']['mean_nll']=1.9
    assert not acceptance(CFG,report,language,new,p,False)['accepted']


def test_budget_reserves_export_room_and_enforces_dollar_and_time_caps():
    s={'initial_credit':20.11,'started_at':1000}
    assert budget(CFG,s,19.,1100)[:2]==(False,False)
    assert budget(CFG,s,16.,1100)[:2]==(False,True)
    assert budget(CFG,s,15.,1100)[0]
    assert budget(CFG,s,19.,1000+3*3600)[0]
