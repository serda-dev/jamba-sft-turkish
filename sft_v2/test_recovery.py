import hashlib
import json
import pytest
from sft_v2.pilot_recovery import verified_pilots,verify_memory_preflight,digest


def fixture(root):
    candidates=[{'name':name,'repo':'owner/'+name,'revision':'a'*40,'subfolder':''} for name in ['sft','cpt']]
    old={'seed':42,'max_length':3072,'learning_rate':5e-6,'pilot_steps':120,'candidates':candidates,'micro_batch':4,'gradient_accumulation':8}
    cfg={**old,'micro_batch':2,'gradient_accumulation':16}
    (root/'manifest.json').write_text('{"dataset":"pinned"}')
    folder=root/'pilot_sft';folder.mkdir();weight=folder/'model.safetensors';weight.write_bytes(b'preserved selected weights')
    outcomes=[{'candidate':c,'path':str(root/('pilot_'+c['name'])),'loss_before':1.,'loss_after':.7 if c['name']=='sft' else .8,'probe_before':.2,'probe_after':.3} for c in candidates]
    (root/'pilot_results.json').write_text(json.dumps(outcomes))
    receipt={'original_config':old,'data_manifest_sha256':digest(root/'manifest.json'),'selected_folder':'pilot_sft','files':{'pilot_sft/model.safetensors':digest(weight)}}
    (root/'pilot_reuse.json').write_text(json.dumps(receipt))
    return cfg


def test_preserved_pilot_is_reusable_with_same_effective_batch(tmp_path):
    cfg=fixture(tmp_path)
    assert len(verified_pilots(tmp_path,cfg))==2


def test_effective_batch_change_is_rejected(tmp_path):
    cfg=fixture(tmp_path);cfg['gradient_accumulation']=8
    with pytest.raises(ValueError,match='effective batch'):verified_pilots(tmp_path,cfg)


def test_changed_data_or_weights_are_rejected(tmp_path):
    cfg=fixture(tmp_path)
    (tmp_path/'pilot_sft/model.safetensors').write_bytes(b'changed weights')
    with pytest.raises(ValueError,match='checksum mismatch'):verified_pilots(tmp_path,cfg)
    (tmp_path/'manifest.json').write_text('{"dataset":"changed"}')
    with pytest.raises(ValueError,match='different dataset'):verified_pilots(tmp_path,cfg)


def test_memory_report_must_match_batch_and_preserved_weights(tmp_path):
    cfg=fixture(tmp_path)
    report={'passed':True,'settings':{k:cfg[k] for k in ['micro_batch','gradient_accumulation','max_length']},'optimizer_steps':2,'pilot_reuse_sha256':digest(tmp_path/'pilot_reuse.json')}
    (tmp_path/'memory_preflight.json').write_text(json.dumps(report));verify_memory_preflight(tmp_path,cfg)
    cfg['micro_batch']=4
    with pytest.raises(ValueError,match='matching worst-length'):verify_memory_preflight(tmp_path,cfg)
    cfg['micro_batch']=2;report['pilot_reuse_sha256']='bad'
    (tmp_path/'memory_preflight.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='different pilot weights'):verify_memory_preflight(tmp_path,cfg)
