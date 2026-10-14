import hashlib
import json
from types import SimpleNamespace
import pytest
import supervisor


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(supervisor,'ROOT',tmp_path)
    folder=tmp_path/'export';folder.mkdir()
    data=b'{"stage":"DONE"}'
    (folder/'status.json').write_bytes(data)
    (tmp_path/'status.json').write_text('{"stage":"DONE","verified":true}')
    digest=hashlib.sha256(data).hexdigest()
    (tmp_path/'upload_receipt.json').write_text(json.dumps({'verified':True,'repo':'test/model',
        'revision':'fixed','path':'benchmarks/date','sha256':{'status.json':digest}}))
    item=SimpleNamespace(rfilename='benchmarks/date/status.json',size=len(data),
        lfs=SimpleNamespace(sha256=digest))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda:SimpleNamespace(model_info=lambda *args,**kwargs:SimpleNamespace(siblings=[item])))
    return folder


def test_live_status_change_does_not_invalidate_immutable_export(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    assert supervisor.verify_export()['verified']


def test_changed_snapshot_cannot_pass_cleanup_verification(tmp_path,monkeypatch):
    folder=fixture(tmp_path,monkeypatch)
    (folder/'status.json').write_text('modified')
    with pytest.raises(AssertionError):supervisor.verify_export()
