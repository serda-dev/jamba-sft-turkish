"""Authenticate preserved pilot weights before continuing a failed full run."""
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):
            value.update(block)
    return value.hexdigest()


def eligible_pilots(outcomes):
    return [o for o in outcomes if o['loss_after']<o['loss_before'] and
            o['probe_after']>=o['probe_before']-.05]


def verified_pilots(root,cfg):
    root=Path(root)
    receipt=json.loads((root/'pilot_reuse.json').read_text())
    if digest(root/'manifest.json')!=receipt['data_manifest_sha256']:
        raise ValueError('Preserved pilots belong to a different dataset')
    for key in ['seed','max_length','learning_rate','pilot_steps','candidates']:
        if cfg[key]!=receipt['original_config'][key]:
            raise ValueError('Preserved pilot configuration changed: '+key)
    batch=cfg['micro_batch']*cfg['gradient_accumulation']
    old=receipt['original_config']
    if batch!=old['micro_batch']*old['gradient_accumulation']:
        raise ValueError('Recovery must preserve the effective batch')
    outcomes=json.loads((root/'pilot_results.json').read_text())
    if {o['candidate']['name'] for o in outcomes}!={c['name'] for c in cfg['candidates']}:
        raise ValueError('Incomplete preserved pilot comparison')
    for o in outcomes:
        candidate=next(c for c in cfg['candidates'] if c['name']==o['candidate']['name'])
        if o['candidate']!=candidate or not all(math.isfinite(o[k]) for k in ['loss_before','loss_after','probe_before','probe_after']):
            raise ValueError('Invalid preserved pilot result')
    eligible=eligible_pilots(outcomes)
    if not eligible:
        raise ValueError('No eligible preserved pilot')
    winner=min(eligible,key=lambda o:o['loss_after'])
    folder=Path(winner['path']).resolve()
    if folder.parent!=root.resolve() or folder.name!=receipt['selected_folder']:
        raise ValueError('Unexpected preserved pilot path')
    weights={p.relative_to(root).as_posix() for p in folder.glob('*.safetensors')}
    if not weights or not weights<=set(receipt['files']):
        raise ValueError('Preserved pilot weights are missing from receipt')
    for relative,sha in receipt['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(folder) or digest(path)!=sha:
            raise ValueError('Preserved pilot checksum mismatch: '+relative)
    return outcomes


def verify_memory_preflight(root,cfg):
    root=Path(root)
    report=json.loads((root/'memory_preflight.json').read_text())
    expected={k:cfg[k] for k in ['micro_batch','gradient_accumulation','max_length']}
    if not report.get('passed') or report['settings']!=expected or report['optimizer_steps']<2:
        raise ValueError('A successful matching worst-length memory preflight is required')
    if report['pilot_reuse_sha256']!=digest(root/'pilot_reuse.json'):
        raise ValueError('Memory preflight used different pilot weights')
