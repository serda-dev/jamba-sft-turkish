"""Pinned-parent merge experiment payload; runs only inside an authorized rental."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from .common import digest, write_json
from .pilot_job import status, ROOT


def main():
    from huggingface_hub import snapshot_download
    from pilot_gpu_preflight import check_gpu_runtime
    from pilot_hub import upload_checkpoint
    from .gpu_experiments import evaluate, model_receipt, evaluation_model
    os.environ['HF_TOKEN'] = (ROOT/'hf_token').read_text().strip()
    cfg = json.loads((ROOT/'config.json').read_text())
    assert cfg['user_authorized'] and cfg['max_cost_usd'] == 5 and cfg['max_seconds'] == 14400
    for name, sha in json.loads((ROOT/'code_manifest.json').read_text()).items():
        if digest(ROOT/name) != sha:
            raise ValueError('Payload checksum mismatch')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'absl-py==2.2.2',
                    'immutabledict==4.2.1', 'langdetect==1.0.9', 'nltk==3.9.1'], check=True)
    status('PREFLIGHT', result=check_gpu_runtime())
    parents = {}
    for key in ['v2', 'v5']:
        receipt = json.loads((ROOT/(key+'_receipt.json')).read_text())
        prefix = receipt['path']+'/'
        status('DOWNLOADING', parent=key, revision=receipt['revision'])
        path = Path(snapshot_download(receipt['repo'], revision=receipt['revision'],
                    allow_patterns=[prefix+name for name in receipt['files']], max_workers=4))/receipt['path']
        for name, meta in receipt['files'].items():
            if (path/name).stat().st_size != meta['size'] or digest(path/name) != meta['sha256']:
                raise ValueError('Parent full-file checksum mismatch')
        parents[key] = str(path)
    status('MERGE_RUNNING', batch_size=16, candidate_count=6)
    subprocess.run([sys.executable, '-u', '-m', 'research.capability_recovery.merge_experiment',
                    '--parent-a', parents['v2'], '--parent-b', parents['v5'],
                    '--manifest', str(ROOT/'manifest.json'), '--output-dir', str(ROOT),
                    '--batch-size', '16'], check=True)
    report = json.loads((ROOT/'comparison.json').read_text())
    if report['superior']:
        # Tie rule is frozen before generation: prefer the smaller V5 coefficient.
        name = sorted(report['superior'])[0]
        alpha = float(name.split('_')[1])
        selected = ROOT/'selected_merge'
        if selected.exists():
            raise ValueError('Refusing to overwrite selected checkpoint')
        model, tok, parent_receipt = evaluation_model(SimpleNamespace(
            model=parents['v2'], parent_b=parents['v5'], alpha=alpha, seed=20261019))
        model.save_pretrained(selected, safe_serialization=True, max_shard_size='2GB')
        tok.save_pretrained(selected)
        write_json(selected/'merge_provenance.json', {'parents': parent_receipt,
                   'arithmetic': 'FP32 CUDA interpolation, export BF16; identical path to development evaluation',
                   'production_accepted': False})
        del model, tok
        gc.collect()
        __import__('torch').cuda.empty_cache()
        receipt = upload_checkpoint(selected, cfg['output_repo'])
        write_json(ROOT/'selected_merge_receipt.json', receipt)
        selection = {'development_complete': True, 'model_receipt': model_receipt(selected),
                     'manifest_sha256': digest(ROOT/'manifest.json'), 'alpha': alpha,
                     'tie_rule': 'smallest successful alpha', 'production_accepted': False}
        write_json(ROOT/'selection.json', selection)
        status('FROZEN_CONFIRMATION', alpha=alpha, checkpoint_receipt=receipt['revision'])
        evaluate(SimpleNamespace(model=str(selected), parent_b=None, alpha=alpha, seed=20261019,
                 manifest=str(ROOT/'manifest.json'), split='confirmation', selection=str(ROOT/'selection.json'),
                 batch_size=16, code_image=None, output=str(ROOT/'confirmation_merge.json')))
        # Both parents are diagnostic references only; selection is frozen and
        # MUST NOT be changed after opening these reserved responses.
        for parent in ['v2', 'v5']:
            write_json(ROOT/('confirmation_'+parent+'_selection.json'), {
                'development_complete': True, 'model_receipt': model_receipt(parents[parent]),
                'manifest_sha256': digest(ROOT/'manifest.json')})
            evaluate(SimpleNamespace(model=parents[parent], parent_b=None, alpha=0., seed=20261019,
                     manifest=str(ROOT/'manifest.json'), split='confirmation',
                     selection=str(ROOT/('confirmation_'+parent+'_selection.json')),
                     batch_size=16, code_image=None, output=str(ROOT/('confirmation_'+parent+'.json'))))
    status('DONE', decision=report['decision'], superior=report['superior'],
           production_accepted=False, training_started=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('FAILED', error_type=type(error).__name__)
        raise
