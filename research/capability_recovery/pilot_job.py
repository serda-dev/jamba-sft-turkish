"""Bounded GPU payload: pinned parent, verified data, disposable diagnostics and backups."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from .common import digest, write_json

ROOT=Path('/mnt/home_extra/capability')


def status(stage, **fields):
    write_json(ROOT/'status.json',{'stage':stage,'updated_at':time.time(),**fields})
    print(stage,json.dumps(fields),flush=True)


def upload_checkpoint(path):
    from pilot_hub import upload_checkpoint as upload
    receipt=upload(path,json.loads((ROOT/'config.json').read_text())['output_repo'])
    write_json(ROOT/(path.name+'_receipt.json'),receipt)
    status('CHECKPOINT_BACKED_UP',checkpoint=path.name,revision=receipt['revision'])
    return receipt


def main():
    from huggingface_hub import snapshot_download
    import torch
    from pilot_gpu_preflight import check_gpu_runtime
    from .gpu_experiments import precision_experiment, interference_experiment
    os.environ['HF_TOKEN']=(ROOT/'hf_token').read_text().strip()
    cfg=json.loads((ROOT/'config.json').read_text())
    code=json.loads((ROOT/'code_manifest.json').read_text())
    for name,sha in code.items():
        if digest(ROOT/name)!=sha:
            raise ValueError('Payload hash mismatch')
    torch.set_num_threads(8)
    status('PREFLIGHT',result=check_gpu_runtime())
    receipt=json.loads((ROOT/'parent_receipt.json').read_text())
    prefix=receipt['path']+'/'
    status('DOWNLOADING_PARENT',revision=receipt['revision'])
    source=Path(snapshot_download(receipt['repo'],revision=receipt['revision'],
                allow_patterns=[prefix+name for name in receipt['files']],max_workers=4))/receipt['path']
    for name,meta in receipt['files'].items():
        if (source/name).stat().st_size!=meta['size'] or digest(source/name)!=meta['sha256']:
            raise ValueError('Parent checkpoint payload checksum mismatch')
    for name,sha in cfg['data_sha256'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('Diagnostic data mismatch')
    write_json(ROOT/'verified_parent.json',{'revision':receipt['revision'],
               'files_verified':len(receipt['files']),'full_weight_hashes_verified':True})
    args=SimpleNamespace(model=str(source),data=str(ROOT/'precision.jsonl'),
            output=str(ROOT/'precision_result.json'),steps=4,accumulation=32,
            lr=7.5e-7,seed=20261019,checkpoint_root=str(ROOT/'checkpoints'),
            checkpoint_callback=upload_checkpoint)
    status('PRECISION_RUNNING',steps_per_arm=4,accumulation=32)
    precision_experiment(args)
    status('INTERFERENCE_RUNNING',families=cfg['gradient_families'])
    args=SimpleNamespace(model=str(source),data=str(ROOT/'interference.jsonl'),
            output=str(ROOT/'interference_result.json'),seed=20261019,
            families=cfg['gradient_families'],repeats=3,sample_size=256)
    interference_experiment(args)
    status('DONE',precision_complete=True,interference_complete=True,
           new_sft_trained=False,production_replaced=False)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        status('FAILED',error_type=type(error).__name__)
        raise
