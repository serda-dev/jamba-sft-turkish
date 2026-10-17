"""Resume final reporting from a completed, exported model; never train again."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

if __package__:
    from .train import ROOT, finalize, status
else:
    from train import ROOT, finalize, status


def completed_inputs(root):
    cfg = json.loads((root / 'config.json').read_text())
    state = json.loads((root / 'trainer_state_final.json').read_text())
    receipt = json.loads((root / 'final_receipt.json').read_text())
    provenance = json.loads((root / 'final/sft_provenance.json').read_text())
    if not (state['global_step'] == state['max_steps'] == provenance['training_steps']
            and state['epoch'] >= cfg['epochs'] and receipt['verified']
            and receipt['repo_id'] == cfg['output_repo']):
        raise ValueError('Completed training and verified final export required')
    if hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest() != provenance['source_manifest_sha256']:
        raise ValueError('Data manifest changed')
    for name, info in receipt['files'].items():
        path = root / 'final' / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != info['size'] or digest.hexdigest() != info['sha256']:
            raise ValueError('Exported model checksum mismatch: ' + name)
    return cfg, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-loss', required=True, type=float)
    args = parser.parse_args()
    if not math.isfinite(args.baseline_loss) or args.baseline_loss <= 0:
        raise ValueError('Finite original development baseline required')
    cfg, provenance = completed_inputs(ROOT)
    os.environ['HF_TOKEN'] = (ROOT / 'hf_token').read_text().strip()
    import torch
    torch.set_num_threads(8)
    status('FINAL_BENCHMARK', recovery=True, verified=True,
           baseline_loss=args.baseline_loss,
           final_loss=provenance['assistant_validation_loss'])
    finalize(ROOT / 'final', cfg, args.baseline_loss,
             provenance['assistant_validation_loss'], provenance['selected_start'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('FAILED', error=type(error).__name__)
        raise
