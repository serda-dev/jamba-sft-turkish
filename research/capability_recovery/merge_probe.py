"""Real-checkpoint coordinate probes, not whole-model merging or inference."""
import argparse
import json
from pathlib import Path
import struct
import torch

from .checkpoints import byte_range
from .common import digest, write_json
from .interference import parameter_group
from .merge import interpolate


def coordinate_probe(receipt, count=2048):
    from huggingface_hub import hf_hub_download, hf_hub_url, get_token
    from .checkpoints import remote_header
    references = receipt['references']
    if not receipt['pairs']['v2:v5']['compatible']:
        raise ValueError('Incompatible parents')
    samples = {}
    for parent in ['v2','v5']:
        ref = references[parent]
        prefix = ref['path']+'/'
        index = json.loads(Path(hf_hub_download(ref['repo'], prefix+'model.safetensors.index.json',
                                             revision=ref['revision'])).read_text())['weight_map']
        chosen = {}
        for name in sorted(ref['tensors']):
            group = parameter_group(name)
            if group not in chosen and ref['tensors'][name]['dtype'] == 'BF16':
                chosen[group] = name
        samples[parent] = {}
        headers = {}
        for group, name in chosen.items():
            shard = index[name]
            url = hf_hub_url(ref['repo'],prefix+shard,revision=ref['revision'])
            if shard not in headers:
                length = struct.unpack('<Q',byte_range(url,0,7,get_token()))[0]
                headers[shard] = (length, remote_header(url,get_token()))
            length, header = headers[shard]
            offsets = header[name]['data_offsets']
            start = 8+length+offsets[0]
            size = min(count*2, offsets[1]-offsets[0])
            raw = byte_range(url,start,start+size-1,get_token())
            samples[parent][name] = torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).clone()
    result = {}
    for name,a in samples['v2'].items():
        b = samples['v5'][name]
        result[name] = {'group':parameter_group(name),'coordinates':len(a),
                        'parent_changed_fraction':float((a!=b).float().mean()), 'alphas':{}}
        for alpha in [.15,.30,.50,.70]:
            merged = interpolate(a,b,alpha)
            result[name]['alphas'][str(alpha)] = {
                'all_finite': bool(torch.isfinite(merged).all()),
                'equal_to_v2_fraction':float((merged==a).float().mean()),
                'equal_to_v5_fraction':float((merged==b).float().mean()),
                'fp32_mean_abs_distance_to_v2':float((merged.float()-a.float()).abs().mean())}
    return {'status':'REAL_CHECKPOINT_COORDINATE_PROBE_ONLY','results':result,
            'scope':'First coordinates of one tensor per parameter group. No full model built, no inference, no capability conclusion.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',default='research/capability_recovery/results/checkpoints.json')
    p.add_argument('--output',default='research/capability_recovery/results/merge_probe.json')
    a=p.parse_args()
    receipt=json.loads(Path(a.checkpoints).read_text())
    result=coordinate_probe(receipt)
    result['checkpoint_receipt_sha256']=digest(a.checkpoints)
    write_json(a.output,result)


if __name__=='__main__':
    main()
