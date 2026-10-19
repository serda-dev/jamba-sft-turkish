"""Pinned remote metadata and tensor headers; deliberately never download weights."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from .common import write_json

REFERENCES = {
    'v2': {'repo': 'serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015',
           'revision': 'c8ee3888c486fa49d8d9e798a632c9f7ce664d7a', 'path': 'checkpoints/final'},
    'v5': {'repo': 'serda-dev/Jamba2-3B-Turkish-SFT-v5-cont-20261018',
           'revision': '7cca33bea32d5d7f745b76fa15fcd224c23fe3a2', 'path': 'checkpoints/final'},
    'v4b': {'repo': 'serda-dev/Jamba2-3B-Turkish-SFT-v4b-20261018',
            'revision': '7acde389be357602af459076a02054db56e7ae09', 'path': 'checkpoints/final'},
    'cpt': {'repo': 'serda-dev/Jamba2-3B-Turkish-CPT-1B-20261007',
            'revision': '8b1a9c812791fa7930e64b9444b41519c023269e', 'path': 'checkpoints/final'},
    'original': {'repo': 'ai21labs/AI21-Jamba2-3B',
                 'revision': '525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9', 'path': ''},
}


def byte_range(url, start, end, token=None):
    import requests
    # Range query prevents intermediary cache collisions between different ranges.
    sep = '&' if '?' in url else '?'
    headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    with requests.get(url + sep + f'capability_range={start}-{end}', headers=headers,
                      timeout=(20, 120), stream=True) as r:
        r.raise_for_status()
        expected = f'bytes {start}-{end}/'
        if r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(expected):
            raise ValueError('Server did not honor the exact byte range; full download refused')
        data = r.raw.read(end - start + 2)
        if len(data) != end - start + 1:
            raise ValueError('Unexpected range length')
        return data


def remote_header(url, token=None, expected_size=None):
    length = struct.unpack('<Q', byte_range(url, 0, 7, token))[0]
    if not 2 <= length <= 32 << 20:
        raise ValueError('Invalid safetensors header length')
    header = json.loads(byte_range(url, 8, 7 + length, token))
    widths = {'BF16':2, 'F16':2, 'F32':4, 'F64':8, 'I64':8, 'I32':4, 'I16':2, 'I8':1, 'U8':1, 'BOOL':1}
    offsets = []
    for name, entry in header.items():
        if name == '__metadata__':
            continue
        start, end = entry['data_offsets']
        if start < 0 or end-start != math.prod(entry['shape'])*widths[entry['dtype']]:
            raise ValueError('Tensor shape/offset mismatch')
        offsets.append((start,end))
    offsets.sort()
    if offsets and (offsets[0][0] != 0 or any(a[1] != b[0] for a,b in zip(offsets,offsets[1:]))):
        raise ValueError('Overlapping or non-contiguous tensor payloads')
    if expected_size is not None and offsets[-1][1] + 8 + length != expected_size:
        raise ValueError('Safetensors size mismatch')
    return header


def signature(header):
    return {k: {'shape': v['shape'], 'dtype': v['dtype']} for k, v in header.items()
            if k != '__metadata__'}


def tokenizer_identity(raw):
    value = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    # Runtime padding/truncation policy is explicitly fixed by the evaluator.
    # All model vocab, merges, normalizer, pre/post processors and added token IDs
    # must remain identical. A byte-hash mismatch alone does not prove ID mismatch.
    value.pop('padding', None)
    value.pop('truncation', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def inspect_reference(ref):
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_token
    api = HfApi()
    info = api.model_info(ref['repo'], revision=ref['revision'], files_metadata=True)
    pinned = {**ref, 'revision': info.sha}
    prefix = ref['path'].rstrip('/') + '/' if ref['path'] else ''
    files = {}
    configs = {}
    tensors = {}
    order = []
    aliases = {}
    identity = None
    for sibling in info.siblings:
        name = sibling.rfilename
        if not name.startswith(prefix) or '/' in name[len(prefix):]:
            continue
        short = name[len(prefix):]
        if short.endswith('.safetensors'):
            head = remote_header(hf_hub_url(ref['repo'], name, revision=info.sha), get_token(), sibling.size)
            sig = signature(head)
            if set(tensors) & set(sig):
                raise ValueError('Duplicate tensor keys across shards')
            tensors.update(sig)
            order.extend(sig)
            aliases.update(head.get('__metadata__', {}))
            files[short] = {'size': sibling.size,
                            'sha256': sibling.lfs.sha256 if sibling.lfs else None,
                            'content_hash_verified': False, 'header_verified': True}
        elif short.endswith(('.json', '.jinja')):
            p = Path(hf_hub_download(ref['repo'], name, revision=info.sha))
            raw = p.read_bytes()
            if short == 'tokenizer.json':
                identity = tokenizer_identity(raw)
            files[short] = {'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                            'content_hash_verified': True}
            if short in ['config.json', 'tokenizer_config.json', 'generation_config.json',
                         'special_tokens_map.json', 'sft_provenance.json']:
                configs[short] = json.loads(raw)
    if not tensors or 'config.json' not in configs or 'tokenizer.json' not in files:
        raise ValueError('Incomplete checkpoint')
    return {**pinned, 'available': True, 'files': files, 'configs': configs,
            'tensors': tensors, 'tensor_order': order, 'serialization_metadata': aliases,
            'tokenizer_semantic_sha256': identity,
            'stored_elements': sum(math.prod(v['shape']) for v in tensors.values()),
            'scope': 'Remote headers + small-file hashes; weight payload integrity requires download hashing'}


def compatibility(a, b):
    failures = []
    if a['tensors'] != b['tensors']:
        failures.append('tensor_names_shapes_or_dtypes')
    if a['tokenizer_semantic_sha256'] != b['tokenizer_semantic_sha256']:
        failures.append('tokenizer_semantics')
    for name in ['tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja']:
        if a['files'].get(name, {}).get('sha256') != b['files'].get(name, {}).get('sha256'):
            failures.append(name)
    fields = ['model_type', 'architectures', 'vocab_size', 'hidden_size', 'num_hidden_layers',
              'num_attention_heads', 'num_key_value_heads', 'intermediate_size', 'layers_block_type',
              'attn_layer_period', 'attn_layer_offset', 'mamba_d_state', 'mamba_d_conv',
              'mamba_expand', 'mamba_dt_rank', 'tie_word_embeddings', 'bos_token_id',
              'eos_token_id', 'pad_token_id']
    for field in fields:
        if a['configs']['config.json'].get(field) != b['configs']['config.json'].get(field):
            failures.append('config.' + field)
    return {'compatible': not failures, 'failures': failures,
            'same_serialized_order': a['tensor_order'] == b['tensor_order'],
            'tokenizer_file_bytes_identical': a['files']['tokenizer.json']['sha256'] == b['files']['tokenizer.json']['sha256'],
            'required_runtime_padding': 'left during generation, no target truncation',
            'matching_by': 'tensor name, never zip by order',
            'runtime_shared_storage_verified': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='research/capability_recovery/results/checkpoints.json')
    args = p.parse_args()
    report = {'references': {}, 'pairs': {}}
    for name, ref in REFERENCES.items():
        try:
            report['references'][name] = inspect_reference(ref)
            print(name, 'OK', flush=True)
        except Exception as e:
            # Avoid recording exception URLs with signed access credentials.
            report['references'][name] = {**ref, 'available': False, 'error_type': type(e).__name__}
            print(name, type(e).__name__, flush=True)
        write_json(args.output, report)
    for name in ['v5', 'v4b', 'cpt', 'original']:
        a, b = report['references'].get('v2'), report['references'].get(name)
        if a.get('available') and b.get('available'):
            report['pairs']['v2:' + name] = compatibility(a, b)
    write_json(args.output, report)


if __name__ == '__main__':
    main()
