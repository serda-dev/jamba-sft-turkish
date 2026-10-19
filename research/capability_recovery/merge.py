"""FP32 two-parent interpolation, bounded by a shard. No implicit downloads/uploads."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .common import digest, write_json
from .checkpoints import tokenizer_identity


def local_signature(path):
    result = {}
    for shard in sorted(Path(path).glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key in result:
                    raise ValueError('Duplicate tensor')
                result[key] = (shard, f.get_slice(key).get_shape(), f.get_slice(key).get_dtype())
    if not result:
        raise ValueError('No safetensors weights')
    return result


def check_local(a, b):
    sa, sb = local_signature(a), local_signature(b)
    if {k: v[1:] for k, v in sa.items()} != {k: v[1:] for k, v in sb.items()}:
        raise ValueError('Tensor names, shapes or dtypes incompatible')
    if tokenizer_identity((Path(a)/'tokenizer.json').read_bytes()) != tokenizer_identity((Path(b)/'tokenizer.json').read_bytes()):
        raise ValueError('Tokenizer vocabulary or processing mismatch')
    for name in ['tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja']:
        pa, pb = Path(a)/name, Path(b)/name
        if pa.exists() != pb.exists() or (pa.exists() and digest(pa) != digest(pb)):
            raise ValueError('Tokenizer/template mismatch: ' + name)
    ca, cb = [json.loads((Path(p)/'config.json').read_text()) for p in [a, b]]
    # Ignore only non-architectural serialization provenance.
    ignore = {'_name_or_path', 'transformers_version', 'torch_dtype', 'dtype'}
    if {k:v for k,v in ca.items() if k not in ignore} != {k:v for k,v in cb.items() if k not in ignore}:
        raise ValueError('Configuration mismatch')
    return sa, sb, ca


def interpolate(a, b, alpha):
    if not 0 <= alpha <= 1 or a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('Invalid interpolation')
    if not a.is_floating_point():
        if not torch.equal(a, b):
            raise ValueError('Unequal non-floating state')
        return a.clone()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite parent')
    return a.float().mul(1-alpha).add(b.float(), alpha=alpha).to(a.dtype)


def merge(a, b, output, alpha, shard_bytes=512 << 20):
    a, b, output = Path(a).resolve(), Path(b).resolve(), Path(output).resolve()
    if output.exists() or output == a or output == b or a in output.parents or b in output.parents:
        raise ValueError('Output must be new and outside parent checkpoints')
    sa, sb, cfg = check_local(a, b)
    required = sum(p.stat().st_size for p in a.glob('*.safetensors'))
    if shutil.disk_usage(output.parent).free < required + (1 << 30):
        raise OSError('Insufficient disk space; parents will not be removed')
    output.mkdir()
    index, tensors, size, shard_no = {}, {}, 0, 0
    tensor_bytes = 0
    def flush():
        nonlocal tensors, size, shard_no
        if not tensors:
            return
        name = f'model-{shard_no:05d}.safetensors'
        save_file(tensors, str(output/name), metadata={'format': 'pt'})
        for key in tensors:
            index[key] = name
        tensors, size, shard_no = {}, 0, shard_no+1
    try:
        for key in sorted(sa):
            with safe_open(sa[key][0], framework='pt', device='cpu') as fa, safe_open(sb[key][0], framework='pt', device='cpu') as fb:
                tensor = interpolate(fa.get_tensor(key), fb.get_tensor(key), alpha)
            if size + tensor.nbytes > shard_bytes:
                flush()
            tensors[key] = tensor
            size += tensor.nbytes
            tensor_bytes += tensor.nbytes
        flush()
        # Tied groups are serialized consistently. Transformers re-establishes ties
        # on load; tied parents with unequal embed/head payloads are rejected below.
        if cfg.get('tie_word_embeddings'):
            embed = next((k for k in index if k.endswith('embed_tokens.weight')), None)
            if embed and 'lm_head.weight' in index:
                with safe_open(output/index[embed], framework='pt') as e, safe_open(output/index['lm_head.weight'], framework='pt') as h:
                    if not torch.equal(e.get_tensor(embed), h.get_tensor('lm_head.weight')):
                        raise ValueError('Inconsistent tied embedding/head payloads')
        write_json(output/'model.safetensors.index.json', {'metadata': {'total_size': tensor_bytes}, 'weight_map': index})
        for path in a.iterdir():
            if path.name in ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                             'special_tokens_map.json', 'chat_template.jinja', 'tokenizer.model']:
                shutil.copy2(path, output/path.name)
        write_json(output/'merge_provenance.json', {'alpha': alpha, 'arithmetic': 'FP32 then parent dtype',
                   'parent_a': {p.name: digest(p) for p in a.glob('*.safetensors')},
                   'parent_b': {p.name: digest(p) for p in b.glob('*.safetensors')},
                   'output': {p.name: digest(p) for p in output.glob('*.safetensors')},
                   'inference_verified': False, 'status': 'BUILT_NOT_EVALUATED'})
    except BaseException:
        write_json(output/'INCOMPLETE.json', {'usable': False})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent-a', required=True)
    p.add_argument('--parent-b', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--alpha', required=True, type=float)
    a = p.parse_args()
    merge(a.parent_a, a.parent_b, a.output, a.alpha)


if __name__ == '__main__':
    main()
