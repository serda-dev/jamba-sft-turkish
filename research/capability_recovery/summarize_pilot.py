"""Summarize actual saved GPU diagnostic reports without loading model weights."""
import argparse
import json
import math
from pathlib import Path
import statistics

from .common import digest,write_json


def summarize(precision,interference):
    if not precision.get('complete') or not interference.get('complete'):
        raise ValueError('Both real diagnostic reports must be complete')
    first=precision['gradient_comparison'][0]
    rel=[v['relative_l2_difference'] for v in first.values() if v['relative_l2_difference'] is not None]
    cosine=[v['cosine'] for v in first.values() if v['cosine'] is not None]
    total_a2=sum(v['norm_a']**2 for v in first.values())
    total_b2=sum(v['norm_b']**2 for v in first.values())
    total_dot=sum((v['cosine'] or 0)*v['norm_a']*v['norm_b'] for v in first.values())
    merged_cosine=total_dot/math.sqrt(total_a2*total_b2)
    merged_relative=math.sqrt(max(0,total_a2+total_b2-2*total_dot)/total_a2)
    arms={}
    for mode,arm in precision['arms'].items():
        steps=[]
        for step in arm['steps']:
            values=list(step['sampled_updates'].values())
            steps.append({'step':step['step'],'loss':step['loss'],'pre_clip_norm':step['pre_clip_norm'],
                'mean_sampled_master_zero_fraction':statistics.mean(v['master_zero_fraction'] for v in values),
                'mean_sampled_working_zero_fraction':statistics.mean(v['working_zero_fraction'] for v in values),
                'mean_sampled_master_changed_working_unchanged':statistics.mean(v['master_changed_but_working_unchanged_fraction'] for v in values)})
        arms[mode]={k:arm[k] for k in ['seconds','peak_allocated_gib','peak_reserved_gib','input_tokens_per_second','assistant_tokens_per_second']}
        arms[mode]['steps']=steps
    pairs={}
    for matrix in interference['matrices']:
        for pair,groups in matrix.items():
            aa=sum(v['norm_a']**2 for v in groups.values())
            bb=sum(v['norm_b']**2 for v in groups.values())
            dot=sum((v['cosine'] or 0)*v['norm_a']*v['norm_b'] for v in groups.values())
            pairs.setdefault(pair,[]).append(dot/math.sqrt(aa*bb) if aa*bb else None)
    return {'real_gpu_measurements':True,'first_step':{
        'compared_parameter_tensors':len(first),
        'concatenated_sample_gradient_cosine':merged_cosine,
        'concatenated_sample_relative_l2_difference':merged_relative,
        'per_tensor_relative_l2_median':statistics.median(rel),
        'per_tensor_relative_l2_max':max(rel),'per_tensor_cosine_min':min(cosine)},
        'arms':arms,'task_pair_cosines':{k:{'replicas':v,'mean':statistics.mean(x for x in v if x is not None)} for k,v in pairs.items()},
        'included_gradient_families':interference['included_families'],
        'missing_gradient_families':interference['missing_families'],
        'limitations':['Coordinate sampling, not exact full-gradient dot products',
            'First A/B update isolates accumulation; later weights diverge',
            'Timing includes diagnostics and no equalized warmup; first-arm cache/compilation effects not isolated',
            'No new capability benchmark or generalization measurement',
            'Diagnostic BF16 weights backed up; optimizer/master state is not resumable',
            'Negative cosine alone does not establish long-term regression causality'],
        'production_optimizer_changed':False,'pcgrad_enabled':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--precision',required=True);p.add_argument('--interference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=summarize(json.loads(Path(a.precision).read_text()),json.loads(Path(a.interference).read_text()))
    result['source_sha256']={'precision':digest(a.precision),'interference':digest(a.interference)}
    write_json(a.output,result)


if __name__=='__main__':main()
