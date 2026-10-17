"""Publish only the audited preparation to a private HF dataset and verify it."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path('/home/serda/sft-v3-run-20261017')
PROJECT=Path(__file__).resolve().parents[1]
REPO='serda-dev/LinguAI-Turkish-Instruction-v3-20261017'


def main():
    from huggingface_hub import HfApi
    from datasets import load_dataset
    audit=json.loads((ROOT/'data_audit.json').read_text())
    assert audit['verified'] and audit['manifest_sha256']==hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    dest=ROOT/'hub_publish';dest.mkdir(exist_ok=True)
    for name in ['data','tokenized','source/sft_v3','tokenizer']:(dest/name).mkdir(parents=True,exist_ok=True)
    for split in ['train','validation']:
        with gzip.open(ROOT/f'{split}_messages.jsonl.gz','rt') as src,gzip.open(dest/f'data/{split}.jsonl.gz','wt',encoding='utf8') as out:
            for line in src:
                r=json.loads(line)
                row={k:r[k] for k in ['messages','source','family','group','prompt_hash']}
                row['checks_json']=json.dumps(r.get('checks',[]),ensure_ascii=False)
                row['source_constraints_json']=json.dumps(r.get('source_constraints',[]),ensure_ascii=False)
                row['target_verification']=r.get('target_verification','deterministic_content_and_format_checks')
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
        shutil.copy2(ROOT/f'{split}.jsonl.gz',dest/f'tokenized/{split}.jsonl.gz')
    # Exercise the actual reader: heterogeneous rule values live inside a string.
    ds=load_dataset('json',data_files={s:str(dest/f'data/{s}.jsonl.gz') for s in ['train','validation']})
    assert {s:len(ds[s]) for s in ds}==audit['counts']
    for name in ['manifest.json','config.json','data_audit.json','excluded_old_prompts.json.gz','probes.json','v2_instruction_errors.json','tulu_source_card.md','reviewed_examples.json','review_notes.json']:
        shutil.copy2(ROOT/name,dest/name)
    for path in (ROOT/'tokenizer').iterdir():
        if path.is_file():shutil.copy2(path,dest/'tokenizer'/path.name)
    for path in (PROJECT/'sft_v3').glob('*.py'):shutil.copy2(path,dest/'source/sft_v3'/path.name)
    shutil.copy2(PROJECT/'sft_v3/PLAN_TR.md',dest/'PLAN_TR.md')
    card='''---
language:
- tr
- en
license: other
license_name: mixed-source-licenses
license_link: https://github.com/serda-dev/jamba-sft-turkish/blob/main/sft_v3/PLAN_TR.md
task_categories:
- text-generation
configs:
- config_name: messages
  default: true
  data_files:
  - split: train
    path: data/train.jsonl.gz
  - split: validation
    path: data/validation.jsonl.gz
- config_name: tokenized
  data_files:
  - split: train
    path: tokenized/train.jsonl.gz
  - split: validation
    path: tokenized/validation.jsonl.gz
---

# LinguAI Turkish instruction preparation v3

Private, audited preparation; no v3 model has been trained yet.
17,000 training and 580 development conversations. New Turkish synthetic
grounded tasks, a checked English subset, and 4,000 prior train-only replay rows.
This dataset is an intervention for instruction formatting and stopping behavior.
Development shares procedural task templates with training; it is not evidence
of broad independent capability. No answer truncation; assistant-only loss.

Attribution: Allen Institute for AI, Tulu 3 SFT Personas Instruction Following,
https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following,
revision fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e, ODC-BY-1.0.
Only single advertised constraints supported by local checkers are accepted.
Factual accuracy and all implicit prompt requirements are not proven.

Replay preserves original MIT / Apache-2.0 source notices and pinned revisions
in manifest.json, including v1 and v2 parents. New local generator is Apache-2.0;
the whole mixture is not relicensed under that one license.
See PLAN_TR.md for steps, exclusions, training settings, budget and acceptance.
checks_json and source_constraints_json preserve rule metadata as JSON text
for stable reader schemas, including rows without external constraints.
'''
    (dest/'README.md').write_text(card)
    api=HfApi();api.create_repo(REPO,repo_type='dataset',private=True,exist_ok=True)
    assert api.dataset_info(REPO).private
    commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(dest),commit_message='Audited instruction v3 preparation and complete training plan')
    remote={f.rfilename:f for f in api.dataset_info(REPO,revision=commit.oid,files_metadata=True).siblings}
    hashes={}
    for path in dest.rglob('*'):
        if not path.is_file():continue
        name=str(path.relative_to(dest));data=path.read_bytes();item=remote[name]
        assert item.size==len(data)
        expected=hashlib.sha256(data).hexdigest() if item.lfs else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert (item.lfs.sha256 if item.lfs else item.blob_id)==expected
        hashes[name]=hashlib.sha256(data).hexdigest()
    receipt={'repo_id':REPO,'repo_type':'dataset','revision':commit.oid,'private':True,'verified':True,'files':hashes}
    (ROOT/'dataset_hub_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))


if __name__=='__main__':main()
