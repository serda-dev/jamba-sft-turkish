"""Publish a reproducible private dataset; verify remote parquet and metadata hashes."""
import gzip,hashlib,json
from pathlib import Path
ROOT=Path('/home/serda/sft-v4-run-20261018');REPO='serda-dev/turkish-grounded-instructions-v4-20261018'

def main():
 import pyarrow as pa
 import pyarrow.parquet as pq
 from huggingface_hub import HfApi
 folder=ROOT/'dataset_publish';(folder/'data').mkdir(parents=True,exist_ok=True)
 for split in ['train','validation']:
  with gzip.open(ROOT/f'{split}_messages.jsonl.gz','rt') as f:original=[json.loads(line) for line in f]
  rows=[]
  for r in original:
   rows.append({'id':r['id'],'group':r['group'],'source':r['source'],'component':r['component'],'family':r['family'],'messages':r['messages'],'task_metadata':json.dumps({k:v for k,v in r.items() if k not in ['id','group','source','component','family','messages']},ensure_ascii=False)})
  path=folder/'data'/f'{split}.parquet';pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd');assert pq.read_table(path).to_pylist()==rows
 (folder/'README.md').write_text('''---
language:
- tr
- en
license: other
license_name: mixed-source-mit-apache-2.0
license_link: https://huggingface.co/datasets/serda-dev/turkish-grounded-instructions-v4-20261018/blob/main/LICENSES.md
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
---
# Grounded Turkish instruction SFT v4

Private training candidate, 3,082 train / 128 development examples. 2,000 source-grounded editing/extraction/formatting tasks, 717 verified arithmetic tasks, 365 natural retention replay examples. Target-token proportions approximately 70/15/15. No newly generated paid Teacher answers.

The instruction tasks include the reference text in the prompt. All formatting and source-content preservation targets are programmatically verified. Source factual truth and open-ended answer quality are not independently certified. Development source questions are excluded from earlier V1/V2 exact prompt identities; original question/answer groups are disjoint from current training. Exact/13-word frozen benchmark exclusions are applied, without claiming semantic leakage is impossible.

`messages` contains the training conversation; `task_metadata` records checker arguments, provenance and the expected answer where applicable. Tasks share format families across splits; this is development data, not a general instruction benchmark.

Source licenses and attribution: LICENSES.md. Full procedure/code: https://github.com/serda-dev/jamba-sft-turkish/blob/main/sft_v4/PLAN_TR.md
''')
 (folder/'LICENSES.md').write_text('''# Source licenses and attribution

Source data retains its original licenses. This mixture does not relicense upstream material.

- AlicanKiraz0/Turkce-Atlas-Instruct — MIT, revision c387738c5deacbb35667ab4057930adc34e647e8: https://huggingface.co/datasets/AlicanKiraz0/Turkce-Atlas-Instruct
- CohereLabs/aya_dataset — Apache-2.0, revision f9ea04583f02a8f86404ff6c58bf75fe637df8a2: https://huggingface.co/datasets/CohereLabs/aya_dataset
- Retention replay inherits V2 training sources, with per-row upstream names: AlicanKiraz0/Turkce-Atlas-Instruct (MIT), CohereLabs/aya_dataset (Apache-2.0), merve/turkish_instructions (Apache-2.0), HuggingFaceH4/ultrachat_200k (MIT).
- New deterministic task construction and verified arithmetic: LinguAI project, source code/provenance in the linked training repository.

MIT: https://opensource.org/license/mit
Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
''')
 for name in ['manifest.json','data_audit.json']:(folder/name).write_bytes((ROOT/name).read_bytes())
 api=HfApi();api.create_repo(REPO,repo_type='dataset',private=True,exist_ok=True);assert api.dataset_info(REPO).private
 commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(folder),commit_message='Verified grounded instruction v4 training and grouped development data')
 files={x.rfilename:x for x in api.dataset_info(REPO,revision=commit.oid,files_metadata=True).siblings};verified={}
 for p in folder.rglob('*'):
  if not p.is_file():continue
  name=p.relative_to(folder).as_posix();b=p.read_bytes();f=files[name];assert f.size==len(b)
  expected=hashlib.sha256(b).hexdigest() if f.lfs else hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
  actual=(f.lfs.sha256 if hasattr(f.lfs,'sha256') else f.lfs['sha256']) if f.lfs else f.blob_id
  assert actual==expected;verified[name]={'size':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 receipt={'repo':REPO,'revision':commit.oid,'private':True,'verified':True,'files':verified};(ROOT/'dataset_hub_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'repo':REPO,'revision':commit.oid,'verified':True,'files':len(verified)}),flush=True)
if __name__=='__main__':main()
