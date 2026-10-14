"""Freeze common test cases before renting; never train on these fixtures."""
import hashlib
import json
import random
from pathlib import Path
from collections import Counter
from huggingface_hub import HfApi
from datasets import load_dataset
from scoring import checks

ROOT=Path('/home/serda/benchmark-run-20261014')
SOURCES=[
 ('turkish_mmlu','AYueksel/TurkishMMLU','All','test','fcbe5bb19af2714186fdd5752671e0748efa0024',None),
 ('belebele_tr','facebook/belebele','tur_Latn','test','7899cdfa4e1e0d733fd77c848e2c273cb1d32be2',None),
 ('gsm8k_tr','malhajar/gsm8k-tr','main','test','1046702fc5d8e14330ae90fdfd98735194ffa4ce',200),
 ('ifeval_tr_subset','bezir/IFEval_tr',None,'train','07e8c2cf4ae66a86fba57d389a73cbc10e865d5d',120)]


def main():
    api=HfApi();rng=random.Random(20261014);rows=[];audit=[]
    for task,repo,config,split,revision,limit in SOURCES:
        ds=load_dataset(repo,config,split=split,revision=revision)
        candidates=[];skipped=Counter()
        for index,row in enumerate(ds):
            if task=='ifeval_tr_subset':
                try:checks(row)
                except (ValueError,KeyError,TypeError) as error:
                    skipped[str(error)]+=1;continue
            if task=='turkish_mmlu':
                prompt='Aşağıdaki sorunun doğru seçeneğini belirle. Yalnızca seçeneğin harfini yaz.\n\n'+row['question']+'\n'+ '\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(row['choices']))
                item={'type':'mcq','prompt':prompt,'choices':len(row['choices']),'gold':row['answer'],'subject':row['subject']}
            elif task=='belebele_tr':
                prompt='Metne dayanarak soruyu yanıtla. Yalnızca doğru seçeneğin harfini yaz.\n\n'+row['flores_passage']+'\n\n'+row['question']+'\n'+'\n'.join(f'{chr(65+i)}. {row[f"mc_answer{i+1}"]}' for i in range(4))
                item={'type':'mcq','prompt':prompt,'choices':4,'gold':int(row['correct_answer_num'])-1}
            elif task=='gsm8k_tr':
                item={'type':'math','prompt':row['question']+'\n\nKısa çözümünü yaz. Son satırda yalnızca #### ve sayısal sonucu yaz.',
                      'gold':row['answer'].split('####')[-1].strip().replace(',','')}
            else:item={'type':'instruction','prompt':row['prompt'],'rules':row}
            item.update(task=task,id=f'{task}/{index}',source_row=index)
            candidates.append(item)
        if limit and len(candidates)>limit:candidates=rng.sample(candidates,limit)
        rows.extend(candidates)
        audit.append({'task':task,'repo':repo,'revision':revision,'config':config,'split':split,
                      'available':len(ds),'selected':len(candidates),'skipped':dict(skipped)})
    models=[
      {'id':'jamba_sft','repo':'serda-dev/Jamba2-3B-Turkish-SFT-20261013','revision':'0b22f86c21610f620826067f573aa5ef2821e9cf','subfolder':'checkpoints/final'},
      {'id':'qwen25_3b','repo':'Qwen/Qwen2.5-3B-Instruct','revision':'aa8e72537993ba99e69dfaafa59ed015b17504d1'},
      {'id':'qwen3_4b_2507','repo':'Qwen/Qwen3-4B-Instruct-2507','revision':'cdbee75f17c01a7cc42f958dc650907174af0554'},
      {'id':'jamba_cpt','repo':'serda-dev/Jamba2-3B-Turkish-CPT-1B-20261007','revision':'8b1a9c812791fa7930e64b9444b41519c023269e','subfolder':'checkpoints/final'},
      {'id':'jamba_original','repo':'ai21labs/AI21-Jamba2-3B','revision':'525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9'}]
    fixture=ROOT/'cases.jsonl'
    fixture.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest={'seed':20261014,'sources':audit,'models':models,'case_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
      'scorer_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('instruction_following_eval').glob('*.py')},
      'protocol':'0-shot choice-letter likelihood; greedy generation; BF16 Transformers SDPA; common fixtures',
      'gsm8k_new_tokens':512,'ifeval_new_tokens':2048,'generation_batch':4,
      'ifeval_notice':'Supported deterministic Turkish-safe subset, not the full published leaderboard metric.',
      'mmlu_notice':'Public AYueksel subset; not the complete 10K paper dataset or a reproduced 5-shot leaderboard.',
      'speed_notice':'Native BF16 Transformers on one GPU; excludes download/compile warmup; tokens and characters per second reported.'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'counts':dict(Counter(r['task'] for r in rows)),'audit':audit},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
