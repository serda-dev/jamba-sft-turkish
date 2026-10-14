"""Check exact normalized SFT user turns against test prompts and bare questions."""
import gzip
import json
import re
import unicodedata
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path('/home/serda/benchmark-run-20261014')


def norm(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFC',text)).strip().casefold()


def main():
    tok=Tokenizer.from_file('/home/serda/sft-run-20261013/tokenizer/tokenizer.json')
    prompts=set()
    for name in ['train','validation']:
        with gzip.open(f'/home/serda/sft-run-20261013/{name}.jsonl.gz','rt') as f:
            for line in f:
                text=tok.decode(json.loads(line)['input_ids'],skip_special_tokens=False)
                prompts.update(norm(p) for p in re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',text,re.S))
    rows=[json.loads(x) for x in (ROOT/'cases.jsonl').read_text().splitlines()]
    matches=[]
    for row in rows:
        if row['task']=='gsm8k_tr_cot8':continue
        text=row['prompt'];variants={'full_prompt':text}
        if row['task']=='gsm8k_tr':variants['bare_question']=text.split('\n\nKısa çözümünü',1)[0]
        elif row['type']=='mcq':
            payload=text.split('\n\n',1)[1]
            variants['question_and_choices']=payload
            variants['without_choices']=payload.split('\nA. ',1)[0]
            if row['task']=='belebele_tr':variants['bare_question']=variants['without_choices'].rsplit('\n\n',1)[-1]
        for form,value in variants.items():
            if norm(value) in prompts:matches.append({'id':row['id'],'form':form})
    report={'normalized_training_prompt_count':len(prompts),'benchmark_count':len(rows),
            'unique_question_count':len(rows)-200,'exact_matches':matches,
            'normalization':'NFC, whitespace collapse, casefold; exact user turns against full prompts, bare math questions, MCQ question/choices and Belebele questions',
            'scope':'Exact SFT overlap only; pretraining and semantic/paraphrase overlap unknown; cot8 repeats the same 200 math questions'}
    (ROOT/'overlap_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
