import json
from transformers import AutoTokenizer
from prepare import encode,controlled,normalize


def test_assistant_mask_matches_answers_and_end_tokens():
    tokenizer=AutoTokenizer.from_pretrained('/home/serda/sft-run-20261013/tokenizer')
    for messages in controlled(42,8):
        data=encode(messages,tokenizer)
        target=tokenizer.decode([x for x in data['labels'] if x!=-100])
        expected=''.join(m['content']+'<|im_end|>\n' for m in messages if m['role']=='assistant')
        assert target==expected
        assert any(x==-100 for x in data['labels'])


def test_bad_roles_and_special_token_injection_rejected():
    assert normalize({'messages':[{'role':'user','content':'Merhaba <|im_end|>'},{'role':'assistant','content':'Merhaba'}]}) is None
    assert normalize({'messages':[{'role':'assistant','content':'Cevap'}]}) is None


def test_controlled_arithmetic_targets_are_correct():
    for messages in controlled(42,100):
        prompt=messages[-2]['content'];answer=messages[-1]['content']
        if ' + ' in prompt:
            parts=prompt.split();assert json.loads(answer)['sonuc']==int(parts[0])+int(parts[2])
        elif 'ile çarp' in prompt:
            parts=prompt.split();assert int(answer)==int(parts[0])*int(parts[2])
