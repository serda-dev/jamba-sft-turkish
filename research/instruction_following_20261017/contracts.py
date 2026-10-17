"""Explicit output contracts for the next pilot; archived v3 remains untouched."""
import json
import re


def validate(contracts):
    seen={}
    for c in contracts:
        kind=c['kind']
        if kind not in {'paragraphs','keyword_lines','two_answers','ending','json_exact','exact','required'}:
            raise ValueError('Unsupported contract: '+kind)
        if kind in {'paragraphs','keyword_lines'}:
            if type(c['count']) is not int or c['count']<1:raise ValueError('Positive count required')
        if kind=='paragraphs':
            if c['separator'] not in ['\n\n','***']:raise ValueError('Unsupported separator')
            key=(kind,c['separator'])
            if key in seen and seen[key]!=c['count']:raise ValueError('Conflicting paragraph counts')
            seen[key]=c['count']
        if kind=='two_answers' and not c['separator']:raise ValueError('Empty answer separator')


def check(contracts,response):
    validate(contracts)
    if not response.strip():return False
    for c in contracts:
        kind=c['kind']
        if kind=='paragraphs':
            parts=re.split(r'\n\s*\n',response.strip()) if c['separator']=='\n\n' else response.strip().split(c['separator'])
            ok=len(parts)==c['count'] and all(p.strip() for p in parts)
        elif kind=='keyword_lines':
            lines=response.strip().splitlines();pattern=r'(?<!\w)'+re.escape(c['keyword'])+r'(?!\w)'
            ok=len(lines)==c['count'] and all(len(re.findall(pattern,line))==1 for line in lines)
        elif kind=='two_answers':
            parts=response.strip().split(c['separator'])
            ok=len(parts)==2 and all(p.strip() for p in parts) and parts[0].strip()!=parts[1].strip()
            if ok and c.get('each_required'):
                ok=all(all(re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)',part) for value in c['each_required']) for part in parts)
        elif kind=='required':ok=all(re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)',response) for value in c['values'])
        elif kind=='ending':ok=response.rstrip().endswith(c['value'])
        elif kind=='exact':ok=response==c['value']
        elif kind=='json_exact':
            try:ok=json.dumps(json.loads(response),sort_keys=True,ensure_ascii=False)==json.dumps(c['value'],sort_keys=True,ensure_ascii=False)
            except (ValueError,TypeError):ok=False
        if not ok:return False
    return True
