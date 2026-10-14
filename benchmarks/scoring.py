"""Pinned upstream IFEval checks, restricted to auditable Turkish-safe rules."""
import ast
from instruction_following_eval import instructions_registry
from instruction_following_eval.instructions import Instruction
from langdetect import detect,LangDetectException,DetectorFactory
DetectorFactory.seed=0


class TurkishCase(Instruction):
    def get_instruction_args_keys(self):return []
    def build_description(self):return 'Turkish text with the requested letter case'
    def check_following(self,value):
        case_ok=value.isupper() if self.id=='change_case:turkish_capital' else value.islower()
        try:return case_ok and detect(value)=='tr'
        except LangDetectException:return False

ALIASES={'change_case:turkish_capital':'change_case:english_capital',
         'change_case:turkish_lowercase':'change_case:english_lowercase'}
SUPPORTED={'punctuation:no_comma','detectable_format:json_format','detectable_format:title',
 'detectable_format:number_bullet_lists','detectable_format:number_highlighted_sections',
 'detectable_format:multiple_sections','detectable_content:number_placeholders',
 'length_constraints:number_words','length_constraints:number_paragraphs',
 'keywords:existence','keywords:frequency','keywords:forbidden_words',
 'startend:end_checker','startend:quotation','combination:two_responses',
 'change_case:turkish_capital','change_case:turkish_lowercase'}


def checks(row):
    ids=row['instruction_id_list'];kwargs=row['kwargs']
    if isinstance(ids,str):ids=ast.literal_eval(ids)
    if isinstance(kwargs,str):kwargs=ast.literal_eval(kwargs)
    if len(ids)!=len(kwargs):raise ValueError('Misaligned constraints')
    result=[]
    for ident,values in zip(ids,kwargs):
        if ident not in SUPPORTED:raise ValueError('Unsupported Turkish rule: '+ident)
        mapped=ALIASES.get(ident,ident)
        instance=TurkishCase(ident) if ident in ALIASES else instructions_registry.INSTRUCTION_DICT[mapped](ident)
        keys=instance.get_instruction_args_keys() or []
        if any(values.get(key) is None for key in keys):raise ValueError('Missing deterministic arguments')
        selected={key:values[key] for key in keys}
        instance.build_description(**selected)
        result.append(instance)
    return result


def score(row,response):
    return [bool(response.strip()) and bool(rule.check_following(response)) for rule in checks(row)]
