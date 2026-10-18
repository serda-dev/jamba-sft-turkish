"""Keep frozen grounded checks; add natural multi-step math scoring."""
import re
try:from grounded_data import verify as grounded_verify
except ModuleNotFoundError:from sft_v4.data import verify as grounded_verify
try:from benchmark_code.numeric_score import numeric
except ModuleNotFoundError:from benchmarks.numeric_score import numeric

def verify(row,response):
 if row['family']!='natural_math':return grounded_verify(row,response)
 values=re.findall(r'-?\d+(?:[.,]\d+)*',response.rsplit('####',1)[-1] if '####' in response else response)
 correct=bool(values and numeric(values[-1])==numeric(row['gold']))
 return {'format':bool(re.search(r'####\s*-?[\d.,]+\s*$',response)),'content':correct,'correct':correct}
