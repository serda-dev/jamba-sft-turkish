"""Score prescribed content exactly; verify open prose constraints structurally."""
import json,re

def check(row,response):
 response=response.strip();expected=row['expected'];q=row['messages'][-1]['content'];family=row['family']
 if family=='math':
  actual=re.findall(r'####\s*(-?\d+)',response);gold=re.findall(r'####\s*(-?\d+)',expected)
  return bool(actual and gold and actual[-1]==gold[-1])
 if family=='reading':return response.upper()==expected
 try:return json.loads(response)==json.loads(expected)
 except (ValueError,TypeError):pass
 if 'tam iki cümle' in q:
  name=re.search(r'İlk cümlede (\w+)',q)[1];parts=[p.strip() for p in re.split(r'[.!?]+',response) if p.strip()]
  return len(parts)==2 and name in parts[0] and 'katılım' in parts[1]
 if 'sözcükten oluşan' in q:
  k=int(re.search(r'tam (\d+) sözcükten',q)[1]);return len(response.split())==k and '\n' not in response and not re.search(r'[^\w\s]',response)
 if 'kısa paragraf' in q:
  target=expected.split('\n\n');actual=[p for p in response.split('\n\n') if p.strip()]
  if len(target)!=len(actual):return False
  for a,b in zip(actual,target):
   name=b.split()[0];num=re.search(r'\d+',b)[0]
   if name not in a or num not in a:return False
  return True
 if 'karşılama yaz' in q and 'Her satırda' in q:
  k=int(re.search(r'(\d+) satırlık',q)[1]);lines=response.splitlines();return len(lines)==k and all(len(re.findall(r'\bmerhaba\b',line))==1 for line in lines)
 if 'altı yıldızla' in q:
  parts=response.split('******');name=re.search(r'(\w+) için kısa',q)[1];return len(parts)==2 and all(name in p and p.strip() for p in parts)
 if 'Yanıtın tam olarak' in q:
  parity='çift' if 'çift sayıdır' in expected else 'tek';num=re.search(r'\d+',expected)[0]
  return response.startswith('Başlangıç') and response.endswith('Bitti.') and parity in response and num in response
 if 'Yanıtın tamamını çift tırnak' in q:
  return response.startswith('"') and response.endswith('"') and ',' not in response and len(re.findall(r'[.!?]',response))==1 and re.search(r'(\w+) için',q)[1] in response
 return response==expected
