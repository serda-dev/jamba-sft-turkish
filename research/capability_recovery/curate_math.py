"""Pre-output manual source-quality exclusions, with unchanged source answers.

No synthetic problem or answer is generated. This curation was performed before
any parent/candidate responses and must never become a score-driven filter.
"""
import argparse
import json
from pathlib import Path
import shutil

from .common import digest, rows, write_json
from .evaluation import group_split

EXCLUDE = {
    '1006': 'Unnatural mistranslated game name', '1051': 'Train capacity versus current passenger count ambiguity',
    '1069': 'Pounds/kilograms and discount mistranslation', '1084': 'Inch versus centimetre mismatch',
    '1112': 'Rises TO 3 instead of rises BY 3', '1139': 'Unintelligible translated quantities',
    '114': 'Original versus current board price ambiguity', '1170': 'Square feet versus square metres mismatch',
    '118': 'Salary relation does not support the source key',
    '1032': 'Question reverses who pays whom', '1043': 'Metre versus inch mismatch',
    '1056': 'Half more versus half as many changes the answer', '1083': 'Impossible translated fence lengths',
    '110': 'Feet versus metres mismatch', '1111': 'Three participant names change mid-question',
    '1165': 'Broken tree-age relationships', '1171': 'Duplicated mistranslated noun phrase',
}
REPLACEMENTS = {
    'dev': {'1207': '72*(11/8-1)=27', '1213': '2*5/2=5', '1223': '2*2+1=5',
            '1226': '3*12+6+4*6=66', '1233': '9/(3/4)=12', '1243': '.7*(70+40)-.6*70=35',
            '1245': '4*40*21=3360', '1271': '3*80+7*.1*80=296', '1286': '3*16+12+24=84'},
    'final': {'1097': '24-4+10=30', '1103': '3*2*4=24', '1138': '(4*10*.9)/(4*9)=1',
              '1161': '26/2-10=3', '12': '(5*2+2)*3/2=18', '1217': '(4*10/8)*45=225',
              '1218': '4*20+2*10=100', '1219': '40*4/5=32'},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise ValueError('New output directory required')
    a.output_dir.mkdir(mode=0o700)
    manifest = json.loads((a.source_dir/'manifest.json').read_text())
    pool_path = a.source_dir/'filtered_pool.json'
    pool = json.loads(pool_path.read_text())['cases']
    assignment = group_split(pool)
    by_id = {c['id']: c for c in pool}
    receipt = {'before_any_model_outputs': True, 'source_manifest_sha256': digest(a.source_dir/'manifest.json'),
               'filtered_pool_sha256': digest(pool_path), 'excluded': EXCLUDE, 'replacement_arithmetic_reviews': REPLACEMENTS,
               'scope': 'Manual pre-generation translation/arithmetic review; no independently verified AMC proofs; full generalization not established'}
    for split, spec in manifest['splits'].items():
        data = list(rows(a.source_dir/spec['file']))
        removed = [c for c in data if c['id'].split('/')[-1] in EXCLUDE and c['id'].startswith('gsm-new/')]
        data = [c for c in data if c not in removed]
        for index in REPLACEMENTS[split]:
            c = by_id['gsm-new/'+index]
            if assignment[c['id']][0] != split:
                raise ValueError('Curation cannot move source groups between splits')
            data.append({**c, 'group': assignment[c['id']][1]})
        if len(removed) != len(REPLACEMENTS[split]):
            raise ValueError('Replacement count differs from excluded cases')
        file = a.output_dir/spec['file']
        file.write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in data)); file.chmod(0o600)
        spec['sha256'] = digest(file)
    manifest['math_quality_receipt'] = 'math_curation.json'
    write_json(a.output_dir/'math_curation.json', receipt)
    write_json(a.output_dir/'manifest.json', manifest)


if __name__ == '__main__':
    main()
