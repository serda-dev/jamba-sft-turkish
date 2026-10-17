"""Prepare a bounded diagnostic, not a new production training corpus."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prepare import encode
from research.instruction_following_20261017.contracts import check

ROOT=Path('/home/serda/instruction-pilot-20261017')
PROJECT=Path(__file__).resolve().parent
TOPICS=[('kütüphane','kitap'),('bahçe','fidan'),('müze','eser'),('okul','defter'),('atölye','ürün'),('tiyatro','bilet'),('arşiv','belge'),('spor merkezi','top')]


def cases(count,split):
    result=[]
    for i in range(count):
        mode=i%8;name=['Ada','Ece','Ozan','Deniz'][(i+i//8)%4];site,item=TOPICS[(i//8)%8]
        code=('P' if split=='train' else 'V')+str(81000+i);a=23+i;b=4+i%5
        facts=f'{name} {site} sorumlusudur. İşlem kodu {code}. Mevcut stok {a} {item}. Yeni gelen miktar {b} {item}.'
        required={'kind':'required','values':[name,code]}
        if mode in [0,1]:
            separator='***' if mode==0 else '\n\n';label='***' if mode==0 else 'bir boş satır'
            prompt=f'Kaydı tam iki kısa paragrafla özetle. Paragrafları {label} ile ayır. İlk paragrafta sorumlu ile kodu belirt; ikinci paragrafta mevcut stok ve yeni miktarı yaz. Başka başlık ekleme.'
            answer=f'Sorumlu {name}. İşlem kodu {code}.'+('\n***\n' if mode==0 else '\n\n')+f'Mevcut stok {a} {item}. Yeni gelen miktar {b} {item}.'
            rules=[{'kind':'paragraphs','count':2,'separator':separator},required,{'kind':'required','values':[str(a),str(b)]}]
        elif mode==2:
            prompt='İki satır yaz. Her satırda "kayıt" kelimesi tam bir kez bulunsun. İlk satır kodu, ikinci satır sorumluyu içersin.'
            answer=f'{code} kodlu kayıt hazır.\n{name} kayıt sorumlusudur.';rules=[{'kind':'keyword_lines','count':2,'keyword':'kayıt'},required]
        elif mode==3:
            prompt='Aynı bilgiyi iki farklı kısa yanıtla aktar. İki yanıtı yalnızca ****** ile ayır. Her iki yanıtta kod bulunsun.'
            answer=f'{code} kodlu işlemin sorumlusu {name}.\n******\n{name} {code} işlemini yönetiyor.';rules=[{'kind':'two_answers','separator':'******','each_required':[code]},required]
        elif mode==4:
            prompt='Sorumlu ile kodu belirt. Yanıtın son ifadesi aynen "Kayıt kapandı." olsun.'
            answer=f'{name} {code} işleminin sorumlusudur. Kayıt kapandı.';rules=[required,{'kind':'ending','value':'Kayıt kapandı.'}]
        elif mode==5:
            value={'kod':code,'toplam':a+b};prompt='Mevcut stok ve yeni miktarı topla. Sadece kod ve toplam alanlarıyla JSON ver. Toplam bir tam sayı olsun. Kod bloğu ve açıklama ekleme.'
            answer=json.dumps(value,ensure_ascii=False);rules=[{'kind':'json_exact','value':value}]
        else:
            text=f'İzmir ve Isparta için kayıt kodu {code}.'
            prompt=('Bütün harfleri Türkçe kurallarına göre BÜYÜK yap' if mode==6 else 'Bütün harfleri Türkçe kurallarına göre küçük yap')+f'. Yalnızca dönüştürülmüş metni ver: {text}'
            answer=text.translate(str.maketrans('iı','İI')).upper() if mode==6 else text.translate(str.maketrans('İI','iı')).lower();rules=[{'kind':'exact','value':answer}]
        prefix='Aşağıdaki kaydı kullan.\n' if split=='train' else 'Verilen kayıt bilgileri:\n'
        messages=[{'role':'user','content':prefix+facts+'\n\n'+prompt},{'role':'assistant','content':answer}]
        row={'id':split+'/'+str(i),'family':['paragraphs_stars','paragraphs_blank','keyword_lines','two_answers','ending','json','uppercase','lowercase'][mode],'messages':messages,'contracts':rules,'expected':answer}
        assert check(rules,answer)
        result.append(row)
    return result


def main():
    from transformers import AutoTokenizer
    ROOT.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained('/home/serda/sft-v2-run-20261015/tokenizer');tok.save_pretrained(ROOT/'tokenizer')
    train=cases(32,'train');dev=cases(128,'dev')
    assert not ({r['messages'][0]['content'] for r in train}&{r['messages'][0]['content'] for r in dev})
    encoded=[]
    for row in train:
        e=encode(row['messages'],tok);assert len(e['input_ids'])<=384
        assert tok.decode([v for v in e['labels'] if v!=-100],skip_special_tokens=False)==row['expected']+'<|im_end|>\n'
        assert len(tok(row['expected'],add_special_tokens=False)['input_ids'])<100
        encoded.append(e)
    (ROOT/'cases.json').write_text(json.dumps({'train':train,'dev':dev},ensure_ascii=False,indent=2))
    (ROOT/'encoded.json').write_text(json.dumps(encoded))
    shutil.copy2('/home/serda/sft-v2-run-20261015/final_receipt.json',ROOT/'base_receipt.json')
    cfg={'base_repo':'serda-dev/Jamba2-3B-Turkish-SFT-v2-20261015','base_revision':'c8ee3888c486fa49d8d9e798a632c9f7ce664d7a','base_subfolder':'checkpoints/final','output_repo':'serda-dev/Jamba2-3B-Turkish-Instruction-Diagnostic-20261017','max_hours':1,'max_run_cost_usd':2,'stop_credit':14,'hard_stop_credit':12,'rental_ceiling_usd_per_hour':1.35,'seed':20261017,'micro_batch':1,'gradient_accumulation':4,'optimizer_steps':100,'max_new_tokens':128,'train_success_threshold':.90,'arms':[{'name':'bf16_lr2e6','storage_dtype':'bfloat16','learning_rate':2e-6},{'name':'fp32_lr2e6','storage_dtype':'float32','learning_rate':2e-6},{'name':'fp32_lr1e5','storage_dtype':'float32','learning_rate':1e-5}],'note':'Scratch weights discarded; all arms start independently from the same v2 snapshot. Same step count, examples and seed. FP32 storage uses BF16 autocast and may change intermediate numerical precision as well as updates.'}
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2))
    manifest={'files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['cases.json','encoded.json','base_receipt.json']},'train':32,'dev':128,'families':8,'all_targets_verified':True,'assistant_labels_checked':True,'note':'Simple task templates share between train/dev. This diagnoses memorization and production paths, not broad instruction generalization.'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
