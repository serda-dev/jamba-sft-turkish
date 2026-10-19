"""Standalone research figures from actual pilot summary; no inferred task accuracy."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    data=json.loads(Path(a.summary).read_text());output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    families=data['included_gradient_families']
    names={'math':'Matematik','reasoning':'Genel reasoning','instruction':'Yönerge',
           'reading':'Okuma','code':'Kod','english_retention':'İngilizce korunum'}
    matrix=np.full((len(families),len(families)),np.nan)
    for key,value in data['task_pair_cosines'].items():
        x,y=key.split(':');i,j=families.index(x),families.index(y);matrix[i,j]=matrix[j,i]=value['mean']
    fig,ax=plt.subplots(figsize=(8,7))
    cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#eeeeee')
    plot=ax.imshow(matrix,cmap=cmap,vmin=-.25,vmax=.25)
    ax.set_xticks(range(len(families)),[names[x] for x in families],rotation=40,ha='right')
    ax.set_yticks(range(len(families)),[names[x] for x in families])
    for i in range(len(families)):
        for j in range(len(families)):
            if i!=j:ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center')
    ax.set_title('V2 görev gradyanları: örneklenmiş cosine\n3 kaynak grubunun ortalaması')
    fig.colorbar(plot,ax=ax,label='Örneklenmiş gradient cosine')
    fig.text(.5,.01,'Tam-model dot product veya uzun vadeli gerileme kanıtı değildir.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.025,1,1));fig.savefig(output/'gradient_cosine.svg');plt.close(fig)
    modes=['existing','fp32_micro'];labels=['Mevcut BF16 birikim','FP32 mikro birikim']
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,field,title in [(axes[0],'seconds','4 update süresi (s)'),(axes[1],'peak_allocated_gib','Tepe GPU belleği (GiB)')]:
        vals=[data['arms'][m][field] for m in modes]
        bars=ax.bar(labels,vals,color=['#547da8','#d3874a'])
        ax.bar_label(bars,fmt='%.1f',padding=3);ax.set_ylim(0,max(vals)*1.2);ax.set_title(title)
    fig.suptitle('A100 SXM4 80 GB — gerçek tanı ölçümü')
    fig.text(.5,.015,'Süre tanı kontrollerini içerir; eşit warmup uygulanmadı. Genel yetenek testi değildir.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(output/'precision_resources.svg');plt.close(fig)


if __name__=='__main__':main()
