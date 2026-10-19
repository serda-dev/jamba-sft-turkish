# Kök neden hipotezleri

## Kanıt ve ayrıştırıcı deney

| Hipotez | Eldeki kanıt | Bilmediğimiz | En küçük anlamlı deney |
|---|---|---|---|
| Kaybın bir bölümü CPT'den devralındı | Düzeltilmiş original→CPT okuma441→259, IFEval56→16 | CPT veri/optimizer/tokenizer bileşenlerinin ayrı etkisi | Original ve CPTyi diagnostik referans tut; vocab farklı olduğundan doğrudan merge yapma |
| Küçük güncellemeler BF16 parametrede kayboluyor | Önceki GPU tanısı ve FP32 master düzeltmesi | Tarihsel toplam kaybın ne kadarı bundan | Yeni large SFT gerekmez; önceki kanıtı koru |
| BF16 mikro gradyan toplamı hatası | Runtime ve taşınan gerçek `precision.py` incelendi; CPU kontrollü1,0→1,030983 farkı | Gerçek Jamba gradient/update/kalite etkisi | Aynı128 örnek,4×32 mikro, LR7.5e-7; mevcut/FP32 mikro toplam A/B |
| Görevler çatışan yönlerde güncelleme yapıyor | IFEval artarken math düşen koşular | Gradient cosine ölçülmedi; negatif cosine nedensellik değildir | Sabit V2/V5, görev başına3 grup; katman grubu örneklenmiş cosine ve tam norm |
| Veri yeterince çeşitli değil | V4 dar biçim görevleri; kolay math19→63 ama dış math73→43 | Daha çeşitli verinin tek başına çözümü | Önce yeni doğal dev ve merge; ardından aynı-bütçeli veri ablationı |
| Hedef token/mikro-batch dengesizliği | V5 ham reading%1,17; weighted hedef%15; gerçekleşen katsayı%13,20 | Gerçek gradient katkısı | Token sampler + exposure receipt; küçük eşit-verili pilot yalnız gerekirse |
| EOS/repetition alışkanlığı ayrı öğrenilmiş sorun | EOS düzeltmesine rağmen V5 kesilme33/120 | Kök neden tek başına veri/optimizer/cache mi | Kısa ve uzun doğru yanıtları birlikte ölç; sonlandırma veri denetimi |
| Dev ölçümü yanlış vekil | Prosedürel/doymuş kontroller, konuşma NLL kapısı | Gerçek yeni doğal görev sonuçları | Grupları ayrılmış değerlendirme; format ve anlamı ayrı puanla |
| V2/V5 yetenekleri merge ile korunabilir | Aynı isim/şekil/vocab; gerçek koordinat probları finite | Ara ağırlık havzası ve genelleme | V2,V5 ve4 alpha, aynı yeni dev; fayda yoksa karmaşık merge yok |

## Birincil araştırma kaynakları

- [Model soups, ICML2022](https://proceedings.mlr.press/v162/wortsman22a.html): ağırlık ortalamanın belirli fine-tune ailelerindeki deneysel dayanağı. Jamba'ya aktarımı hipotezdir; garanti değildir.
- [Gradient Surgery / PCGrad](https://arxiv.org/abs/2001.06782): çakışan gradyanları projekte eden yöntem. Burada yalnız küçük-vector referansı var; üretimde etkin değil.
- [PyTorch autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html): leaf `.grad` birikimi. Yeni yol, her backwarddan sonra gradyanı FP32 buffera taşıyıp working `.grad`ı temizler; forward BF16 kalır.

Bu kaynaklar yöntem seçimini destekler; bizim modelimizde ölçülmemiş sonucu kanıtlamaz. Tek seed ve küçük örnekle kesin nedensellik/üstünlük iddiası yapılmaz.
