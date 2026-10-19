# Görev gradyanları ve çatışma araştırması

Henüz gerçek Jamba task-gradient sonucu yok. IFEval/math tradeoffu negatif cosine ölçümü değildir.

`gpu_experiments interference` sabit checkpointte dropout kapalı ve autograd açık çalışır; optimizer adımı yapmaz. Math, reasoning, instruction, reading, dialogue, code ve English retention için üç ayrı grup önerilir. Her görevden sonra gradient temizlenir. Full normlar katman grubuyla toplanır; cosine için tensor başına en fazla256 deterministik koordinat CPUya taşınır. Böylece yedi tam FP32 gradient vektörü VRAMde tutulmaz. Sonuç örneklenen altuzaya aittir; tüm parametre uzayının tarafsız cosine tahmini değildir.

Gruplar: Mamba, attention, feed-forward, embeddings, output head, diğer. Tied output head embedding altında tek kez görülebilir; boş head grubu ölçülmüş gibi doldurulmaz. Mimari isim eşlemesi rapordaki tensor adlarıyla denetlenmeli.

Hazırlanan diagnostik dilimler eski V5 train örneklerinden gelir. Bunlar yeni bağımsız performans testi değildir. Doğal Türkçe çok turlu `general` dilimi bulunamazsa diyalog eksik bırakılır; eski15 proje-kodu hatırlama örneği doğal konuşma yerine konmaz. `--families` ile altı aileli kısmi test açıkça çalıştırılabilir; eksik aile sonuçta yazılır. İngilizceyi yalnızca genel diyalog saymak yerine math/reading/general kaynakları ayrı izlenir; English code hâlâ eksiktir. Çeviri çiftlerinin semantik bağımsızlığı ayrıca doğrulanmalı.

## PCGrad kararı

[interference.py](interference.py) test edilmiş küçük-vector PCGrad referansı içerir. Aktif training algoritması değildir. Tek negatif cosine, uzun vadeli gerilemenin nedeni sayılmaz. Üç ayrı kaynak-grupta tekrarlanan, büyüklüğü anlamlı çatışmalar ve gerçek dev kaybıyla tutarlı işaret varsa ordinary averaginge karşı aynı veri/seed/optimizer-adımı ve **ayrı wall-clock maliyeti** raporlanan kısa pilot düşünülür. Fazladan backwardların maliyeti gizlenmez; aynı optimizer-step sayısı eşit maliyet demek değildir. GPU kanıtı olmadan ek optimizer platformu kurulmadı.
