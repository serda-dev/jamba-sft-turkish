# V2/V5 gerçek merge deneyi — sonuçlar

Altı model aynı yeni development setinde, aynı greedy protokol ve örnek başına aynı çıktı bütçesiyle değerlendirildi. Tarihî benchmark checkpoint seçiminde kullanılmadı.

| Model | Matematik /64 | GSM-TR /32 | AMC-TR /32 | Yönerge /80 | Çoklu kısıt /64 | Okuma /64 | Konuşma belleği /64 | Cap /272 | EOS /272 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 | 9 | 8 | 1 | 16 | 7 | 8 | 8 | 49 | 223 |
| v5 | 13 | 11 | 2 | 18 | 9 | 11 | 8 | 38 | 234 |
| alpha_0.15 | 11 | 11 | 0 | 18 | 9 | 8 | 8 | 47 | 225 |
| alpha_0.30 | 12 | 11 | 1 | 17 | 9 | 11 | 8 | 38 | 234 |
| alpha_0.50 | 9 | 9 | 0 | 17 | 8 | 10 | 8 | 41 | 231 |
| alpha_0.70 | 12 | 10 | 2 | 18 | 9 | 11 | 8 | 41 | 231 |

## Önceden belirlenen seçim kapısı

| Aday | İki ebeveyne karşı tüm ailelerde nonregression | İki ebeveyne karşı pozitif paired CI bulunan aileler |
|---|---|---|
| alpha_0.15 | Geçmedi | Yok |
| alpha_0.30 | Geçmedi | Yok |
| alpha_0.50 | Geçmedi | Yok |
| alpha_0.70 | Geçti | Yok |

**Karar:** no_demonstrated_superior_merge.

Üstünlük gösterilememesi bütün interpolasyonların her görevde kötü olduğu anlamına gelmez. Görev bazındaki farklar ve paired kaynak-grubu bootstrap aralıkları makine raporunda korunur. Dört aday/örtüşen ailelerde çoklu karşılaştırma olduğundan keşifsel CI sonuçları düzeltilmiş istatistiksel anlamlılık diye sunulmaz.

## Ölçümün sınırları

- Yönerge: beyan edilen kısıtların uyumu; olgusal/konusal yanıt doğruluğu değil.
- Matematik: kaynak anahtarına göre sayısal son yanıt. Ara muhakemenin geçerliliği bağımsız doğrulanmadı.
- Okuma: kısa kaynak ifadesinin exact matchi; eş anlamlı uzun yanıtlar false negative olabilir.
- Konuşma: gerçek Türkçe insan diyaloglarında bilgi hatırlama; serbest sohbet akıcılığı puanı değil.
- Sonlandırma: cap/EOS/tekrar/uzunluk ve doğru yanıtla birlikte EOS ayrı tutulur. Zorla kısa cevap genel başarı yerine geçmez.
- Ayrılmış çoklu kısıt havuzu24 örnektir; tam64/32 final kabulü kapalıdır. Coding, English retention ve dil NLL bu merge feasibility deneyinde ölçülmedi. Production kabulü verilmedi.

## Gerçek süre

| Model | Saniye | Üretilen token |
|---|---:|---:|
| v2 | 564.0 | 70127 |
| v5 | 578.3 | 66000 |
| alpha_0.15 | 580.8 | 69233 |
| alpha_0.30 | 586.8 | 65004 |
| alpha_0.50 | 580.7 | 64526 |
| alpha_0.70 | 583.3 | 67831 |

## Sonuç sonrası içerik/biçim denetimi — seçim için kullanılmadı

Katı exact-match okuma/bellek skorları anlam doğruluğu olarak yorumlanamaz. Örnek denetiminde doğru bir kişi sayısının birimle yazılması ve doğru özel ismin noktayla bitmesi false negative üretti. TQuAD'da tarih sorusunun kaynak gold spanına fiilin de girdiği bir sınır sorunu görüldü. Bu bulguları gizlemiyoruz; dondurulmuş birincil skorları, örnekleri ve seçim eşiklerini sonuçtan sonra değiştirmedik.

Ek Unicode punctuation-normalized EM ve token-overlap F1, [standart extractive-QA değerlendirmesindeki](https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/squad/compute_score.py) ayrımdan yararlanır; Türkçe Unicode tokenları kullanır ve İngilizce article silmez. Bellek fact proxy'si literal slot bilgisini arar; sayısal cevaplarda ek/çelişkili sayıyı kabul etmez. Bunlar **lexical proxy**; entailment, muhakeme veya sohbet akıcılığı kanıtı değildir. Bu ek ölçümler altı üretim tamamlandıktan sonra geliştirildiği için checkpoint seçmekte veya eski başarısız kapıyı başarılıya çevirmekte kullanılmadı. Sonraki gerçek deneyde baştan sürümlenip dondurulmalıdır.

| Model | Okuma katı EM /64 | Noktalama-normalized EM /64 | Okuma lexical token F1 | Bellek katı /64 | Literal fact proxy /64 |
|---|---:|---:|---:|---:|---:|
| v2 | 8 | 11 | 0.424 | 8 | 41 |
| v5 | 11 | 14 | 0.480 | 8 | 38 |
| alpha_0.15 | 8 | 11 | 0.429 | 8 | 40 |
| alpha_0.30 | 11 | 14 | 0.445 | 8 | 40 |
| alpha_0.50 | 10 | 13 | 0.474 | 8 | 39 |
| alpha_0.70 | 11 | 13 | 0.478 | 8 | 39 |

Dolayısıyla bellek için8/64 katı skorunu yalnız8 bilgiyi hatırladı diye sunmak yanlıştır; literal içerik proxy'si38–41/64 aralığındadır. V5 okuma lexical F1'de de en yüksek noktasal değerdedir; hiçbir merge bu tanıda belirgin bir genel üstünlük göstermemiştir. Ancak bu sonradan eklenen proxyler başarı kanıtı veya bağımsız final test yerine kullanılamaz.

## Kapılardan neden geçilmedi?

- Alpha0.15: V5'e karşı math farkı−2/64 ve reading−3/64; retention kapıları geçmedi.
- Alpha0.30: math−1/64, instruction−1/80. Noktasal farklar küçük; ilgili paired CI alt sınırları sırasıyla−0.09375 ve−0.0875 olduğundan önceden belirlenen−0.08 kapısını geçmedi. Büyük regresyon kanıtı diye yorumlanmaz.
- Alpha0.50: V5'e karşı math−4/64; kapı geçmedi.
- Alpha0.70: İki ebeveyne karşı bütün nonregression kapılarını geçti, fakat iki ebeveyne karşı hiçbir ailede pozitif CI alt sınırıyla bir üstünlük göstermedi. V5'e göre math−1/64; instruction/reading eşit; cap3 daha fazla.

**Somut model kararı:** Bu dört alpha arasından üstünlüğü doğrulanmış bir checkpoint çıkmadı. Yeni merged model kaydedilmedi; production modelleri ve bütün ebeveyn checkpointleri korundu. Ayrılmış set açılmadı; final sonuçlarına bakarak başka alpha veya eşik seçilmedi. Bu, model merging hiçbir zaman çalışmaz ya da birleştirilmiş bütün modeller her görevde kötüdür sonucu değildir. Bu küçük deney yeni SFT'yi otomatik olarak gerekçelendirmez.

## Kaynaklar ve kalıcı kanıt

- [IFEval-TR kaynak](https://huggingface.co/datasets/bezir/IFEval_tr), [GSM8K-TR kaynak](https://huggingface.co/datasets/malhajar/gsm8k-tr), [AMC-TR kaynak](https://huggingface.co/datasets/barandinho/amc_turkish), [TQuAD kaynak](https://huggingface.co/datasets/husnu/tquad-v1v2).
- [Multi3WOZ resmî araştırma/kaynak](https://github.com/cambridgeltl/multi3woz); kullandığımız HF kopyası/revizyonu manifestte.
- Makine özeti `summary.json`, tüm paired farklar orada; ikincil hata denetimi `error_audit.json`.
- Private HF ham sonuç arşivi: `serda-dev/Jamba2-3B-Capability-Merge-20261019`, revision `90d26ca5ce78d5c9a950bdca1755eb44299d3500`, `run/logs.tar.gz`. SHA256 `f39b8701719c5bd7a28512e808a8a18170132e79db0a1b7ffd51302cf06b1998`. Bağımsız Hub/API hash/size kontrolü geçti. Ham prompt/yanıtlar public Git'e konmadı.
- Instance56745476 otomatik silindi; son live instance listesinde aktif instance yok, supervisor ve watchdog timer inactive. Son maliyet ve gecikmeli billing notu `completion.json` içindedir.
