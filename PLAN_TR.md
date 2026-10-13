# CPT sonrası Türkçe SFT planı — 13 Ekim 2026

## Hedef

1B token CPT tamamlandı. Son ayrılmış doğrulama ölçümleri Türkçe NLL 1,693 /
perplexity 5,44; İngilizce NLL 1,829 / perplexity 6,23. Küçük JSON/talimat testi
3/12 olarak kaldı. SFT'nin amacı, öğrenilen Türkçeyi koruyarak soruya doğrudan
yanıt verme, biçim kısıtlarına uyma ve konuşma takibini geliştirmek.

Başlangıç modeli, özel HF reposundaki doğrulanmış `checkpoints/final`:
`serda-dev/Jamba2-3B-Turkish-CPT-1B-20261007`, revision
`8b1a9c812791fa7930e64b9444b41519c023269e`.
Eski SFT GitHub reposuna erişildi; repo boş olduğundan önceki eğitim kodu veya
ayarları referans alınamadı. Mevcut çalışan CPT ortamı temel alındı.

## Hazırlanan veri

| Kaynak | Eğitim | Doğrulama | Kullanım |
| --- | ---: | ---: | --- |
| [Türkçe Atlas](https://huggingface.co/datasets/AlicanKiraz0/Turkce-Atlas-Instruct) | 23.737 | 263 | Talimat, biçim ve metin dönüştürme |
| [Merve Turkish Instructions](https://huggingface.co/datasets/merve/turkish_instructions) | 11.868 | 132 | Kısa görev ve yanıtlar |
| [Aya Dataset, Türkçe](https://huggingface.co/datasets/CohereLabs/aya_dataset) | 3.783 | 34 | İnsan yazımı/düzenlemesi örnekler |
| [Turkish Knowledge SFT](https://huggingface.co/datasets/Uunan/turkish-knowledge-sft) | 7.922 | 78 | Çeşitli açıklama görevleri |
| [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 7.913 | 87 | İngilizce ve çok turlu konuşma |
| Yerelde üretilen doğrulanmış hedefler | 7.860 | 86 | JSON, aritmetik, bilgi çıkarma, konuşma belleği |
| **Toplam** | **63.083** | **680** | |

Eğitimde 32.549.318 giriş tokenı ve 17.618.670 asistan hedef tokenı var.
İki epoch yaklaşık 65,1M giriş / 35,2M hedef tokenı işleyecek. Padding hesaplama
maliyetini artırır; bu rakamlar gerçek veri tokenlarıdır. Konuşmaların yaklaşık
%87'si Türkçe; tokenların yaklaşık %73'ü Türkçe, %27'si İngilizcedir.

Veri revision'ları sabitlendi. Aynı normalleştirilmiş kullanıcı talimatı farklı
kaynaklarda tekrar kullanılamaz. %1 doğrulama prompt hash'ine göre ayrıldı.
2.048 tokenı geçen konuşmalar kesilmeden elendi; bozuk roller, boş yanıtlar,
özel token enjeksiyonu ve belirgin tekrarlı çıktılar filtrelendi. Sentetik
yanıtların tümünde doğruluk garanti edilmiyor; örnek inceleme dosyası kaydedildi.
Anlamsal olarak benzer tüm soruları kapsayan bir dedup iddiası yok.

## Eğitim

- Tek A100 SXM4 80GB; instance `55724188`, disk dahil 1,220833 $/saat.
- **Tüm 3B ağırlıklar** BF16 olarak güncellenir. AdamW optimizer durumları 8-bit;
  model ağırlıkları quantize edilmez.
- En fazla 2 epoch, yaklaşık 3.944 optimizer adımı.
- Learning rate 1e-5, cosine schedule, %3 warmup, weight decay 0,01.
- Microbatch 4, gradient accumulation 8: etkin batch 32 konuşma.
- Yalnızca asistan yanıtı ve yanıt bitiş tokenları loss'a dahil.
- Dinamik padding ve uzunluğa göre gruplama. Ayrı konuşmalar birleştirilmez.
- Her 300 adımda doğrulama ve checkpoint; doğrulama loss'u en düşük model seçilir.

İlk 10 optimizer adımı GPU üzerinde hatasız tamamlandı; ilerleme göstergesi bu
noktada yaklaşık 5 saat kalan eğitim öngörüyordu. İlk gruplar uzun konuşmalar
içerdiği için hız henüz sabit değil. Değerlendirme/yüklemeler dahil ilk tahmin
**5–8 saat ve yaklaşık 6–10 $**; daha uzun bir ölçümle güncellenmelidir.
Başlangıç asistan doğrulama loss'u 1,5307; 80 kesin yanıt testinde sonuç 0/80.
İlk 10 adımdaki eğitim loss'u 1,3204. Farklı veri olduğundan bu iki loss değeri
tek başına iyileşme kanıtı olarak karşılaştırılmamalıdır.
Mevcut kredi 41,80 $ civarı. 20 saat veya kalan kredi 12 $ sınırında düzgün
durdurma istenir; export sorunu varsa 7 $ sınırında GPU durdurulup disk korunur.
Yeni instance veya ek kredi otomatik alınmaz.

## Başarıyı nasıl ölçeceğiz?

1. Ayrılmış 680 konuşmadaki asistan loss'u: CPT başlangıcı / en iyi SFT.
2. Eğitim dışında 80 adet kesin hedefli JSON, aritmetik, çıkarma ve bellek testi.
   Eğitim ve testte sayı aralıkları farklıdır. Bu bir genel yetenek benchmark'ı
   değildir; aritmetik ve biçim başarısı birlikte ölçülür.
3. CPT'deki aynı 2M token Türkçe/İngilizce doğrulama seti ve aynı 12 küçük test.
4. Otomatik kabul koşulları: asistan loss'u iyileşsin, 80 test toplamı düşmesin,
   Türkçe NLL en fazla %5, İngilizce NLL en fazla %10 artsın. Sonuç ayrıca
   yanıt örnekleriyle değerlendirilmelidir; koşulları geçmek genel sohbet
   kalitesinin kanıtı değildir.

## Saklama ve operasyon

Çıktı: özel `serda-dev/Jamba2-3B-Turkish-SFT-20261013` reposu.
Hazırlanmış veri ve kaynak incelemesi `run/prepared/` altında saklandı.
Checkpoint'ler tokenizer ve optimizer durumlarıyla yüklenir; final klasörü
seçilen model ve tokenizer'ı içerir. Eğitim sürdürmek için periyodik checkpoint
kullanılır. Final model, `checkpoints/final` altında bulunacak.

Loglar on dakikada bir VPS ve HF'ye yedeklenir. Final model ve logların uzak
kopyası boyut/hash doğrulamasından geçmeden instance silinmez. Hata durumunda
GPU durdurulur, disk ve tanı logları saklanır. API tokenları arşivlenmez.

VPS çalışma dizini: `/home/serda/sft-run-20261013`.
Kalıcı servis: `linguai-sft-supervisor.service`.
