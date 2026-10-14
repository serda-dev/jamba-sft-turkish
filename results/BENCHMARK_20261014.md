# Türkçe model yetenek ve hız karşılaştırması

Durum: Tüm testler tamamlandı.

## Yetenek testleri

| Model | MMLU | Belebele | Matematik kısa | Matematik 8 örnekli | Yönerge strict |
|---|---:|---:|---:|---:|---:|
| Jamba Türkçe SFT (3B) | %25.9 (233/900) | %49.9 (449/900) | %22.5 (45/200) | %12.5 (25/200) | %31.7 (38/120) |
| Qwen2.5 Instruct (3B) | %40.8 (367/900) | %67.0 (603/900) | %7.5 (15/200) | %42.0 (84/200) | %41.7 (50/120) |
| Qwen3 Instruct 2507 (4B) | %54.7 (492/900) | %78.0 (702/900) | %69.0 (138/200) | %70.0 (140/200) | %75.8 (91/120) |
| Jamba Türkçe CPT (3B) | %19.8 (178/900) | %28.8 (259/900) | %9.5 (19/200) | %3.0 (6/200) | %13.3 (16/120) |
| Orijinal Jamba2 (3B) | %22.8 (205/900) | %49.0 (441/900) | %19.0 (38/200) | %20.0 (40/200) | %46.7 (56/120) |

Orijinal/CPT üretim sonuçları, kalıtılan EOS=2 ile sohbet bitişi EOS=519 uyuşmazlığı düzeltilerek yeniden ölçüldü. Her üç yapılandırmanın bitiş tokenları kabul edilir. Bu iki sürümün üretim düzeltmesi önceki RTX 4090 fiziksel sunucusunda, aynı BF16 backend ve sorularla yapıldı; ilk üç modelin geçerli yanıtları korundu. Batch farkları küçük sayısal değişiklikler oluşturabilir. İlk hatalı baseline üretim raporu geçersizdir; arşivlenmiştir.


## Hız: RTX 4090, BF16, Transformers SDPA

512 giriş tokenı, 128 zorunlu çıkış tokenı, batch=1, ısınma hariç üç tekrarın medyanı.

| Model | Parametre (B) | Decode token/sn | Uçtan uca token/sn | İlk token (ms) | VRAM (GiB) | Batch 4 toplam token/sn |
|---|---:|---:|---:|---:|---:|---:|
| Jamba Türkçe SFT (3B) | 3.04 | 48.6 | 48.4 | 30.6 | 5.73 | 190.1 |
| Qwen2.5 Instruct (3B) | 3.09 | 51.0 | 50.8 | 30.2 | 5.81 | 199.2 |
| Qwen3 Instruct 2507 (4B) | 4.02 | 42.7 | 42.5 | 35.0 | 7.61 | 166.8 |
| Jamba Türkçe CPT (3B) | 3.04 | 48.6 | 48.4 | 30.3 | 5.73 | 189.3 |
| Orijinal Jamba2 (3B) | 3.03 | 48.8 | 48.6 | 30.5 | 5.71 | 189.7 |

## Yorum ve sınırlar

- Qwen3 yaklaşık 4B parametredir; 3B modellerle birebir aynı büyüklükte değildir.
- MMLU, yayımlanmış 900 soruluk alt kümedir. 0-shot seçenek harfi olasılığı kullanılır; makaledeki 5-shot liderlik tablosu yeniden üretilmemiştir.
- Belebele Türkçe okuduğunu anlama ölçer; 900 sorunun tümü kullanıldı.
- Matematikte aynı 200 sorunun iki istem biçimi saklandı. 8 örnekli test, lm-eval GSM8K-CoT örneklerinin Türkçe çevirisini kullanır. Kısa istem Qwen2.5 yanıtlarının 171/200’ünde tek satır ürettirdi; testin isteme duyarlılığı nedeniyle ikinci koşul tüm modellere eklendi.
- IFEval 120 deterministik Türkçeye uygun soruyla sınırlıdır; tüm benchmark skoru değildir. Strict, bir yanıttaki bütün koşulların sağlanmasıdır; olgusal doğruluk ölçmez.
- Her skorun Wilson %95 güven aralığı CSV/JSON dosyalarında bulunur. Özellikle 200 soruluk matematikte yakın farkları kesin üstünlük saymayın.
- SFT verileriyle 2.120 özgün test isteminde tam eşleşme yoktur. Yeni matematik koşulu aynı 200 soruyu kullanır. Anlamsal benzerlik ve ön eğitimdeki olası test verileri dışlanmış değildir.
- Greedy üretim ve BF16 ağırlıklar kullanıldı. Matematik kısa/8 örnekli bütçeleri 512/1.024, yönerge bütçesi 2.048 token. Sınıra dayanan çıktılar aşağıda ayrıca gösterilir.
- Hız ölçümleri bu donanım ve backend için geçerlidir. İndirme/derleme/tokenizasyon/ağ gecikmesi hariçtir; vLLM gibi optimize sunucuların ulaşabileceği azami hız değildir.
- Tokenizer farkları nedeniyle token/sn, aynı Türkçe metin miktarı/sn anlamına gelmez. Ham hız dosyalarında karakter/sn bulunur; zorunlu 128 token testinin ürettiği içerik modeller arasında farklıdır.

## Üretim bütçesine dayanan çıktılar ve yönerge ayrıntısı

| Model | Matematik kısa | Matematik 8 örnekli | Yönerge | Tek tek yönerge başarısı |
|---|---:|---:|---:|---:|
| Jamba Türkçe SFT (3B) | 15/200 | 58/200 | 33/120 | %37.2 |
| Qwen2.5 Instruct (3B) | 0/200 | 3/200 | 11/120 | %51.2 |
| Qwen3 Instruct 2507 (4B) | 10/200 | 22/200 | 8/120 | %80.5 |
| Jamba Türkçe CPT (3B) | 125/200 | 132/200 | 97/120 | %22.6 |
| Orijinal Jamba2 (3B) | 11/200 | 23/200 | 8/120 | %54.3 |

## Eğitim aşamalarının etkisi

Aynı test koşulunda gözlenen yüzde puan farkları; farklı görevler tek bir genel başarı oranında birleştirilmez.

| Test | CPT − orijinal | SFT − CPT |
|---|---:|---:|
| Türkçe MMLU | -3.0 | +6.1 |
| Türkçe Belebele | -20.2 | +21.1 |
| GSM8K-TR kısa yanıt | -9.5 | +13.0 |
| GSM8K-TR 8 örnekli çözüm | -17.0 | +9.5 |
| IFEval-TR alt küme (strict) | -33.3 | +18.3 |

## Koruma ve maliyet

GPU instance: 55857180; durum: COMPLETE; saatlik ücret: $0.4278.
Team kredi farkı: $1.08 (hesaptaki diğer hareketler ve gecikmeli tahakkuk etkileyebilir).
Ham yanıtlar, test manifesti, hız tekrarları ve loglar özel Hugging Face SFT reposunda `benchmarks/20261014` dizinine yüklenir. Dosyaların hash/ölçümleri doğrulandıktan sonra instance silinir.
