# Yönerge takibi: V3 sonrası teşhis ve önerilen çözüm

Tarih: 17 Ekim 2026. Bu çalışma CPU üzerinde kayıt incelemesi ve kaynak
araştırmasıdır. GPU kiralanmadı, yeni eğitim veya ücretli öğretmen çağrısı yapılmadı.

## Kısa sonuç

**V3 için hazırladığım veride görev kapsamı ve denetleyici eksikleri buldum.**
Paragraf örneklerinde öğrettiğim ayırıcı ile başarısız olunan testlerde istenen
ayırıcı örtüşmüyordu. Genel içerik/tekrar verisinin hedef-token payı da satır
oranından daha yüksekti. Bunlar düzeltilmesi gereken somut tasarım sorunlarıdır.
Toplam başarısızlığın tek nedeninin veri olduğu henüz kanıtlanmış değildir.

Önerim, V2 üzerinden **küçük, ölçümlü bir teşhis pilotu**, ardından doğrulayıcıdan
geçen daha doğal Türkçe görevlerle odaklı SFT'dir. Yeni bir büyük karışım veya
ek epoch, bu teşhis kapısından önce başlatılmamalı.

## 1. Doğrudan ölçülen bulgular

### 1.1. Paragraf sözleşmesi farklıydı

V3'te `paragraphs` ve `bounded_explanation` ailelerinden **1.411** eğitim
hedefi boş satırlarla ayrılıyor. İlgili eğitim istemleri de boş satır istiyor;
bu örnekler kendi istemlerine göre yanlış etiketli değil.

Ancak sabit testte başarısız olunan sekiz paragraf istemi açıkça `***` ayırıcı
istiyor. Puanlayıcı da bu ayırıcıyı sayıyor. Eğitimdeki bu 1.411 hedefin hiçbiri
aynı sayı için `***` sözleşmesini sağlamıyor. V3'ün küçük geliştirme örneklerinde
paragraf 2/2 iken dış testte 0/8 kalması bu kapsam farkıyla uyumlu.

Tekrarlanabilir örnek:

```text
İstenen: iki paragraf, ayırıcı ***

"Birinci paragraf.\n\nİkinci paragraf."  -> geçmez
"Birinci paragraf.\n***\nİkinci paragraf." -> geçer
```

**Düzeltme:** ayırıcıyı ve paragraf sayısını örnek başına açık parametre yap;
boş satır, `***`, istenen başlık biçimi ve farklı paragraf sayılarını çeşitlendir.
Öğrenilecek davranış kullanıcının belirttiği biçime uymaktır. Test soruları ve
yanıtları yeni eğitime alınmayacak; farklı içerik ve istemler üretilecek.
Kaynak [Google IFEval denetleyicisi](https://github.com/google-research/google-research/blob/master/instruction_following_eval/instructions.py)
bu ayırıcıyı açıkça tanımlar. Mevcut test skoru doğru hesaplanmış; skoru düzeltmek
için puanlayıcı değiştirilmedi.

### 1.2. Satır ve hedef-token oranları farklıydı

| V3 eğitim bileşeni | Satır | Assistant hedef tokenı | Hedef-token payı | Ortanca son yanıt uzunluğu |
|---|---:|---:|---:|---:|
| Yeni Türkçe kurallı görev | 12.000 | 688.846 | %38,18 | 9 kelime |
| İngilizce dış kaynak | 1.000 | 97.492 | %5,40 | 12 kelime |
| Önceki eğitimden tekrar | 4.000 | 1.017.710 | %56,41 | 53 kelime |

Türkçe görev satırları çoğunlukta olsa da uzun yanıtların büyük kısmı tekrar
verisinden geliyordu. On yedi görev ailesinin çoğu aynı kayıt formundaki
ad/kod/sayıları değiştiriyor. Bu, doğal sorulara uzun ve birden fazla koşulu
sağlayan yanıt üretme çeşitliliğini yeterince kapsamıyor.

Önemli sınır: tablodaki oran **token kütlesidir**, gradyan katkısının birebir
oranı değildir. Kullanılan Trainer mikro batch loss ortalamaları alıyor.
“Öğrenmenin %56'sı kesinlikle tekrar verisine gitti” sonucu çıkarılamaz.

### 1.3. Yerel denetleyicilerde kaçaklar var

Arşivlenmiş `sft_v3/data.py` üzerinde karşı örneklerle gösterildi:

- İki satırda birer kez kelime istenmesine rağmen aynı kelimeyi tek satırda
  iki kez yazmak geçebiliyor; toplam sıklık ölçülmüş, satır koşulu ölçülmemiş.
- İki farklı yanıt istenirken `aynı yanıt ****** aynı yanıt` geçebiliyor;
  ayırıcı kontrol edilmiş, yanıtların farklılığı kontrol edilmemiş.

Bunlar denetleyici güvenilirliği sorunlarıdır. Pozitif hedeflerimiz çoğunlukla
doğru kurulduğu için bu karşı örnekler, yanlış hedeflerin mutlaka eğitildiğini
kanıtlamaz. Sonraki veri doğrulamasında ve tercih çifti üretiminde giderilmeli.

Araştırma klasöründeki yeni `contracts.py` açık ayırıcı, satır başına sıklık,
farklı iki yanıt, kapanış ve JSON tipini denetler; desteklenmeyen veya bariz
çelişkili kuralları reddeder. Beş negatif/pozitif test geçti. Bu bir sonraki
pilot için küçük doğrulama prototipidir; bütün yönerge türleri uygulanmış değildir.
Arşivlenmiş V3 verisi ve skorları değiştirilmedi.

### 1.4. Etiket/EOS bozukluğu saptanmadı

17.000 eğitim kaydında, eski `encode()` fonksiyonunu tekrar kullanmadan:

- Etiketli tokenlar decode edilerek yalnızca gerçek assistant yanıtları ve
  sonlarındaki `<|im_end|>\n` ile birebir karşılaştırıldı: **0 fark**.
- Assistant sayısı ile EOS 519 etiketi sayısı karşılaştırıldı: **0 hata**.
- Etiketli tokenlarda role-start 518 veya pad 0: **0 kayıt**.

Araç kullanmayan üç farklı konuşmada, generation prompt açık/kapalı toplam altı
durumda sohbet metni resmi sabit Jamba şablonuyla aynı. Token ID'lerinin tümü
aynı olmak zorunda değil: resmi tokenizer 65.536, CPT tokenizer 69.632 token;
4.096 Türkçe token genişletmesi mevcut CPT tasarımının parçası. Özel BOS/EOS/pad
kimlikleri uyumlu. Burada açık bir maskeleme veya sohbet-çerçevesi hatası bulunmadı.

### 1.5. Sorun yalnızca biçim skorundan ibaret değil

V2 ve V3'te 120 yönerge yanıtının **56'sı**, tekrarlayan 8-gram oranı %20'yi
aşıyor. V3'te token sınırına takılma 34 → 41 oldu. Bazı kural alt skorları yükseldi
(yer tutucu 3/12 → 5/12, kelime sıklığı 5/12 → 7/12), toplam strict başarı 41/120
kaldı. “Model hiçbir şey öğrenmedi” demek de doğru değil; hedef davranış bütünü
iyileşmedi. Geliştirme loss düşüşü, üretim başarısının yerini tutmadı.

Tüm V3 eğitim örneklerinde system mesajı varken dış test yalnızca user mesajı
kullanıyor. Bu koşullandırma farkı ayrıca ölçülmeli; neden olduğu kanıtlanmış değil.
V3 prosedürel geliştirme üretimi zaten system mesajıyla 13/34 olduğu için bütün
sorunu system mesajı eksikliğine bağlamak mümkün değil.

## 2. Araştırma neyi destekliyor?

1. [AI21'in resmi model kartı](https://huggingface.co/ai21labs/AI21-Jamba2-3B)
   bu ailenin post-training aşamalarında SFT, DPO ve doğrulanabilir ödüllü
   on-policy eğitim kullandığını anlatıyor. Bizim kısa SFT'miz aynı eğitim
   tarifinin eşdeğeri değil. Bu, V3'ün neden başarısız olduğunun doğrudan kanıtı da değil.
2. [Generalizing Verifiable Instruction Following](https://arxiv.org/abs/2507.02833)
   farklı koşullara genelleme, doğrulama fonksiyonları ve doğrulanabilir ödülle
   eğitim üzerinde duruyor. Yalnızca sabit biçim kalıplarını çoğaltmak yerine
   koşul ve içerik çeşitliliği ile üretim doğrulamasını birlikte kullanmalıyız.
3. [Ai2 IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5)
   yapılandırılmış kural/argüman örnekleri için uygun bir referans. Hazır kaliteli
   Türkçe SFT yanıtları içermiyor; prompt ve ödül sözleşmeleri var. Ham görüntüde
   birbiriyle uyumsuz paragraf sayıları içeren örnek de görüldü. Bu kaynak olduğu
   gibi eğitime eklenmemeli; dil, kural desteği ve çelişki filtresi gerekir.
4. [Tulu preference verisi](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following)
   doğru/yanlış yanıt çifti yaklaşımına örnek. İngilizce veriyi körlemesine
   büyütmek yerine Türkçe görevler ve kendi modelimizin gerçek hataları kullanılabilir.

## 3. Önerilen uygulama — küçük pilotla doğrula

### A. Önce eğitim ve üretim yolunu ayır — pilot üst sınırı $2

Bu **gelecek pilot önerisidir**, kiralama yapılmadı. Ölçümler aynı GPU üzerinde
toplanıp bitince doğrulanmış kayıt sonrası instance silinmeli.

1. 32 yeni, basit Türkçe sözleşme/yanıt örneği hazırla; içerik ve tüm koşullar
   doğrulansın. Bu örnekleri küçük bir ezber testinde tekrar eğit, scratch
   checkpoint'i at. Beklenti en az %90 train sözleşme başarısı ve doğru EOS.
   Burada başarısızlık olursa veri büyütmeden optimizer/etiket/üretim yoluna dön.
2. Aynı promptlarda batch 1/16 ve cache açık/kapalı ilk token logits/üretimlerini
   karşılaştır; system mesajı var/yok etkisini ayrı kaydet. Sadece model çıktısının
   değişmesi bug kanıtı değildir; BF16 yakın logit durumları ayrıca değerlendirilir.
   Uzun tekrarların ve bitiş hatalarının tek bir yola yığılması incelenir.
3. Eski compiler/Mamba kontrolü küçük bir kernel hesabını test ediyordu;
   gerçek modelin bu dört üretim yolunun eşdeğerliğini kanıtlamıyordu. Mevcut
   inference kodunda left padding kullanılıyor. Bildirilen
   [right-padding cache sorunu](https://github.com/huggingface/transformers/issues/48256)
   bu nedenle bizim koşuya doğrudan teşhis olarak uygulanamaz; sadece bu testi
   tamamlamamız için teknik bir referanstır.
4. Sonraki sınırlı pilot için v2 referans skorunu 128 **yeni** doğal Türkçe
   geliştirme isteminde ölç. İlk 120 IFEval sorusunu checkpoint seçmek için kullanma.

### B. Düzeltilmiş veriyle odaklı SFT

- Başlangıç **V2**; başarısız V3'ün üstüne devam etmeyi önermiyorum.
- İlk paket **2.000–4.000** çeşitli, doğal Türkçe istek. İçerik: açıklama,
  özetleme, plan, metin dönüştürme, bilgi çıkarma ve konuşma takibi. Aynı stok
  kaydındaki isim/sayı değişiklikleri, çeşitliliğin çoğunluğunu oluşturmamalı.
- Her istekte **1–3 uyumlu koşul**: tam istenen ayırıcı, paragraf/kelime aralığı,
  kapanış, satır/listeler, gerekli/yasak sözcük. Kısa yanıt ve 80–160 kelimelik
  yanıtlar isteğe uygun dağıtılsın; her yanıtı kısaltmak hedef değil.
- Öğretmen model Türkçe yanıt üretir; kabul, bütün açık biçim kurallarından
  geçme **ve** soruyu/bağlamı doğru yanıtlama gerektirir. Biçim kontrolü programla,
  içerik referans bilgi ve örnek incelemesiyle değerlendirilir. Mevcut kalite
  etiketi teacher şeması, serbest SFT yanıt üretimi için ayrıca uyarlanmalı.
- Başlangıç hedef-token bütçesi: yaklaşık **%70 yönerge, %15 matematik,
  %15 genel tekrar**. Gerçek sampler/mikro batch katkısını da raporla; sadece
  örnek sayısına bakma. Tekrar örnekleri çeşitlendirilsin, kalıp/metayönerge
  cevapları ve doğrulanmamış uzun muhakeme metinleri elensin.
- Train/dev ayrımı temel soru/kaynak içerik grubuna göre yapılsın. Aynı sorunun
  farklı koşullu varyantları splitler arasında dolaşmasın. Bazı içerik türleri
  ve talimat ifade biçimleri geliştirme/son değerlendirme için ayrı tutulsun.
- System mesajı kullanılan ve kullanılmayan örnekler bulunsun. Geliştirme
  üretimleri gerçek kullanım biçimini kapsasın.
- Öğrenme oranı, 2e-6 ve 1e-5 gibi iki küçük pilot kolunda aynı yeni görevlerle
  karşılaştırılabilir. Önceki düşük LR'nin neden olduğu kanıtlanmış değil.
  Checkpoint seçimi yeni geliştirme sözleşme başarısıyla, loss ve matematik
  koruma kontrolü birlikte kullanılarak yapılsın.
- İlk 100–150 adımda geliştirmede en az +10 yüzde-puan ve tekrarlama/kesilmede
  düşüş aranır. Bu bir **erken devam ölçütü**, beklenen garanti sonuç değildir.
  İyileşme yoksa sonraki epoch'a geçilmez. Ek maliyet için yeni sınır konur.

### C. Gerekirse sınırlı tercih eğitimi

SFT pilotu koşulları kısmen öğreniyor ama tekrar/uyumsuzluk sürüyorsa, yeni train
istemlerinde v2/pilot modelin ürettiği gerçek hatalardan **500–1.000** doğrulanmış
doğru/yanlış çift oluştur. Aynı isteğe doğru, tamamlanmış yanıt chosen; koşul
ihlal eden gerçek yanıt rejected olur. DPO pilotu bu sinyali kullanabilir.

Bu yöntem koşullu öneridir; şu anda DPO'nun mevcut sorunu çözeceği kanıtlanmış
değil. Yanıtlar konu doğruluğunu korumalı; yalnızca kısa olduğu için chosen
sayılmamalı. Donmuş test yanıtları tercih verisine alınmaz. Yeni RL altyapısı
kurmadan bu sınırlı kontrolle ilerlemek yeterli bir sonraki deneydir.

## 4. Bu turda yapılan işler ve kanıtlar

| İş | Sonuç | Dosya |
|---|---|---|
| 17.000 kaydın bağımsız assistant/EOS kontrolü | 0 hata | `diagnosis.json` |
| Satır/token/yanıt uzunluğu dağılımı | Ölçüldü | `findings.json`, `diagnosis.json` |
| Paragraf sözleşmesi çapraz kontrolü | 1.411 boş-satır hedefi; yıldız sözleşmesiyle 0 eşleşme | `contract_findings.json` |
| V2/V3 kural ve tekrar karşılaştırması | Yapıldı | `diagnosis.json` |
| Eski checker yanlış kabul örnekleri | Satır ve farklı-yanıt açığı yeniden üretildi | `diagnose.py` |
| Yeni dar checker prototipi | 5 test geçti | `contracts.py`, `test_contracts.py` |
| Yeni GPU/eğitim/ücretli öğretmen | Yapılmadı | Bu tur analiz ve çözüm önerisi |

Yeniden üretim, repo kökünden:

```bash
/home/serda/.local/share/linguai-training/venv/bin/python research/instruction_following_20261017/diagnose.py
/home/serda/.local/share/linguai-training/venv/bin/python -m pytest -q research/instruction_following_20261017/test_contracts.py
```

**Karar:** önce sözleşme kapsamını ve checker açıklarını düzelt, $2 tavanlı
tanı pilotuyla modelin öğrenme/üretim yolunu doğrula; ardından doğal Türkçe ve
tam doğrulanmış yanıtlarla küçük SFT. Bu bulgularla aynı V3 karışımını uzatmayı
veya doğrudan daha büyük bir veri setine geçmeyi önermiyorum.


## Onaylanan tanı pilotu

Kullanıcı onayıyla [$2 pilot prosedürü](PILOT_TR.md) hazırlanmıştır. Parametre saklama hassasiyeti hipotezi de gerçek optimizer güncellemeleriyle karşılaştırılır. Henüz üretim modeli değişikliği anlamına gelmez.


## Tamamlanan GPU tanısı — 17 Ekim, 22:26 UTC

[$2 pilotu tamamlandı](PILOT_TR.md), tüm denemeler dahil yaklaşık **$0.46**. Ayrı tanı başarısı: saf BF16 LR2e-6 **79/128**, BF16 working + FP32 ana ağırlık LR2e-6 **112/128**, aynı yöntem LR1e-5 **128/128**. Her iki master kolu train 32/32 yaptı. Sayısal ölçümler, küçük optimizer güncellemelerinin saf BF16da çoğu örneklenmiş ağırlıkta kaybolduğunu doğruladı. Sonraki SFTde **FP32 ana ağırlık biriktirme + düzeltilmiş doğal Türkçe veri** birlikte kullanılmalı. Bu küçük şablon tanısı genel model başarısı değildir. Scratch ağırlıkları saklanmadı; V2 korunuyor. GPUlar silindi, 736 yanıtın yeniden puanlanmasında 0 fark ve 10 CPU testi geçti.
