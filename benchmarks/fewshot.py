"""Faithful Turkish translations of the eight lm-eval GSM8K-CoT examples."""
EXAMPLES=[
 ('Koruda 15 ağaç var. İşçiler bugün ağaç dikecek. İşleri bitince 21 ağaç olacak. Kaç ağaç diktiler?',
  'Başta 15 ağaç vardı, sonunda 21 ağaç oldu. Dikilen ağaç sayısı 21 - 15 = 6.\n#### 6'),
 ('Otoparkta 3 araba var ve 2 araba daha geliyor. Otoparkta kaç araba var?',
  'Başlangıçtaki 3 arabaya 2 araba eklenir: 3 + 2 = 5.\n#### 5'),
 ('Leah\'ın 32, kız kardeşinin 42 çikolatası vardı. 35 tanesini yediler. Toplam kaç çikolataları kaldı?',
  'Başlangıçta 32 + 42 = 74 çikolataları vardı. 35 tanesini yiyince 74 - 35 = 39 kaldı.\n#### 39'),
 ('Jason\'ın 20 lolipopu vardı. Denny\'ye bir kısmını verdi. Şimdi 12 lolipopu var. Denny\'ye kaç lolipop verdi?',
  'Başlangıçta 20 vardı ve 12 kaldı. Verdiği miktar 20 - 12 = 8.\n#### 8'),
 ('Shawn\'ın beş oyuncağı var. Noel için annesinden iki, babasından iki oyuncak aldı. Şimdi kaç oyuncağı var?',
  'Yeni oyuncak sayısı 2 + 2 = 4. Toplam 5 + 4 = 9 oyuncak.\n#### 9'),
 ('Sunucu odasında dokuz bilgisayar vardı. Pazartesiden perşembeye kadar her gün beş bilgisayar eklendi. Şimdi kaç bilgisayar var?',
  'Dört gün boyunca 4 × 5 = 20 bilgisayar eklenir. Başlangıçtaki dokuzla birlikte 9 + 20 = 29 olur.\n#### 29'),
 ('Michael\'ın 58 golf topu vardı. Salı günü 23, çarşamba günü 2 top kaybetti. Çarşamba sonunda kaç topu kaldı?',
  'Salı sonunda 58 - 23 = 35 top kalır. Çarşamba sonunda 35 - 2 = 33 top kalır.\n#### 33'),
 ('Olivia\'nın 23 doları vardı. Tanesi 3 dolardan beş simit aldı. Kaç doları kaldı?',
  'Simitlerin maliyeti 5 × 3 = 15 dolar. Kalan para 23 - 15 = 8 dolar.\n#### 8')]


def prompt(question):
    examples='\n\n'.join(f'Soru: {q}\nÇözüm: {a}' for q,a in EXAMPLES)
    return ('Aşağıdaki örneklerdeki gibi soruyu Türkçe adım adım çöz. Hesaplamaları göster. '
            'Son satırda #### işaretinden sonra sayısal nihai cevabı yaz.\n\n'+examples+
            '\n\nSoru: '+question+'\nÇözüm:')
