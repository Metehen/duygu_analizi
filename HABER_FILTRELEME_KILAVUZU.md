# 🎯 Haber Filtreleme Kılavuzu

## Genel Bakış

Sistem artık çekilen haberleri seçilen hisse senedi ile ilişkisine göre filtreleyebiliyor. Bu sayede daha doğru tahminler elde edebilirsiniz.

## Nasıl Çalışır?

### 1. Anahtar Kelime Eşleştirme

Her hisse senedi için önceden tanımlanmış anahtar kelimeler var:

**Örnek: THYAO (Türk Hava Yolları)**
- türk hava yolları
- thy
- turkish airlines
- havayolu
- uçak
- havacılık

**Örnek: GARAN (Garanti Bankası)**
- garanti
- garanti bankası
- bbva
- garanti bbva

### 2. Sektör Bazlı Filtreleme

Hisse özel kelime bulunamazsa, sektör bazlı genel kelimeler kullanılır:

**Bankacılık Sektörü:**
- banka, kredi, mevduat, faiz, finansal

**Havacılık Sektörü:**
- havayolu, uçak, havacılık, yolcu, kargo

**Enerji Sektörü:**
- enerji, petrol, doğalgaz, elektrik

### 3. İlişki Skoru

Her haber için 0.0 - 1.0 arası bir ilişki skoru hesaplanır:

- **0.7 - 1.0**: Yüksek ilişkili (hisse özel haberler)
- **0.3 - 0.7**: Orta ilişkili (sektör haberleri)
- **0.0 - 0.3**: Düşük ilişkili (genel ekonomi haberleri)

## Kullanım

### Sidebar Ayarları

**Gelişmiş Ayarlar** bölümünde:

1. **Hisse Özel Haber Filtresi**: Filtrelemeyi aktif/pasif yapın
2. **Minimum İlişki Skoru**: Eşik değerini ayarlayın
   - 0.0: Tüm haberler (filtreleme yok)
   - 0.2: Gevşek filtreleme (önerilen)
   - 0.5: Orta filtreleme
   - 0.7: Sıkı filtreleme (sadece çok ilgili haberler)
3. **Sektör Haberlerini Dahil Et**: Sektör haberlerini de kullanın

### Önerilen Ayarlar

**Popüler Hisseler için (THYAO, GARAN, vb.):**
```
✅ Hisse Özel Haber Filtresi: Aktif
📊 Minimum İlişki Skoru: 0.2
✅ Sektör Haberlerini Dahil Et: Aktif
```

**Az Bilinen Hisseler için:**
```
✅ Hisse Özel Haber Filtresi: Aktif
📊 Minimum İlişki Skoru: 0.1
✅ Sektör Haberlerini Dahil Et: Aktif
```

**Genel Piyasa Analizi için:**
```
❌ Hisse Özel Haber Filtresi: Pasif
```

## Desteklenen Hisseler

Şu anda anahtar kelimeler tanımlı hisseler:

- **THYAO**: Türk Hava Yolları
- **GARAN**: Garanti Bankası
- **ISCTR**: İş Bankası
- **AKBNK**: Akbank
- **TUPRS**: Tüpraş
- **ASELS**: Aselsan
- **EREGL**: Ereğli Demir Çelik
- **KCHOL**: Koç Holding
- **SAHOL**: Sabancı Holding
- **PETKM**: Petkim
- **SISE**: Şişe Cam
- **TCELL**: Turkcell
- **TTKOM**: Türk Telekom
- **VAKBN**: Vakıfbank
- **YKBNK**: Yapı Kredi

## Yeni Hisse Ekleme

Eğer kullandığınız hisse listede yoksa, `src/news_filter.py` dosyasındaki `STOCK_KEYWORDS` dictionary'sine ekleyebilirsiniz:

```python
STOCK_KEYWORDS = {
    'YENI_HISSE': ['anahtar1', 'anahtar2', 'anahtar3'],
    # ...
}
```

## Sonuçları Yorumlama

Analiz sonrasında göreceğiniz metrikler:

### Haber Filtreleme Özeti
```
🎯 THYAO ile ilgili 45/100 haber filtrelendi

Yüksek İlişkili: 15
Orta İlişkili: 20
Düşük İlişkili: 10
```

Bu örnek:
- 100 haber çekildi
- 45 tanesi THYAO ile ilgili bulundu
- 15 tanesi çok ilgili (hisse özel)
- 20 tanesi orta ilgili (sektör)
- 10 tanesi düşük ilgili (genel)

### Örnek Haberler Tablosu

Filtrelenmiş haberlerin ilk 5'ini gösterir. `relevance_score` sütunu ilişki skorunu gösterir.

## Sorun Giderme

### "Hiç ilgili haber bulunamadı"

**Çözüm 1**: Minimum ilişki skorunu düşürün (0.1 veya 0.0)

**Çözüm 2**: Sektör haberlerini dahil edin

**Çözüm 3**: Filtrelemeyi tamamen kapatın

### "Çok az haber kaldı"

Eğer 10'dan az haber kalıyorsa:
- Minimum ilişki skorunu düşürün
- Tarih aralığını genişletin
- Daha fazla RSS kaynağı ekleyin

### "İlgisiz haberler geliyor"

Eğer hala ilgisiz haberler geliyorsa:
- Minimum ilişki skorunu yükseltin (0.5 veya 0.7)
- Hisse için özel anahtar kelimeler ekleyin
- Sektör haberlerini devre dışı bırakın

## Gelecek Geliştirmeler

- [ ] Makine öğrenmesi ile otomatik anahtar kelime çıkarma
- [ ] Şirket ismi NER (Named Entity Recognition)
- [ ] Haber kaynağı güvenilirlik skoru
- [ ] Tarih bazlı ağırlıklandırma (yeni haberler daha önemli)
- [ ] Kullanıcı tanımlı anahtar kelimeler (UI'dan)

## Notlar

- Filtreleme tamamen anahtar kelime bazlıdır, semantik analiz yapmaz
- Türkçe karakterler dikkate alınır (ş, ğ, ı, vb.)
- Büyük/küçük harf duyarlı değildir
- Haber başlığı ve içeriği birlikte değerlendirilir
