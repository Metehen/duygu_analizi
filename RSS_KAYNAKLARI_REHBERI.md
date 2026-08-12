# 📰 RSS Kaynakları Rehberi

## Yeni Gruplandırılmış Yapı

RSS kaynakları artık daha organize bir şekilde gruplandırılmıştır:

### 1. 📊 Borsa Odaklı Kaynaklar (7 Kaynak)
Doğrudan borsa, hisse senetleri ve finansal piyasalara odaklanan kaynaklar:

- **Bloomberg HT** - `https://www.bloomberght.com/rss`
- **Para Analiz** - `https://www.paraanaliz.com/feed/`
- **Investing.com Türkiye** - `https://www.investing.com/rss/news_285.rss`
- **Trading Economics** - `https://tr.tradingeconomics.com/rss/turkey.xml`
- **Foreks** - `https://www.foreks.com/rss`
- **BigPara** - `https://www.bigpara.com/rss`
- **KAP (Kamuyu Aydınlatma Platformu)** - `https://www.kap.org.tr/tr/rss`

### 2. 🏭 Sektörel Kaynaklar (6 Kaynak)

#### 2.1 ✈️ Turizm Sektörü (2 Kaynak)
THYAO, MAALT gibi turizm şirketleri için:
- **Turizm Güncel** - `https://www.turizmguncel.com/rss`
- **Turizm Haberleri** - `https://www.turizmhaberleri.com/rss`

#### 2.2 ⚡ Enerji Sektörü (2 Kaynak)
TUPRS, PETKM, AKSEN gibi enerji şirketleri için:
- **Enerji Gündem** - `https://www.enerjigundem.com/rss`
- **Enerji Portalı** - `https://www.enerjiportali.com/rss`

#### 2.3 💻 Teknoloji Sektörü (2 Kaynak)
TCELL, TTKOM, ASELS gibi teknoloji şirketleri için:
- **Teknoloji Turu** - `https://www.teknolojituru.com/rss`
- **Teknoloji Oku** - `https://www.teknolojioku.com/rss`

### 3. 📈 Analiz ve Yorum Kaynakları (4 Kaynak)
Uzman analizleri ve piyasa yorumları:

- **Analiz Market** - `https://www.analizmarket.com/rss`
- **Borsa Gündem** - `https://www.borsagundem.com/rss`
- **Dünya Gazetesi** - `https://www.dunya.com/rss`
- **Ekonomim** - `https://www.ekonomim.com/rss`

### 4. 📰 Genel Ekonomi Haberleri (8 Kaynak)
Ana akım haber kaynakları:

- **Mynet Ekonomi** - `https://www.mynet.com/haber/rss/ekonomi`
- **Habertürk Ekonomi** - `https://www.haberturk.com/rss/ekonomi.xml`
- **Hürriyet Ekonomi** - `https://www.hurriyet.com.tr/rss/ekonomi`
- **Sabah Ekonomi** - `https://www.sabah.com.tr/rss/ekonomi.xml`
- **Milliyet Ekonomi** - `https://www.milliyet.com.tr/rss/rssnew/ekonomirssnew.xml`
- **Anadolu Ajansı** - `https://www.aa.com.tr/tr/rss/default?cat=ekonomi`
- **NTV Ekonomi** - `https://www.ntv.com.tr/ekonomi.rss`
- **CNN Türk Ekonomi** - `https://www.cnnturk.com/feed/rss/ekonomi/news`

## 📊 İstatistikler

### Toplam Kaynak Sayısı
- **Borsa Odaklı**: 6 kaynak (KAP hariç)
- **Sektörel**: 6 kaynak (3 kategori)
  - Turizm: 2
  - Enerji: 2
  - Teknoloji: 2
- **Analiz ve Yorum**: 4 kaynak
- **Genel Ekonomi**: 8 kaynak
- **TOPLAM**: 24 kaynak 🚀

### Önceki Durumla Karşılaştırma
- **Önceki**: 15 kaynak
- **Şimdi**: 24 kaynak
- **Artış**: %60 📈

**Not:** KAP RSS'i format hatası nedeniyle devre dışı bırakıldı. KAP bildirimleri diğer haber kaynaklarında (Bloomberg HT, Para Analiz vb.) yayınlanır.

### 3. THYAO Anahtar Kelimeleri
**Önceki** (6 kelime):
- türk hava yolları, thy, turkish airlines, havayolu, uçak, havacılık

**Şimdi** (24 kelime):
- türk hava yolları, thy, turkish airlines, havayolu, uçak, havacılık
- türk hava, hava yolları, thy a.o, thy a.ş, thy hisse
- uçuş, pilot, hostes, yolcu, kargo, boeing, airbus
- istanbul havalimanı, sabiha gökçen, atatürk havalimanı
- bilet, rezervasyon, miles, miles&smiles

### 4. Havacılık Sektörü Kelimeleri
**Önceki** (5 kelime):
- havayolu, uçak, havacılık, yolcu, kargo

**Şimdi** (17 kelime):
- havayolu, uçak, havacılık, yolcu, kargo, uçuş
- havalimanı, pilot, hostes, bilet, rezervasyon
- boeing, airbus, sefer, rota, destinasyon
- iata, icao, sivil havacılık

## Beklenen Sonuçlar

### THYAO İçin
- **Önceki**: ~1-5 ilgili haber
- **Beklenen**: ~30-50 ilgili haber (turizm sektörü kaynakları sayesinde)
- **İyileşme**: 6-10x daha fazla 🎯

### Diğer Hisseler İçin
- **Bankacılık** (GARAN, AKBNK, vb.): ~40-60 haber
- **Teknoloji** (TCELL, TTKOM): ~25-40 haber (teknoloji sektörü kaynakları sayesinde)
- **Enerji** (TUPRS): ~20-35 haber (enerji sektörü kaynakları sayesinde)
- **Perakende** (BIMAS, MGROS): ~20-35 haber

## 🔧 Kullanım

### config.py'de Yapı

```python
RSS_FEEDS = {
    'borsa_odakli': [...],
    'sektorel': {
        'turizm': [...],
        'enerji': [...],
        'teknoloji': [...],
    },
    'analiz_yorum': [...],
    'genel_ekonomi': [...],
}

# Tüm kaynakları tek liste olarak al
RSS_FEEDS_LIST = get_all_rss_feeds()
```

### Belirli Bir Grubu Kullanma

```python
# Sadece borsa odaklı kaynakları kullan
from config import RSS_FEEDS
borsa_feeds = RSS_FEEDS['borsa_odakli']

# Sadece turizm sektörü kaynakları
turizm_feeds = RSS_FEEDS['sektorel']['turizm']

# Tüm kaynakları kullan
from config import RSS_FEEDS_LIST
all_feeds = RSS_FEEDS_LIST
```

## 📝 Yeni Kaynak Ekleme

### Borsa Odaklı Kaynak Ekle

```python
RSS_FEEDS = {
    'borsa_odakli': [
        # Mevcut kaynaklar...
        'https://yeni-borsa-kaynagi.com/rss',  # Yeni kaynak
    ],
    # ...
}
```

### Yeni Sektör Ekle

```python
RSS_FEEDS = {
    # ...
    'sektorel': {
        'turizm': [...],
        'enerji': [...],
        'teknoloji': [...],
        # Yeni sektör ekle
        'otomotiv': [
            'https://www.otomotivhaber.com/rss',
            'https://www.autoturk.com/rss',
        ],
    },
    # ...
}
```

## Ek Kaynak Önerileri

### ✅ Eklenen Kaynaklar

Aşağıdaki kaynaklar artık projeye eklenmiştir:

**Borsa Odaklı:**
- ✅ `https://www.foreks.com/rss` - Foreks
- ✅ `https://www.bigpara.com/rss` - BigPara
- ✅ `https://www.kap.org.tr/tr/rss` - KAP (Kamuyu Aydınlatma Platformu)

**Sektörel - Turizm:**
- ✅ `https://www.turizmguncel.com/rss` - Turizm Güncel
- ✅ `https://www.turizmhaberleri.com/rss` - Turizm Haberleri

**Sektörel - Enerji:**
- ✅ `https://www.enerjigundem.com/rss` - Enerji Gündem
- ✅ `https://www.enerjiportali.com/rss` - Enerji Portalı

**Sektörel - Teknoloji:**
- ✅ `https://www.teknolojituru.com/rss` - Teknoloji Turu
- ✅ `https://www.teknolojioku.com/rss` - Teknoloji Oku

**Analiz ve Yorum:**
- ✅ `https://www.analizmarket.com/rss` - Analiz Market
- ✅ `https://www.borsagundem.com/rss` - Borsa Gündem

### 🔮 Gelecekte Eklenebilecek Kaynaklar

**Bankacılık Sektörü:**
```python
'bankacilik': [
    'https://www.bankacilik.com/rss',
    'https://www.finansgundem.com/rss',
]
```

**Otomotiv Sektörü:**
```python
'otomotiv': [
    'https://www.otomotivhaber.com/rss',
    'https://www.autoturk.com/rss',
]
```

**Perakende Sektörü:**
```python
'perakende': [
    'https://www.perakendegundem.com/rss',
    'https://www.retailturkiye.com/rss',
]
```

**İnşaat ve Gayrimenkul:**
```python
'insaat_gayrimenkul': [
    'https://www.insaatgundem.com/rss',
    'https://www.emlakkulisi.com/rss',
]
```

## RSS Kaynağı Ekleme

### Yeni Kaynak Ekleme Adımları

1. **config.py dosyasını açın**
2. **Uygun kategoriyi bulun** (borsa_odakli, sektorel, analiz_yorum, genel_ekonomi)
3. **RSS URL'sini ekleyin**

**Örnek - Borsa Odaklı Kaynak:**
```python
RSS_FEEDS = {
    'borsa_odakli': [
        'https://www.bloomberght.com/rss',
        # ... mevcut kaynaklar
        'https://yeni-kaynak.com/rss',  # Yeni kaynak ekle
    ],
}
```

**Örnek - Yeni Sektör Ekle:**
```python
RSS_FEEDS = {
    # ...
    'sektorel': {
        'turizm': [...],
        'enerji': [...],
        'teknoloji': [...],
        'bankacilik': [  # Yeni sektör
            'https://www.bankacilik.com/rss',
            'https://www.finansgundem.com/rss',
        ],
    },
}
```

**Önemli:** Yeni sektör ekledikten sonra `get_all_rss_feeds()` fonksiyonunu güncellemeye gerek yoktur, otomatik olarak tüm sektörleri tarar.

### RSS URL'si Bulma

1. Haber sitesine gidin
2. Sayfanın alt kısmında "RSS" ikonu arayın
3. Veya `/rss`, `/feed`, `/rss.xml` uzantılarını deneyin
4. Tarayıcıda XML formatında açılıyorsa doğru URL'dir

### Test Etme

```python
import feedparser

feed = feedparser.parse('https://test-url.com/rss')
print(f"Başlık: {feed.feed.title}")
print(f"Haber sayısı: {len(feed.entries)}")
```

## Sorun Giderme

### "Bazı RSS kaynakları çalışmıyor"

**Neden**: Bazı siteler RSS'i kaldırmış veya değiştirmiş olabilir

**Çözüm**: 
- Hata veren kaynağı config.py'den kaldırın
- Alternatif kaynak ekleyin
- Log'larda hangi kaynağın hata verdiğini kontrol edin

### "Hala az haber geliyor"

**Çözüm 1**: Minimum ilişki skorunu düşürün (0.1 veya 0.0)

**Çözüm 2**: Daha fazla RSS kaynağı ekleyin

**Çözüm 3**: Anahtar kelimeleri genişletin

### "Çok fazla haber, yavaş çalışıyor"

**Çözüm**: `NEWS_CONFIG['max_articles']` değerini düşürün (150 veya 100)

## Performans İpuçları

### Hızlı Analiz İçin
```python
NEWS_CONFIG = {
    'max_articles': 100,  # Daha az haber
    'timeout': 15,        # Daha kısa timeout
}
```

### Kapsamlı Analiz İçin
```python
NEWS_CONFIG = {
    'max_articles': 300,  # Daha fazla haber
    'timeout': 60,        # Daha uzun timeout
}
```

## Yasal Uyarı

- RSS beslemeleri haber sitelerinin izniyle kullanılır
- Ticari kullanım için site sahiplerinden izin alın
- Robot.txt kurallarına uyun
- Aşırı istek göndermekten kaçının (rate limiting)

## Güncelleme Geçmişi

- **v1.0** (İlk versiyon): 2 kaynak, 100 haber
- **v2.0**: 15 kaynak, 200 haber, genişletilmiş anahtar kelimeler
- **v3.0** (Şu anki): 25 kaynak, gruplandırılmış yapı, sektörel kategoriler
  - Borsa odaklı: 7 kaynak
  - Sektörel (turizm, enerji, teknoloji): 6 kaynak
  - Analiz ve yorum: 4 kaynak
  - Genel ekonomi: 8 kaynak

## Gelecek Planlar

- [ ] Otomatik RSS keşfi
- [ ] Kaynak güvenilirlik skoru
- [ ] Haber kaynağı performans analizi
- [ ] Kullanıcı tanımlı RSS kaynakları (UI'dan)
- [ ] RSS cache mekanizması (daha hızlı)
- [x] RSS kaynaklarını gruplandırma (✅ Tamamlandı)
- [x] Sektörel kategoriler ekleme (✅ Tamamlandı)
- [ ] Dinamik sektör seçimi (UI'dan hangi sektörlerin kullanılacağını seçme)
- [ ] RSS kaynak sağlık kontrolü (hangi kaynaklar çalışıyor/çalışmıyor)

## 🎯 Avantajlar

### Gruplandırılmış Yapının Faydaları

1. **Daha İyi Organizasyon**: Kaynaklar kategorilere ayrılmış
2. **Kolay Yönetim**: Belirli bir kategoriyi kolayca ekle/çıkar
3. **Sektörel Odaklanma**: Hisse senedine göre ilgili sektör kaynaklarını kullan
4. **Esneklik**: İhtiyaca göre sadece belirli kategorileri kullan
5. **Ölçeklenebilirlik**: Yeni sektörler kolayca eklenebilir

### Kullanım Senaryoları

**Senaryo 1: Sadece Borsa Odaklı Analiz**
```python
from config import RSS_FEEDS
feeds = RSS_FEEDS['borsa_odakli']
```

**Senaryo 2: Turizm Hissesi Analizi (THYAO)**
```python
feeds = RSS_FEEDS['borsa_odakli'] + RSS_FEEDS['sektorel']['turizm']
```

**Senaryo 3: Tüm Kaynakları Kullan**
```python
from config import RSS_FEEDS_LIST
feeds = RSS_FEEDS_LIST
```
