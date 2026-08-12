# 📰 RSS Kaynak Filtreleme Özelliği

## 🎯 Genel Bakış

Streamlit arayüzüne dinamik RSS kaynak filtreleme özelliği eklendi. Kullanıcılar artık hangi kaynaklardan haber çekilmesini istediklerini seçebilirler.

## ✨ Yeni Özellikler

### 1. Kategori Bazlı Filtreleme

Kullanıcılar şu kategorilerden seçim yapabilir:

#### 📊 Borsa Odaklı Kaynaklar
- Bloomberg HT, Para Analiz, Foreks, BigPara, KAP vb.
- **Toplam**: 7 kaynak
- **Ne zaman kullanılmalı**: Her analiz için önerilir

#### 🏭 Sektörel Kaynaklar
Alt kategoriler:
- ✈️ **Turizm** (2 kaynak): THYAO, MAALT gibi hisseler için
- ⚡ **Enerji** (2 kaynak): TUPRS, PETKM gibi hisseler için
- 💻 **Teknoloji** (2 kaynak): TCELL, TTKOM gibi hisseler için

**Ne zaman kullanılmalı**: Analiz edilen hissenin sektörüne göre

#### 📈 Analiz ve Yorum Kaynakları
- Uzman analizleri ve piyasa yorumları
- **Toplam**: 4 kaynak
- **Ne zaman kullanılmalı**: Daha derinlemesine analiz için

#### 📰 Genel Ekonomi Haberleri
- Ana akım haber kaynakları
- **Toplam**: 8 kaynak
- **Ne zaman kullanılmalı**: Geniş kapsamlı analiz için

### 2. Dinamik Sektör Seçimi

"Sektörel Kaynaklar" işaretlendiğinde:
- Alt seçenekler otomatik olarak görünür
- İstediğiniz sektörleri seçebilirsiniz
- Birden fazla sektör seçilebilir

### 3. Canlı Kaynak Sayacı

Seçimlerinize göre:
- ✅ Yeşil: Kaynaklar seçildi (örn: "✅ 15 kaynak seçildi")
- ⚠️ Kırmızı: Hiç kaynak seçilmedi (uyarı)

## 🎨 Kullanıcı Arayüzü

### Gelişmiş Ayarlar Bölümü

```
🔧 Gelişmiş Ayarlar
├── Haber Filtreleme
│   ├── ☑ Hisse Özel Haber Filtresi
│   └── 🎚️ Minimum İlişki Skoru (0.0 - 1.0)
│
├── 📰 RSS Kaynak Seçimi
│   ├── ☑ 📊 Borsa Odaklı Kaynaklar (7 kaynak)
│   ├── ☑ 🏭 Sektörel Kaynaklar
│   │   ├── ☑ ✈️ Turizm (2 kaynak)
│   │   ├── ☑ ⚡ Enerji (2 kaynak)
│   │   └── ☑ 💻 Teknoloji (2 kaynak)
│   ├── ☑ 📈 Analiz ve Yorum Kaynakları (4 kaynak)
│   └── ☑ 📰 Genel Ekonomi Haberleri (8 kaynak)
│   
│   ✅ 25 kaynak seçildi
│
├── LSTM Parametreleri
└── Duygu Analizi
```

## 💡 Kullanım Örnekleri

### Örnek 1: THYAO (Turizm) Analizi

**Önerilen Seçim**:
- ✅ Borsa Odaklı
- ✅ Sektörel > Turizm
- ✅ Analiz ve Yorum
- ❌ Genel Ekonomi (opsiyonel)

**Sonuç**: ~13 kaynak, 30-50 ilgili haber

### Örnek 2: TUPRS (Enerji) Analizi

**Önerilen Seçim**:
- ✅ Borsa Odaklı
- ✅ Sektörel > Enerji
- ✅ Analiz ve Yorum
- ❌ Genel Ekonomi (opsiyonel)

**Sonuç**: ~13 kaynak, 20-35 ilgili haber

### Örnek 3: TCELL (Teknoloji) Analizi

**Önerilen Seçim**:
- ✅ Borsa Odaklı
- ✅ Sektörel > Teknoloji
- ✅ Analiz ve Yorum
- ❌ Genel Ekonomi (opsiyonel)

**Sonuç**: ~13 kaynak, 25-40 ilgili haber

### Örnek 4: Hızlı Test

**Önerilen Seçim**:
- ✅ Borsa Odaklı
- ❌ Sektörel
- ❌ Analiz ve Yorum
- ❌ Genel Ekonomi

**Sonuç**: 7 kaynak, hızlı analiz

### Örnek 5: Kapsamlı Analiz

**Önerilen Seçim**:
- ✅ Tümü işaretli

**Sonuç**: 25 kaynak, maksimum haber

## 🔧 Teknik Detaylar

### Kod Yapısı

**config.py**:
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
```

**app.py**:
```python
# Kullanıcı seçimlerini topla
selected_feeds = []
if use_borsa:
    selected_feeds.extend(RSS_FEEDS['borsa_odakli'])
if use_sektorel:
    for sector in selected_sectors:
        selected_feeds.extend(RSS_FEEDS['sektorel'][sector])
# ...

# Analizi çalıştır
run_analysis(..., selected_rss_feeds=selected_feeds)
```

### Yeni Parametreler

**run_analysis() fonksiyonu**:
- `selected_rss_feeds`: Kullanılacak RSS kaynakları listesi
- Varsayılan: `None` (tüm kaynaklar kullanılır)

## 📊 Performans

### Kaynak Sayısına Göre Süre

| Kaynak Sayısı | Haber Sayısı | Tahmini Süre |
|---------------|--------------|--------------|
| 7 (Sadece Borsa) | ~50-70 | 3-5 dakika |
| 13 (Borsa + 1 Sektör + Analiz) | ~80-120 | 5-8 dakika |
| 25 (Tümü) | ~150-200 | 8-12 dakika |

*Not: Süreler BERT kullanımı ve GPU varlığına göre değişir*

## ✅ Avantajlar

1. **Esneklik**: İhtiyaca göre kaynak seçimi
2. **Hız**: Daha az kaynak = daha hızlı analiz
3. **Odaklanma**: Sektörel odaklı haberler
4. **Kontrol**: Kullanıcı tam kontrol sahibi
5. **Verimlilik**: Gereksiz haberler filtrelenir

## 🚀 Gelecek Geliştirmeler

- [ ] Kaynak bazlı önceliklendirme
- [ ] Özel kaynak ekleme (UI'dan)
- [ ] Kaynak performans analizi
- [ ] Favori kombinasyonları kaydetme
- [ ] Otomatik sektör tespiti (hisse sembolüne göre)

## 📝 Notlar

- En az bir kaynak kategorisi seçilmelidir
- Sektörel kaynaklar işaretlenmişse en az bir sektör seçilmelidir
- Daha fazla kaynak = daha fazla haber = daha uzun süre
- Haber filtreleme ile birlikte kullanılabilir

## 🎓 İpuçları

1. **İlk Analiz**: Tüm kaynakları kullanarak başlayın
2. **Optimizasyon**: Sonuçlara göre kaynak sayısını azaltın
3. **Sektörel Odak**: Hissenin sektörünü mutlaka dahil edin
4. **Hızlı Test**: Sadece borsa odaklı kaynakları kullanın
5. **Kapsamlı Analiz**: Tüm kategorileri dahil edin

