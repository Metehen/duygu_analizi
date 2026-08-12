# Streamlit Web Arayüzü Kullanım Kılavuzu

## Kurulum

Öncelikle tüm bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

## Uygulamayı Çalıştırma

Streamlit uygulamasını başlatmak için:

```bash
py -m streamlit run src/app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılacaktır.

## Kullanım

### 1. Parametreleri Ayarlama

Sol taraftaki sidebar'dan:
- **Hisse Sembolü**: Listeden seçin veya "Manuel Gir..." seçeneğiyle kendi sembolünüzü yazın (örn: THYAO.IS)
- **Tarih Aralığı**: Analiz için başlangıç ve bitiş tarihlerini seçin
- **Gelişmiş Ayarlar**: 
  - **Haber Filtreleme**: 
    - Hisse özel haber filtresi (aktif/pasif)
    - Minimum ilişki skoru (0.0 - 1.0)
  - **RSS Kaynak Seçimi**: Hangi kaynaklardan haber çekilsin?
    - 📊 Borsa Odaklı Kaynaklar (Bloomberg HT, Para Analiz, Foreks vb.)
    - 🏭 Sektörel Kaynaklar (Turizm, Enerji, Teknoloji)
    - 📈 Analiz ve Yorum Kaynakları
    - 📰 Genel Ekonomi Haberleri
  - **LSTM Parametreleri**: Epoch sayısı ve lookback penceresi
  - **Duygu Analizi**: BERT veya Lexicon yöntemi

### 2. Analizi Başlatma

"🚀 Analiz Et" düğmesine tıklayın. Uygulama şu adımları gerçekleştirecektir:

1. **Haber Toplama**: RSS beslemelerinden Türkçe ekonomi haberleri çeker
2. **Duygu Analizi**: Haberlere duygu analizi uygular (BERT veya Lexicon yöntemi)
3. **Özellik Hazırlama**: Hisse fiyat verilerini çeker ve teknik göstergeler hesaplar
4. **Model Eğitimi**: LSTM ve XGBoost modellerini eğitir
5. **Geriye Dönük Test**: Modellerin performansını değerlendirir
6. **Görselleştirme**: Sonuçları grafikler ve metriklerle gösterir

### 3. Sonuçları İnceleme

Analiz tamamlandığında şu görselleştirmeleri göreceksiniz:

- **Fiyat ve Tahminler Grafiği**: Gerçek fiyatlar vs LSTM tahminleri
- **Duygu Skorları Grafiği**: Günlük pozitif/negatif/compound skorlar
- **Model Performans Metrikleri**: RMSE, MAE, R², doğruluk oranı
- **Özellik Önemleri**: XGBoost modelinin en önemli bulduğu özellikler

## Özellikler

### Haber Kaynakları

Uygulama artık 25 farklı RSS kaynağından haber çekebilir ve bunları kategorilere ayırır:

**📊 Borsa Odaklı (7 kaynak)**:
- Bloomberg HT, Para Analiz, Investing.com, Trading Economics, Foreks, BigPara, KAP

**🏭 Sektörel Kaynaklar (6 kaynak)**:
- ✈️ Turizm (2): Turizm Güncel, Turizm Haberleri
- ⚡ Enerji (2): Enerji Gündem, Enerji Portalı
- 💻 Teknoloji (2): Teknoloji Turu, Teknoloji Oku

**📈 Analiz ve Yorum (4 kaynak)**:
- Analiz Market, Borsa Gündem, Dünya, Ekonomim

**📰 Genel Ekonomi (8 kaynak)**:
- Mynet, Habertürk, Hürriyet, Sabah, Milliyet, AA, NTV, CNN Türk

#### RSS Kaynak Seçimi

Gelişmiş Ayarlar'dan hangi kaynak kategorilerinin kullanılacağını seçebilirsiniz:

1. **Tüm Kaynaklar**: Tüm kategorileri işaretleyin (25 kaynak)
2. **Sadece Borsa Odaklı**: Finansal piyasalara odaklı haberler
3. **Sektörel Odaklı**: Örneğin THYAO için sadece Turizm sektörü
4. **Özel Kombinasyon**: İstediğiniz kategorileri seçin

**Örnek Kullanım**:
- THYAO analizi için: Borsa Odaklı + Turizm + Analiz/Yorum
- TUPRS analizi için: Borsa Odaklı + Enerji + Analiz/Yorum
- TCELL analizi için: Borsa Odaklı + Teknoloji + Analiz/Yorum

Daha fazla kaynak eklemek için `config.py` dosyasındaki `RSS_FEEDS` dictionary'sini düzenleyin.

### Duygu Analizi Yöntemleri

1. **BERT** (Varsayılan):
   - Daha doğru sonuçlar
   - Daha yavaş işlem
   - GPU desteği varsa hızlanır

2. **Lexicon** (Sözlük Tabanlı):
   - Hızlı işlem
   - Basit kelime sayımı
   - GPU gerektirmez

### Model Parametreleri

**LSTM Parametreleri**:
- Epoch Sayısı: Model eğitim iterasyon sayısı (varsayılan: 50)
- Lookback Penceresi: Geçmiş kaç günün verisi kullanılacak (varsayılan: 60)

**XGBoost Parametreleri**:
- Otomatik olarak `config.py`'den alınır
- Yön tahmini için binary classification kullanır

## Sorun Giderme

### Haber Çekilemiyor

- İnternet bağlantınızı kontrol edin
- RSS kaynaklarının erişilebilir olduğundan emin olun
- Bazı kaynaklar geçici olarak erişilemez olabilir

### BERT Modeli Yüklenemiyor

- `transformers` ve `torch` kütüphanelerinin yüklü olduğundan emin olun
- İlk çalıştırmada model indirilir, bu biraz zaman alabilir
- Alternatif olarak "Lexicon" yöntemini kullanın

### Hisse Verisi Çekilemiyor

- Hisse sembolünün doğru olduğundan emin olun (Borsa Istanbul için `.IS` eki gerekli)
- Tarih aralığının geçerli olduğunu kontrol edin
- `yfinance` API'sinin erişilebilir olduğundan emin olun

### Yetersiz Veri Hatası

- Daha geniş bir tarih aralığı seçin
- Lookback penceresini küçültün
- En az 100 günlük veri önerilir

### Hiç Haber Çekilemedi

- En az bir RSS kaynak kategorisi seçili olduğundan emin olun
- Farklı kaynak kombinasyonları deneyin
- Bazı RSS kaynakları geçici olarak erişilemez olabilir
- Haber filtresini devre dışı bırakın veya minimum ilişki skorunu düşürün

### Seçilen Hisse İçin İlgili Haber Bulunamadı

- Minimum ilişki skorunu düşürün (0.2 → 0.1 veya 0.0)
- Daha fazla RSS kaynak kategorisi seçin (özellikle ilgili sektörü dahil edin)
- Haber filtresini tamamen devre dışı bırakın

## Performans İpuçları

1. **İlk Çalıştırma**: BERT modeli ilk kez indirilirken zaman alabilir
2. **GPU Kullanımı**: BERT için GPU varsa otomatik olarak kullanılır
3. **Veri Miktarı**: Daha fazla veri daha iyi sonuçlar verir ama işlem süresi artar
4. **Epoch Sayısı**: Daha fazla epoch daha iyi eğitim ama daha uzun süre

## Örnek Kullanım Senaryoları

### Senaryo 1: Turizm Hissesi Analizi (THYAO)

1. **Hisse**: THYAO.IS (Türk Hava Yolları)
2. **Tarih**: 2023-01-01 - 2024-01-01
3. **RSS Kaynakları**: 
   - ✅ Borsa Odaklı
   - ✅ Sektörel > Turizm
   - ✅ Analiz ve Yorum
   - ❌ Genel Ekonomi (opsiyonel)
4. **LSTM Epochs**: 50
5. **Lookback**: 60
6. **Duygu Analizi**: BERT

**Beklenen Süre**: 5-10 dakika (GPU ile daha hızlı)
**Beklenen Haber Sayısı**: ~30-50 ilgili haber

### Senaryo 2: Enerji Hissesi Analizi (TUPRS)

1. **Hisse**: TUPRS.IS (Tüpraş)
2. **Tarih**: 2023-01-01 - 2024-01-01
3. **RSS Kaynakları**:
   - ✅ Borsa Odaklı
   - ✅ Sektörel > Enerji
   - ✅ Analiz ve Yorum
4. **Haber Filtresi**: Aktif, Minimum İlişki: 0.2
5. **Duygu Analizi**: BERT

**Beklenen Haber Sayısı**: ~20-35 ilgili haber

### Senaryo 3: Hızlı Test (Lexicon)

1. **Hisse**: GARAN.IS (Garanti Bankası)
2. **Tarih**: Son 6 ay
3. **RSS Kaynakları**: Sadece Borsa Odaklı
4. **LSTM Epochs**: 30 (daha hızlı)
5. **Lookback**: 30
6. **Duygu Analizi**: Lexicon (çok daha hızlı)

**Beklenen Süre**: 2-3 dakika

## Notlar

- Bu bir deneysel sistemdir, finansal tavsiye değildir
- Sonuçlar geçmiş verilere dayanır, gelecek garantisi vermez
- Gerçek yatırım kararları için profesyonel danışmanlık alın
