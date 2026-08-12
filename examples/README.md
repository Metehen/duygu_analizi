# Örnekler (Examples)

Bu dizin, Türkçe Ekonomi Haberleri Duygu Analizi & Hisse Tahmin Sistemi için örnek dosyalar ve kullanım örnekleri içerir.

## Dosyalar

### Yapılandırma Örnekleri

- **config_example.py**: Detaylı yapılandırma örneği
  - Tüm sistem parametrelerini içerir
  - Her parametre için açıklamalar ve öneriler
  - Kendi `config.py` dosyanızı oluşturmak için kullanın

- **sample_rss_feeds.txt**: RSS besleme listesi
  - Türkçe ekonomi haberleri için RSS kaynakları
  - Güncel ve çalışan RSS URL'leri
  - Yeni kaynaklar eklemek için referans

- **sample_stock_symbols.txt**: Hisse sembol listesi
  - Borsa Istanbul (BIST) hisse sembolleri
  - Sektörlere göre gruplandırılmış
  - yfinance ile kullanım için `.IS` uzantılı

### Örnek Veriler

- **sample_news_data.csv**: Örnek haber verisi
  - 10 örnek ekonomi haberi
  - CSV formatında
  - Veri yapısını anlamak için kullanın

### Kullanım Örnekleri

- **quick_start_example.py**: Hızlı başlangıç scripti
  - Tam analiz akışını gösterir
  - Haber toplama → Duygu analizi → Model eğitimi → Tahmin
  - Yeni başlayanlar için ideal

- **demo_news_fetcher.py**: Haber toplama demo
  - RSS beslemelerinden haber çekme
  - Veri kaydetme ve yükleme
  - Hata yönetimi örnekleri

- **test_features_demo.py**: Özellik hazırlama demo
  - Teknik gösterge hesaplama
  - Veri birleştirme
  - Eğitim verisi hazırlama

## Kullanım

### 1. Hızlı Başlangıç

Tüm sistemi denemek için:

```bash
cd examples
py quick_start_example.py
```

Bu script:
- RSS'den haber toplar
- Duygu analizi yapar
- Hisse verisi çeker
- LSTM ve XGBoost modellerini eğitir
- Tahminler yapar ve performans gösterir

### 2. Yapılandırma Dosyası Oluşturma

```bash
# Örnek yapılandırmayı kopyalayın
cp examples/config_example.py config.py

# Kendi ihtiyaçlarınıza göre düzenleyin
nano config.py  # veya favori editörünüz
```

### 3. RSS Kaynakları Ekleme

`sample_rss_feeds.txt` dosyasındaki URL'leri `config.py` dosyanıza ekleyin:

```python
RSS_FEEDS = [
    'https://www.bloomberght.com/rss',
    'https://www.dunya.com/rss',
    # Daha fazla kaynak...
]
```

### 4. Hisse Sembolleri Seçme

`sample_stock_symbols.txt` dosyasından ilgilendiğiniz hisseleri seçin:

```python
DEFAULT_SYMBOLS = [
    'THYAO.IS',  # Türk Hava Yolları
    'GARAN.IS',  # Garanti Bankası
    # Daha fazla hisse...
]
```

## İpuçları

### Performans

- **İlk çalıştırma**: BERT modeli indirilecek (~500MB), 5-10 dakika sürebilir
- **Haber toplama**: 100 haber için ~30 saniye
- **Duygu analizi**: 
  - BERT: 100 haber için ~2-3 dakika (CPU), ~30 saniye (GPU)
  - Lexicon: 100 haber için ~5 saniye
- **Model eğitimi**:
  - LSTM: 50 epoch için ~5-10 dakika (CPU), ~1-2 dakika (GPU)
  - XGBoost: 100 ağaç için ~10-30 saniye

### Veri Miktarı

- **Minimum**: 90 gün veri önerilir
- **Optimal**: 6-12 ay veri
- **Maksimum**: 2 yıl (daha fazlası eğitimi yavaşlatabilir)

### Model Parametreleri

**Hızlı test için (quick_start_example.py gibi):**
```python
LSTM_CONFIG = {
    'lstm_units': [25, 25],
    'epochs': 5,
    'lookback': 30
}

XGBOOST_CONFIG = {
    'n_estimators': 50,
    'max_depth': 4
}
```

**Üretim kalitesi için:**
```python
LSTM_CONFIG = {
    'lstm_units': [50, 50],
    'epochs': 50,
    'lookback': 60
}

XGBOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6
}
```

## Sorun Giderme

### RSS Beslemeleri Çalışmıyor

- RSS URL'lerini tarayıcıda test edin
- Bazı siteler RSS'i kaldırmış olabilir
- `sample_rss_feeds.txt` dosyasını güncelleyin

### Hisse Verisi Çekilemiyor

- Hisse sembolünün doğru olduğundan emin olun (`.IS` uzantısı)
- İnternet bağlantınızı kontrol edin
- Yahoo Finance'in erişilebilir olduğunu doğrulayın

### BERT Modeli Yüklenemiyor

- İnternet bağlantınızı kontrol edin
- Disk alanınızın yeterli olduğundan emin olun (~1GB)
- Alternatif olarak `lexicon` yöntemini kullanın

### Bellek Hatası

- Batch size'ı küçültün
- Lookback değerini azaltın
- Daha az veri kullanın
- BERT yerine lexicon kullanın

## Daha Fazla Bilgi

- Ana README: `../README.md`
- Dokümantasyon: `.kiro/specs/turkish-sentiment-stock-prediction/`
- Testler: `../tests/`

## Katkıda Bulunma

Yeni örnekler eklemek veya mevcut örnekleri iyileştirmek için:

1. Örnek dosyanızı oluşturun
2. Bu README'ye ekleyin
3. Pull request gönderin

## Lisans

Bu örnekler ana proje ile aynı lisans altındadır.
