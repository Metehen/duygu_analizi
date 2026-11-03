# Türkçe Ekonomi Haberleri Duygu Analizi & Hisse Tahmin Sistemi

Türkçe ekonomi haberlerinden duygu analizi yaparak hisse senedi fiyat tahminleri üreten deneysel bir finansal yapay zeka sistemi.

## Genel Bakış

Bu proje, Türkçe ekonomi haberlerini toplayarak duygu analizi yapar ve bu duygu skorlarını teknik göstergelerle birleştirerek hisse senedi fiyat tahminleri üretir. Sistem iki farklı makine öğrenmesi yaklaşımı kullanır:

- **LSTM (Long Short-Term Memory)**: Zaman serisi tahmini için fiyat tahminleri
- **XGBoost**: Yön tahmini (yukarı/aşağı) ve özellik önem analizi

## Özellikler

- 📰 RSS beslemelerinden Türkçe ekonomi haberleri toplama
- 🧠 Türkçe BERT modeli ile duygu analizi
- 📊 Teknik gösterge hesaplama (EMA, RSI, MACD)
- 🤖 LSTM ve XGBoost modelleri ile tahmin
- 📈 Basit geriye dönük test (backtesting)
- 🖥️ Streamlit tabanlı web arayüzü
- 📉 İnteraktif görselleştirmeler

## Proje Yapısı

```
turkish_sentiment_stock/
├── data/               # Veri dosyaları (CSV, JSON)
├── models/             # Eğitilmiş model dosyaları
├── src/                # Kaynak kod
│   ├── news_fetcher.py     # Haber toplama modülü
│   ├── sentiment.py        # Duygu analizi modülü
│   ├── features.py         # Özellik hazırlama modülü
│   ├── model_lstm.py       # LSTM model modülü
│   ├── model_xgboost.py    # XGBoost model modülü
│   ├── backtest.py         # Geriye dönük test modülü
│   └── app.py              # Streamlit web arayüzü
├── tests/              # Test dosyaları
├── config.py           # Yapılandırma dosyası
├── requirements.txt    # Python bağımlılıkları
└── README.md           # Bu dosya
```

## Kurulum

### Gereksinimler

- Python 3.10 veya üzeri
- pip (Python paket yöneticisi)
- 2GB+ RAM (BERT modeli için)
- İnternet bağlantısı (ilk çalıştırmada model indirme için)

### Adımlar

1. Repoyu klonlayın veya indirin:
```bash
git clone <repo-url>
cd turkish_sentiment_stock
```

2. Sanal ortam oluşturun (önerilir):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

4. Gerekli dizinleri oluşturun (otomatik oluşturulmazsa):
```bash
mkdir data models
```

### İlk Çalıştırma

İlk çalıştırmada Türkçe BERT modeli otomatik olarak indirilecektir (~500MB). Bu işlem birkaç dakika sürebilir:

```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased'); AutoModel.from_pretrained('dbmdz/bert-base-turkish-cased')"
```

### Sorun Giderme

**TensorFlow/Keras Kurulum Sorunları:**
```bash
pip install tensorflow==2.15.0
```

**XGBoost Kurulum Sorunları:**
```bash
pip install xgboost --upgrade
```

**Streamlit Çalışmıyor:**
```bash
streamlit --version
# Eğer hata alırsanız:
pip install streamlit --upgrade
```

## Kullanım

### Web Arayüzü ile Kullanım

Streamlit uygulamasını başlatın:
```bash
streamlit run src/app.py
```

Tarayıcınızda açılan arayüzde:
1. Hisse sembolünü girin (örn: THYAO.IS, GARAN.IS, ISCTR.IS)
2. Tarih aralığını seçin (en az 90 gün önerilir)
3. Model parametrelerini ayarlayın (opsiyonel)
4. "Analiz Et" düğmesine tıklayın
5. Sonuçları görselleştirmelerde inceleyin:
   - Fiyat ve tahmin grafiği
   - Duygu skorları zaman serisi
   - Performans metrikleri (RMSE, MAE, R2)
   - Özellik önem analizi

### Komut Satırı ile Kullanım

#### Örnek 1: Haber Toplama ve Duygu Analizi

```python
from src.news_fetcher import fetch_rss_news, save_news
from src.sentiment import SentimentAnalyzer

# RSS kaynaklarından haber topla
rss_urls = [
    'https://www.bloomberght.com/rss',
    'https://www.dunya.com/rss'
]
news_df = fetch_rss_news(rss_urls, max_articles=100)

# Haberleri kaydet
save_news(news_df, 'data/news.csv', format='csv')

# Duygu analizi yap
analyzer = SentimentAnalyzer(method='bert')  # veya 'lexicon'
sentiment_df = analyzer.analyze_batch(news_df['content'].tolist())

print(f"Toplam haber: {len(news_df)}")
print(f"Ortalama duygu skoru: {sentiment_df['compound'].mean():.2f}")
```

#### Örnek 2: Hisse Verisi ve Teknik Göstergeler

```python
from src.features import (
    fetch_stock_data,
    calculate_technical_indicators,
    merge_sentiment_with_prices,
    prepare_training_data
)

# Hisse verisi çek
stock_df = fetch_stock_data('THYAO.IS', '2023-01-01', '2024-01-01')

# Teknik göstergeler hesapla
stock_df = calculate_technical_indicators(stock_df)

# Duygu skorlarıyla birleştir
merged_df = merge_sentiment_with_prices(stock_df, sentiment_df)

# Eğitim verisi hazırla
X, y, scaler = prepare_training_data(merged_df, lookback=60)

print(f"Eğitim verisi boyutu: X={X.shape}, y={y.shape}")
```

#### Örnek 3: LSTM Model Eğitimi ve Tahmin

```python
from src.model_lstm import build_lstm_model, train_lstm_model, predict_lstm

# Model oluştur
model = build_lstm_model(
    input_shape=(60, X.shape[2]),
    lstm_units=[50, 50],
    dropout=0.2
)

# Train-test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Model eğit
history = train_lstm_model(
    model,
    X_train, y_train,
    X_test, y_test,
    epochs=50,
    batch_size=32
)

# Tahmin yap
predictions = predict_lstm(model, X_test, scaler)

# Model kaydet
model.save('models/lstm_model.keras')
```

#### Örnek 4: XGBoost Model Eğitimi

```python
from src.model_xgboost import (
    prepare_classification_data,
    train_xgboost_model,
    get_feature_importance,
    predict_direction
)

# Sınıflandırma verisi hazırla
X_class, y_class = prepare_classification_data(merged_df)

# Train-test split
split_idx = int(len(X_class) * 0.8)
X_train = X_class.iloc[:split_idx]
y_train = y_class.iloc[:split_idx]
X_test = X_class.iloc[split_idx:]
y_test = y_class.iloc[split_idx:]

# Model eğit
xgb_model = train_xgboost_model(X_train, y_train)

# Özellik önem analizi
importance_df = get_feature_importance(xgb_model)
print("En önemli 5 özellik:")
print(importance_df.head())

# Tahmin yap
predictions, probabilities = predict_direction(xgb_model, X_test)
```

#### Örnek 5: Geriye Dönük Test

```python
from src.backtest import simple_backtest, calculate_metrics

# Basit geriye dönük test
backtest_results = simple_backtest(
    predictions=predictions,
    actual_prices=y_test,
    initial_capital=10000.0
)

print(f"Toplam Getiri: {backtest_results['total_return']:.2f}%")
print(f"Doğruluk Oranı: {backtest_results['accuracy']:.2f}%")
print(f"İşlem Sayısı: {backtest_results['num_trades']}")

# Performans metrikleri
metrics = calculate_metrics(y_test, predictions)
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")
print(f"R2 Score: {metrics['r2']:.4f}")
```

## Yapılandırma

`config.py` dosyasında tüm sistem parametrelerini özelleştirebilirsiniz:

### RSS Kaynakları
```python
RSS_FEEDS = [
    'https://www.bloomberght.com/rss',
    'https://www.dunya.com/rss',
    # Daha fazla kaynak ekleyebilirsiniz
]
```

### Hisse Sembolleri
```python
DEFAULT_SYMBOLS = [
    'THYAO.IS',  # Türk Hava Yolları
    'GARAN.IS',  # Garanti Bankası
    'ISCTR.IS',  # İş Bankası
    # Borsa Istanbul sembolleri .IS uzantısı ile
]
```

### LSTM Model Parametreleri
```python
LSTM_CONFIG = {
    'lstm_units': [50, 50],  # Her katmandaki LSTM birim sayısı
    'dropout': 0.2,          # Dropout oranı (overfitting önleme)
    'epochs': 50,            # Eğitim epoch sayısı
    'batch_size': 32,        # Batch boyutu
    'lookback': 60           # Geçmiş gün sayısı (pencere boyutu)
}
```

### XGBoost Model Parametreleri
```python
XGBOOST_CONFIG = {
    'max_depth': 6,          # Ağaç derinliği
    'learning_rate': 0.1,    # Öğrenme oranı
    'n_estimators': 100,     # Ağaç sayısı
    'random_state': 42       # Rastgelelik seed'i
}
```

### Duygu Analizi Ayarları
```python
SENTIMENT_METHOD = 'bert'  # 'bert' veya 'lexicon'
BERT_MODEL = 'dbmdz/bert-base-turkish-cased'  # Türkçe BERT modeli
```

### Teknik Gösterge Parametreleri
```python
TECHNICAL_INDICATORS = {
    'ema_periods': [10, 20, 50],  # EMA periyotları
    'rsi_period': 14,              # RSI periyodu
    'macd_params': (12, 26, 9),    # MACD parametreleri
    'volume_ma_period': 20         # Hacim MA periyodu
}
```

### Veri Yolları
```python
DATA_DIR = 'data/'
MODELS_DIR = 'models/'
NEWS_FILE = 'data/news.csv'
```

### Tarih Aralıkları
```python
DEFAULT_START_DATE = '2023-01-01'
DEFAULT_END_DATE = '2024-01-01'
```

### Yapılandırma İpuçları

- **Lookback değeri**: Daha büyük değerler daha fazla geçmiş veri kullanır ama eğitim yavaşlar (30-90 arası önerilir)
- **LSTM units**: Daha büyük değerler model kapasitesini artırır ama overfitting riski artar
- **Dropout**: 0.2-0.3 arası değerler genellikle iyidir
- **XGBoost max_depth**: 3-10 arası değerler deneyin
- **Sentiment method**: 'bert' daha doğru ama yavaş, 'lexicon' hızlı ama basit

## Test

Testleri çalıştırmak için:

```bash
# Tüm testler
pytest tests/

# Belirli bir test dosyası
pytest tests/test_news_fetcher.py

# Özellik tabanlı testler
pytest tests/ -k property
```

## Bilinen Sınırlamaları

### Teknik Sınırlamalar

- **Deneysel Prototip**: Bu sistem üretime hazır değildir, sadece araştırma ve eğitim amaçlıdır
- **Gerçek Zamanlı Veri Yok**: Sistem gerçek zamanlı veri akışı desteklemez, geçmiş verilerle çalışır
- **Sınırlı Hata Yönetimi**: Kapsamlı hata yönetimi ve logging eksiktir
- **Tek Kullanıcı**: Çoklu kullanıcı desteği yoktur
- **Veritabanı Yok**: Veriler CSV/JSON dosyalarında saklanır
- **Güvenlik**: Güvenlik önlemleri minimal düzeydedir

### Veri Sınırlamaları

- **Veri Kalitesi**: Model performansı haber kalitesine ve RSS kaynaklarının güvenilirliğine bağlıdır
- **Veri Miktarı**: Küçük veri setleriyle (< 100 haber) performans düşüktür
- **Tarih Aralığı**: En az 90 günlük veri önerilir
- **Eksik Veri**: Tatil günleri ve hafta sonları için veri eksik olabilir
- **RSS Kaynakları**: Bazı RSS beslemeleri zaman zaman erişilemez olabilir

### Model Sınırlamaları

- **Tahmin Doğruluğu**: Finansal piyasalar tahmin edilmesi zor sistemlerdir, yüksek doğruluk beklemeyin
- **Overfitting Riski**: Küçük veri setlerinde model overfitting yapabilir
- **BERT Model Boyutu**: İlk çalıştırmada ~500MB model indirilir
- **Eğitim Süresi**: LSTM eğitimi CPU'da yavaş olabilir (GPU önerilir)
- **Bellek Kullanımı**: BERT modeli 2GB+ RAM gerektirir

### Kullanım Sınırlamaları

- **Finansal Tavsiye Değildir**: Bu sistem finansal tavsiye vermez, yatırım kararlarınızı kendi araştırmanıza dayanarak alın
- **Geçmiş Performans**: Geçmiş performans gelecek performansı garanti etmez
- **Piyasa Koşulları**: Sistem ani piyasa değişimlerini tahmin edemez
- **Dış Faktörler**: Politik olaylar, doğal afetler gibi dış faktörler modelde yer almaz

### Bilinen Hatalar

- **NaN Değerler**: Bazı teknik göstergeler ilk günlerde NaN değer üretebilir (otomatik doldurulur)
- **Tarih Uyumsuzluğu**: Haber tarihleri ve hisse fiyat tarihleri tam eşleşmeyebilir
- **Encoding Sorunları**: Bazı RSS beslemelerinde Türkçe karakter sorunları olabilir
- **Streamlit Yenileme**: Streamlit arayüzü bazen manuel yenileme gerektirebilir

### Performans Notları

- **İlk Çalıştırma**: BERT modeli indirme nedeniyle 5-10 dakika sürebilir
- **Haber Toplama**: 100 haber için ~30 saniye
- **Duygu Analizi**: BERT ile 100 haber için ~2-3 dakika (CPU), ~30 saniye (GPU)
- **LSTM Eğitimi**: 50 epoch için ~5-10 dakika (CPU), ~1-2 dakika (GPU)
- **XGBoost Eğitimi**: 100 ağaç için ~10-30 saniye (CPU)

## Geliştirme Notları

### Veri Kaynakları

- RSS beslemeleri ücretsiz ve herkese açıktır
- yfinance Yahoo Finance'den veri çeker
- Borsa Istanbul sembolleri `.IS` uzantısı ile kullanılır

### Model Eğitimi

- LSTM modeli eğitimi GPU ile hızlandırılabilir
- XGBoost CPU'da yeterince hızlıdır
- Modeller `models/` dizinine kaydedilir

### Performans İpuçları

- İlk çalıştırmada BERT modeli indirilir (~500MB)
- Küçük veri setleriyle başlayın (birkaç yüz haber)
- Lookback parametresini azaltarak eğitim hızlandırılabilir

## Katkıda Bulunma

Bu deneysel bir projedir. Öneriler ve iyileştirmeler için:
1. Issue açın
2. Pull request gönderin
3. Dokümantasyonu güncelleyin

## Lisans

Bu proje [GNU AGPL v3](LICENSE) lisansı altında dağıtılmaktadır.

Copyright (C) 2026 Metehen

Bu lisans:
- Kodu kullanmanıza, değiştirmenize ve dağıtmanıza izin verir
- Türev çalışmaların da aynı lisans altında açık kaynak olmasını zorunlu kılar
- Web servisi olarak kullanımda bile kaynak kodu paylaşmayı gerektirir
- Atıf (copyright notice) korunmalıdır

## İletişim

Sorular ve geri bildirimler için issue açabilirsiniz.

## Teşekkürler

- Hugging Face Transformers
- Yahoo Finance (yfinance)
- Streamlit
- Türkçe NLP topluluğu

---

**Uyarı**: Bu sistem finansal tavsiye vermez. Yatırım kararlarınızı kendi araştırmanıza dayanarak alın.
