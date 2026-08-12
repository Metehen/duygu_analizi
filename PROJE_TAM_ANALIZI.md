# PROJE TAM ANALİZİ
# Türkçe Ekonomi Haberleri Duygu Analizi & Hisse Senedi Tahmin Sistemi

---

## 1. PROJE GENEL BAKIŞI

Bu proje, Türkçe ekonomi haberlerini RSS kaynaklarından toplayarak duygu analizi yapan,
bu duygu skorlarını teknik göstergelerle birleştiren ve makine öğrenmesi modelleriyle
Borsa Istanbul hisse senetleri için fiyat ve yön tahminleri üreten deneysel bir finansal
yapay zeka sistemidir.

Proje tamamen araştırma ve eğitim amaçlıdır. Finansal tavsiye niteliği taşımaz.

---

## 2. TEKNOLOJİ YIĞINI (TECH STACK)

### Programlama Dili
- Python 3.10+

### Temel Kütüphaneler
- pandas >= 2.0.0          (Veri işleme)
- numpy >= 1.24.0           (Matematiksel işlemler)
- feedparser >= 6.0.10      (RSS besleme okuma)
- requests >= 2.31.0        (HTTP istekleri)
- beautifulsoup4 >= 4.12.0  (HTML parse etme)
- yfinance >= 0.2.28        (Yahoo Finance hisse verisi)
- scikit-learn >= 1.3.0     (ML yardımcı araçlar)
- xgboost >= 2.0.0          (XGBoost sınıflandırıcı)
- tensorflow >= 2.13.0      (LSTM model)
- transformers >= 4.30.0    (Türkçe BERT modeli)
- torch >= 2.0.0            (PyTorch - BERT arka ucu)
- streamlit >= 1.25.0       (Web arayüzü)
- plotly >= 5.15.0          (İnteraktif görselleştirme)
- matplotlib >= 3.7.0       (Statik grafik)
- pytest >= 7.4.0           (Test framework)
- hypothesis >= 6.82.0      (Özellik tabanlı test)

### Türkçe NLP Modeli
- dbmdz/bert-base-turkish-cased (Ana model, ~500MB)
- savasy/bert-base-turkish-sentiment-cased (Alternatif)

---

## 3. PROJE KLASÖR YAPISI

```
turkish_sentiment_stock/
│
├── src/                         # Kaynak kod modülleri
│   ├── __init__.py
│   ├── app.py                   # Ana Streamlit web arayüzü (1323 satır)
│   ├── news_page.py             # Haber okuma sayfası
│   ├── news_fetcher.py          # RSS haber toplama modülü
│   ├── news_filter.py           # Haber filtreleme modülü
│   ├── sentiment.py             # Duygu analizi modülü
│   ├── features.py              # Özellik mühendisliği modülü
│   ├── model_lstm.py            # LSTM model modülü
│   ├── model_xgboost.py         # XGBoost model modülü
│   ├── backtest.py              # Geriye dönük test modülü
│   └── model_evaluation.py     # Model değerlendirme modülü
│
├── tests/                       # Test dosyaları
│   ├── test_app_basic.py        # Uygulama birim testleri
│   ├── test_app_properties.py   # Uygulama özellik testleri
│   ├── test_backtest_basic.py
│   ├── test_backtest_properties.py
│   ├── test_features_basic.py
│   ├── test_features_properties.py
│   ├── test_integration_e2e.py  # End-to-end entegrasyon testleri
│   ├── test_model_evaluation.py
│   ├── test_model_evaluation_properties.py
│   ├── test_model_lstm_basic.py
│   ├── test_model_lstm_properties.py
│   ├── test_model_xgboost_basic.py
│   ├── test_model_xgboost_properties.py
│   ├── test_news_fetcher_basic.py
│   ├── test_news_fetcher_properties.py
│   ├── test_sentiment_basic.py
│   └── test_sentiment_properties.py
│
├── models/                      # Eğitilmiş model dosyaları
│   └── training_history/        # Model checkpoint ve geçmişleri
│       ├── lstm_THYAO_*.keras   # THYAO için LSTM modeli
│       ├── lstm_SASA_*.keras    # SASA için LSTM modeli
│       ├── *_history.csv        # Eğitim geçmişi CSV
│       └── *_report.json       # Performans raporu JSON
│
├── data/                        # Veri dosyaları (.gitkeep)
├── examples/                    # Örnek kullanım dosyaları
│   ├── config_example.py
│   ├── demo_news_fetcher.py
│   ├── quick_start_example.py
│   ├── test_features_demo.py
│   ├── sample_news_data.csv
│   ├── sample_rss_feeds.txt
│   ├── sample_stock_symbols.txt
│   └── README.md
│
├── .kiro/                       # Kiro IDE ayarları
│   ├── hooks/                   # Agent hook'ları
│   └── specs/                   # Spec dosyaları
│
├── config.py                    # Merkezi yapılandırma
├── requirements.txt             # Python bağımlılıkları
├── README.md                    # Proje açıklaması
├── STREAMLIT_USAGE.md          # Streamlit kullanım kılavuzu
├── RSS_KAYNAKLARI_REHBERI.md   # RSS kaynakları rehberi
├── RSS_FILTRELEME_OZELLIGI.md  # RSS filtreleme özellikleri
├── HABER_FILTRELEME_KILAVUZU.md # Haber filtreleme kılavuzu
├── HABER_CEKME_IYILESTIRMESI.md # Haber çekme iyileştirme notları
└── DEGISIKLIK_OZETI.md         # Değişiklik özeti
```

---

## 4. MERKEZI YAPILANDIRMA (config.py)

### 4.1 RSS Kaynakları - Gruplandırılmış Yapı

Toplam 24 RSS kaynağı 4 ana kategoride düzenlenmiştir:

**Borsa Odaklı (6 aktif kaynak):**
- Bloomberg HT: https://www.bloomberght.com/rss
- Para Analiz: https://www.paraanaliz.com/feed/
- Investing.com Türkiye: https://www.investing.com/rss/news_285.rss
- Trading Economics: https://tr.tradingeconomics.com/rss/turkey.xml
- Foreks: https://www.foreks.com/rss
- BigPara: https://www.bigpara.com/rss
- KAP (devre dışı - RSS formatı hatalı): https://www.kap.org.tr/tr/rss

**Sektörel Kaynaklar:**
- Turizm (2): turizmguncel.com, turizmhaberleri.com
- Enerji (2): enerjigundem.com, enerjiportali.com
- Teknoloji (2): teknolojituru.com, teknolojioku.com

**Analiz ve Yorum (4):**
- analizmarket.com, borsagundem.com, dunya.com, ekonomim.com

**Genel Ekonomi (8):**
- mynet.com, haberturk.com, hurriyet.com.tr, sabah.com.tr,
  milliyet.com.tr, aa.com.tr, ntv.com.tr, cnnturk.com

### 4.2 Varsayılan Hisse Sembolleri
- THYAO.IS  (Türk Hava Yolları)
- GARAN.IS  (Garanti Bankası)
- ISCTR.IS  (İş Bankası)
- AKBNK.IS  (Akbank)
- TUPRS.IS  (Tüpraş)

### 4.3 LSTM Model Parametreleri
- lstm_units: [50, 50]         (Her katmandaki birim sayısı)
- dropout: 0.2                  (Overfitting önleme)
- epochs: 50                    (Eğitim iterasyonu)
- batch_size: 32
- lookback: 60                  (Geçmiş 60 günlük pencere)
- validation_split: 0.2
- early_stopping_patience: 10

### 4.4 XGBoost Model Parametreleri
- max_depth: 6
- learning_rate: 0.1
- n_estimators: 100
- objective: binary:logistic
- eval_metric: logloss
- random_state: 42

### 4.5 Duygu Analizi Ayarları
- Varsayılan yöntem: 'bert'
- BERT model: dbmdz/bert-base-turkish-cased
- Alternatif: savasy/bert-base-turkish-sentiment-cased

### 4.6 Teknik Gösterge Parametreleri
- EMA periyotları: [10, 20, 50]
- RSI periyodu: 14
- MACD fast/slow/signal: 12/26/9
- Hacim MA periyodu: 20

### 4.7 Geriye Dönük Test Ayarları
- initial_capital: 10,000 TL
- transaction_cost: %0.1
- test_split: Son %20

### 4.8 Haber Toplama Ayarları
- max_articles: 200
- timeout: 30 saniye
- retry_attempts: 3

### 4.9 Veri Dizinleri
- DATA_DIR: data/
- MODELS_DIR: models/
- NEWS_FILE: data/news.csv
- SENTIMENT_FILE: data/sentiment.csv
- DEFAULT_START_DATE: 2023-01-01
- DEFAULT_END_DATE: 2024-01-01

---

## 5. KAYNAK KOD MODÜLLERİ - DETAYLI ANALİZ

### 5.1 src/app.py — Ana Streamlit Web Arayüzü (1323 satır)

Ana uygulama dosyasıdır. Tüm kullanıcı arayüzünü ve analiz orkestrasyon mantığını içerir.

#### Sayfa Yapısı (Session State ile Yönetilen)
Uygulama 3 ana ekrandan oluşur:
1. **Ana Menü (_display_home)**: İki modül arasında geçiş
2. **Tahmin Uygulaması (_display_tahmin_app)**: ML analiz akışı
3. **Haber Okuma (news_page.display_news_page)**: Haber görüntüleme

#### Ana Menü Özellikleri
- "Hisse Tahmin Uygulamasına Git" butonu
- "Haber Okumasına Git" butonu
- Session state ile aktif sayfa takibi

#### Tahmin Uygulaması Sidebar Parametreleri
- **Hisse Sembolü Seçimi**: Listeden seçim veya manuel giriş
  - Varsayılan seçenekler: THYAO.IS, GARAN.IS, ISCTR.IS, AKBNK.IS, TUPRS.IS
  - "Manuel Gir..." seçeneğiyle özel sembol girişi
  - .IS uzantısı otomatik ekleniyor (yoksa)
- **Tarih Aralığı**: Başlangıç ve bitiş tarihi (2020'den bugüne)
- **Gelişmiş Ayarlar** (expander içinde):
  - Haber Filtreleme: aktif/pasif checkbox
  - Minimum İlişki Skoru: 0.0–1.0 slider
  - RSS Kaynak Seçimi: kategori bazlı checkbox'lar
    - Borsa Odaklı (kaynak sayısı gösterilir)
    - Sektörel (Turizm, Enerji, Teknoloji alt seçenekleri)
    - Analiz ve Yorum
    - Genel Ekonomi
  - Canlı seçilen kaynak sayacı (yeşil/kırmızı)
  - LSTM Parametreleri: epoch (10-200) ve lookback (10-120)
  - Duygu Analizi: bert/lexicon dropdown

#### Analiz Akışı (run_analysis fonksiyonu)
6 aşamalı tam analiz süreci:

**Aşama 1 - Haber Toplama:**
- Seçilen RSS kaynaklarından haber çekme
- İsteğe bağlı anahtar kelime filtresi (hisse + sektör kelimeleri)
- Her kaynaktan 100 haber, toplamda 200 haber limiti
- İki aşamalı filtreleme:
  1. Geniş filtre (haber çekme aşamasında)
  2. Hassas filtre (relevance score bazlı)
- İlişki özeti metrikleri: Yüksek/Orta/Düşük ilişkili haber sayısı
- Örnek haberler expander'da gösterimi

**Aşama 2 - Duygu Analizi:**
- SentimentAnalyzer ile batch analiz
- BERT veya Lexicon yöntemi
- Sonuç: positive, negative, neutral, compound skorları
- İstatistikler metric widget'larıyla gösterilir

**Aşama 3 - Özellik Hazırlama:**
- yfinance ile hisse fiyat verisi çekme (OHLCV)
- Teknik gösterge hesaplama
- Duygu skorlarıyla birleştirme (left join, günlük ortalama)
- Eksik günler için sıfır duygu skoru

**Aşama 4 - Model Eğitimi:**
4.1 LSTM Modeli:
- prepare_training_data ile lookback window oluşturma
- %80 eğitim / %20 test ayrımı
- Model checkpoint models/training_history/ dizinine kaydedilir
- Eğitim geçmişi CSV olarak kaydedilir
- Performans raporu JSON olarak kaydedilir
- Metrikler: RMSE, MAE, R²

4.2 XGBoost Modeli:
- prepare_classification_data ile binary etiket oluşturma
- train_test_split ile bölme (shuffle=False)
- Doğruluk ve yukarı/aşağı tahmin sayısı gösterimi
- Özellik önem analizi

**Aşama 5 - Geriye Dönük Test:**
- 10,000 TL başlangıç sermayesiyle simülasyon
- Al-Sat stratejisi
- Metrikler: Toplam Getiri, Doğruluk, İşlem Sayısı, Son Sermaye

**Aşama 6 - Görselleştirme:**
- Fiyat ve tahmin grafiği (Plotly)
- Duygu skorları zaman serisi (Plotly)
- Model performans metrikleri
- Özellik önem bar grafiği (Plotly)

#### Model Performansı Sekmesi
- Kaydedilmiş model raporlarını yükleme
- Model seçim dropdown'u
- Performans metrikleri: RMSE, MAE, MAPE, R², Yönsel Doğruluk, MSE
- Eğitim geçmişi grafiği (loss, MAE)
- Overfitting analizi
- Model karşılaştırma (birden fazla model varsa)
- En iyi model seçimi (4 farklı metrik bazında)

#### Görselleştirme Fonksiyonları
- **plot_price_predictions(actual, predicted, dates)**: Gerçek vs tahmin fiyat grafiği
  - Mavi: Gerçek fiyat, Kırmızı kesik: Tahmin
  - Tarih veya index eksenli
  - Yükseklik: 500px, template: plotly_white

- **plot_sentiment_scores(sentiment_df)**: Duygu zaman serisi
  - Yeşil: Pozitif, Kırmızı: Negatif, Mor noktalı: Compound
  - Çift Y ekseni (compound için sağ eksen, -1/1 aralığı)
  - Günlük ortalama hesaplama, tarih sıralama

- **plot_feature_importance(importance_df, top_n=15)**: Özellik önem grafiği
  - Yatay bar, Viridis renk skalası
  - Dinamik yükseklik (top_n * 30, min 400)
  - En önemli N özellik

---

### 5.2 src/news_page.py — Haber Okuma Sayfası

#### İki Alt Modül
1. **Hisse Haberleri (display_stock_news)**:
   - Bilinen 21 hisse için dropdown
   - Manuel hisse kodu girişi
   - Google News RSS ile dinamik haber çekme
   - URL şablonu: https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr
   - Lexicon ile duygu analizi
   - 4 metrik: Ortalama Duygu, Pozitif, Nötr, Negatif sayısı
   - Kart formatında haber gösterimi

2. **Genel Piyasa Haberleri (display_market_news)**:
   - 6 kategori dropdown: Borsa, Turizm, Enerji, Teknoloji, Analiz, Genel Ekonomi
   - Config'den ilgili RSS URL'leri otomatik yükleme
   - Aynı duygu analizi ve metrik gösterimi

#### Haber Kart Formatı (_render_news_card)
- Başlık (bold)
- Kaynak ve zaman (caption)
- Habere git linki
- Duygu emoji: 🟢 pozitif (>=0.3), 🔴 negatif (<=-0.3), 🟡 nötr
- Renkli skor değeri
- İsteğe bağlı ilişki skoru

#### Desteklenen Hisse Sembolleri (news_page)
THYAO, GARAN, ISCTR, AKBNK, TUPRS, ASELS, EREGL, KCHOL, SAHOL,
PETKM, SISE, TCELL, TTKOM, VAKBN, YKBNK, BIMAS, MGROS, TOASO, FROTO,
SASA, REEDR (21 hisse)

---

### 5.3 src/news_fetcher.py — Haber Toplama Modülü

#### fetch_rss_news(rss_urls, max_articles, filter_keywords, per_feed_limit)
RSS beslemelerinden haber toplama ana fonksiyonu.

Parametreler:
- rss_urls: RSS URL listesi
- max_articles: Toplam maksimum haber sayısı (varsayılan 100)
- filter_keywords: Opsiyonel anahtar kelime filtresi (erken filtreleme)
- per_feed_limit: Her RSS kaynağından çekilecek maksimum haber (varsayılan 50)

İşleyiş:
1. Her URL için feedparser ile RSS parse etme
2. Bozo exception kontrolü (geçersiz RSS formatı)
3. Her entry için: tarih, başlık, içerik, kaynak, link çıkarma
4. Başlık boşsa entry atlanır
5. İçerik boşsa entry atlanır
6. filter_keywords varsa: başlık+içerik içinde en az bir eşleşme aranır
7. Google News için özel kaynak adı çıkarma (entry.source.title)
8. Tüm haberler tarihe göre azalan sırada sıralanır
9. max_articles limiti uygulanır

Döndürülen DataFrame sütunları: date, title, content, source, link

#### _extract_date(entry)
RSS entry'den tarih çıkarma, öncelik sırası:
1. published_parsed
2. published string (dateutil ile parse)
3. updated_parsed
4. updated string
5. Hiçbiri yoksa: datetime.now()

#### _extract_source_from_url(url)
URL'den domain adı çıkarma (www. öneki kaldırılır)

#### save_news(df, filepath, format)
Haberleri CSV veya JSON formatında kaydetme

#### load_news(filepath, format)
Kaydedilmiş haberleri DataFrame olarak yükleme, tarih sütunu otomatik parse edilir

---

### 5.4 src/news_filter.py — Haber Filtreleme Modülü

#### STOCK_KEYWORDS Sözlüğü
21 hisse için önceden tanımlı anahtar kelime listeleri:

THYAO için 24 kelime:
'türk hava yolları', 'thy', 'turkish airlines', 'havayolu', 'uçak', 'havacılık',
'türk hava', 'hava yolları', 'thy a.o', 'thy a.ş', 'thy hisse', 'uçuş', 'pilot',
'hostes', 'yolcu', 'kargo', 'boeing', 'airbus', 'istanbul havalimanı',
'sabiha gökçen', 'atatürk havalimanı', 'bilet', 'rezervasyon', 'miles'

Diğer hisseler için daha az kelime (2-5 kelime arası).

#### SECTOR_KEYWORDS Sözlüğü
7 sektör için genel kelime listeleri:
- banking: banka, kredi, mevduat, faiz, finansal... (13 kelime)
- airline: havayolu, uçak, havacılık... (18 kelime)
- energy: enerji, petrol, doğalgaz... (14 kelime)
- technology: teknoloji, yazılım, dijital... (12 kelime)
- manufacturing: üretim, sanayi, fabrika... (10 kelime)
- retail: perakende, market, mağaza... (10 kelime)
- defense: savunma, askeri, silah... (10 kelime)

#### get_stock_info(symbol)
Hisse kodu için keywords ve sektör bilgisi döndürür.
.IS uzantısını kaldırır, STOCK_KEYWORDS'ten kelimeleri alır.
Sektör eşleşmesi:
- banking: GARAN, ISCTR, AKBNK, VAKBN, YKBNK
- airline: THYAO
- energy: TUPRS, PETKM
- technology: TCELL, TTKOM
- manufacturing: TOASO, FROTO
- retail: BIMAS, MGROS
- defense: ASELS

#### calculate_relevance_score(text, keywords)
Kelime eşleşme sayısına göre 0.0–1.0 arası skor.
3 veya daha fazla eşleşme = 1.0 puan (normalize edilmiş).

#### filter_news_by_stock(news_df, symbol, min_relevance, include_sector_news)
Haberleri ilişki skoruna göre filtreler.

Skor hesaplama mantığı:
- Başlıkta eşleşme varsa: 0.5 + (eşleşme sayısı × 0.25), max 1.0
- Sadece içerikte eşleşme varsa: eşleşme sayısı / max(keyword sayısı, 3)
- Hiç eşleşme yoksa ve sektör haberleri aktifse: sektör skoru × 0.4
- min_relevance'tan düşük olanlar filtrelenir
- Bilinmeyen hisse için otomatik keyword üretimi

#### add_custom_keywords(symbol, keywords)
Çalışma zamanında hisse için özel kelime ekleme.

#### get_relevance_summary(news_df)
Filtrelenmiş haberlerin istatistik özeti:
- total_news, avg_relevance
- high_relevance (>= 0.7)
- medium_relevance (0.3–0.7)
- low_relevance (< 0.3)

---

### 5.5 src/sentiment.py — Duygu Analizi Modülü

#### SentimentAnalyzer Sınıfı

**__init__(method, model_name)**
- method: 'bert' veya 'lexicon'
- BERT yüklenemezse otomatik olarak lexicon'a geçer

**_load_bert_model(model_name)**
- AutoTokenizer ve AutoModelForSequenceClassification yükleme
- GPU varsa otomatik CUDA kullanımı
- model.eval() modu

**_load_lexicon()**
Pozitif kelimeler (31 kelime):
iyi, güzel, harika, mükemmel, başarılı, kazanç, artış, yükseliş, büyüme,
gelişme, olumlu, pozitif, karlı, avantaj, fırsat, güçlü, sağlam, istikrarlı,
rekor, yüksek, artan, iyileşme, toparlanma, canlanma, patlama, sıçrama,
yükseldi, arttı, kazandı, başardı, geçti, aştı, ulaştı

Negatif kelimeler (26 kelime):
kötü, berbat, başarısız, kayıp, düşüş, azalma, gerileme, olumsuz, negatif,
zararlı, dezavantaj, risk, tehlike, zayıf, kırılgan, istikrarsız, düşük,
azalan, kötüleşme, çöküş, kriz, sorun, problem, sıkıntı, düştü, azaldı,
kaybetti, geriledi, düşüşte, kaybında

**analyze(text) → Dict[str, float]**
Tek metin analizi. Döndürür:
- positive: float (0.0–1.0)
- negative: float (0.0–1.0)
- neutral: float (0.0–1.0)
- compound: float (-1.0–1.0)

Boş/None metin için: {'positive':0, 'negative':0, 'neutral':1, 'compound':0}
Hata durumunda aynı nötr skor döner.

**_analyze_bert(text)**
- BERT tokenization (max 512 token, truncation)
- 3 sınıflı çıktı: [negative, neutral, positive]
- 2 sınıflı çıktı: [negative, positive], neutral=0
- compound = positive - negative

**_analyze_lexicon(text)**
- Metni küçük harfe çevirip kelimelere ayırır
- Pozitif ve negatif kelime sayımı
- positive_score = positive_count / len(words)
- negative_score = negative_count / len(words)
- compound = (positive - negative) / len(words) × 5, [-1,1] ile sınırlandırılır

**analyze_batch(texts) → pd.DataFrame**
Liste halinde metinleri analiz eder, DataFrame döndürür.

---

### 5.6 src/features.py — Özellik Mühendisliği Modülü

#### fetch_stock_data(symbol, start_date, end_date) → pd.DataFrame
yfinance API ile OHLCV hisse verisi çekme.

- Tarih geçerlilik kontrolü (start >= end ise ValueError)
- Boş veri kontrolü (sembol yanlışsa ValueError)
- Sütun isimleri küçük harfe: open, high, low, close, volume
- Index date sütununa dönüştürülür

#### calculate_technical_indicators(df) → pd.DataFrame
7 teknik gösterge hesaplar:

1. **EMA_10**: 10 günlük Üstel Hareketli Ortalama
2. **EMA_20**: 20 günlük Üstel Hareketli Ortalama
3. **EMA_50**: 50 günlük Üstel Hareketli Ortalama
4. **RSI_14**: Relative Strength Index (14 günlük)
5. **MACD**: MACD çizgisi (12-26 fark)
6. **MACD_signal**: Sinyal çizgisi (9 günlük MACD EMA'sı)
7. **Volume_MA_20**: 20 günlük Hacim Hareketli Ortalaması

#### _calculate_rsi(prices, period=14)
- delta = fiyat farkları
- gain = pozitif farklar, loss = negatif farklar
- Exponential weighted mean
- RSI = 100 - (100 / (1 + RS))

#### _calculate_macd(prices, fast=12, slow=26, signal=9)
- ema_fast = 12 günlük EMA
- ema_slow = 26 günlük EMA
- macd = ema_fast - ema_slow
- macd_signal = 9 günlük macd EMA'sı

#### merge_sentiment_with_prices(price_df, sentiment_df) → pd.DataFrame
- Her iki DataFrame'in tarih sütunlarını sadece gün seviyesine indirger
- Günlük ortalama duygu skorları hesaplar
- Left join ile fiyat verileriyle birleştirir (tüm fiyat günleri korunur)
- Haber olmayan günler için 0.0 doldurulur
- Sütun isimleri: sentiment_positive, sentiment_negative, sentiment_neutral, sentiment_compound

#### prepare_training_data(df, target_column='close', lookback=60, feature_columns=None)
LSTM için lookback window verisi oluşturur.

- Minimum veri kontrolü: lookback + 1 satır gerekli
- Özellik sütunları: date ve target hariç tüm numeric sütunlar
- NaN değerler: forward fill + backward fill
- MinMaxScaler ile 0-1 arasında normalizasyon
- Sliding window: X[i] = features[i-lookback:i], y[i] = target[i]
- Döndürür: (X, y, scaler)
  - X shape: (samples, lookback, features)
  - y shape: (samples,)

---

### 5.7 src/model_lstm.py — LSTM Model Modülü

#### LSTM Model Mimarisi
```
Input Layer (timesteps=60, features=N)
    ↓
LSTM Layer 1 (50 birim, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (50 birim, return_sequences=False)
    ↓
Dropout (0.2)
    ↓
Dense Layer (25 birim, ReLU aktivasyon)
    ↓
Dense Layer (1 birim, Linear aktivasyon) — Fiyat tahmini
```

Optimizer: Adam
Loss: MSE
Metrics: MAE, RMSE

#### build_lstm_model(input_shape, lstm_units, dropout)
- Varsayılan parametreler config'den alınır
- Sequential API kullanımı
- compile ile eğitime hazır hale getirilir
- Toplam parametre sayısı loglanır

#### train_lstm_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, ...)
Veri boyutu kontrolleri (3D X, 1D y, eşleşen sample sayıları)

Callbacks:
1. EarlyStopping: val_loss izleme, patience=10, best_weights restore
2. ModelCheckpoint (opsiyonel): en iyi modeli .keras formatında kaydetme

Eğitim sırasında loglar: final loss, val_loss

#### predict_lstm(model, X, scaler=None) → np.ndarray
- 3D girdi kontrolü
- model.predict() ile tahmin
- Negatif tahminler için uyarı
- scaler verilmişse inverse_transform ile orijinal ölçeğe dönüş
- Döndürür: (samples,) shape ndarray

#### save_lstm_model(model, filepath)
.keras formatında kaydetme, dizin otomatik oluşturulur

#### load_lstm_model(filepath)
Dosya varlığı kontrolü, keras.models.load_model ile yükleme

---

### 5.8 src/model_xgboost.py — XGBoost Model Modülü

#### prepare_classification_data(df, target_column='close') → (X, y)
- Boş DataFrame ve target sütun kontrolleri
- Yön etiketleri: 1 = fiyat artacak, 0 = fiyat düşecek/sabit
- Hesaplama: direction = (close.shift(-1) > close)
- Son satır çıkarılır (gelecek fiyat bilgisi yok)
- Target sütunu X'ten çıkarılır (data leakage önleme)
- Sınıf dağılımı loglanır

#### train_xgboost_model(X_train, y_train, X_val, y_val, params) → XGBClassifier
- Boş veri ve boyut kontrolleri
- Varsayılan params:
  max_depth=6, learning_rate=0.1, n_estimators=100,
  objective=binary:logistic, eval_metric=logloss, random_state=42
- Kullanıcı params ile merge edilir
- Validasyon seti varsa eval_set kullanılır, verbose=False

#### get_feature_importance(model, feature_names) → pd.DataFrame
- feature_importances_ attribute'u kontrol edilir
- feature_names verilmezse booster'dan alınmaya çalışılır
- DataFrame: feature, importance sütunları
- Azalan sıraya göre sıralanır

#### predict_direction(model, X) → (predictions, probabilities)
- Binary tahminler: 0 veya 1
- predict_proba ile olasılıklar: sadece pozitif sınıf ([:, 1]) döner
- Yukarı ve aşağı tahmin sayısı loglanır

#### save_model(model, filepath) / load_model(filepath)
pickle ile kaydetme/yükleme, dizin otomatik oluşturulur

---

### 5.9 src/backtest.py — Geriye Dönük Test Modülü

#### simple_backtest(predictions, actual_prices, initial_capital=10000.0)
Al-Sat Stratejisi:
- Tahmin > mevcut fiyat → AL (pozisyon yoksa)
- Tahmin < mevcut fiyat → SAT (pozisyon varsa)
- Pozisyon açıksa son fiyattan kapatılır

Kısıtlama: len(predictions) = len(actual_prices) - 1

Döndürür:
- total_return: Toplam getiri (%)
- accuracy: Yön tahmini doğruluğu (%)
- num_trades: Toplam işlem sayısı
- final_capital: Son sermaye (TL)

#### calculate_metrics(y_true, y_pred) → Dict
- mse: sklearn mean_squared_error
- rmse: sqrt(mse)
- mae: sklearn mean_absolute_error
- r2: sklearn r2_score
Boş veya uyumsuz boyut kontrolü yapılır.

---

### 5.10 src/model_evaluation.py — Model Değerlendirme Modülü

#### ModelTracker Sınıfı
- model_name ve save_dir parametreleriyle başlatılır
- get_callbacks(): ModelCheckpoint, EarlyStopping, CSVLogger callback listesi döner
- save_training_history(history): Eğitim geçmişini CSV'ye kaydeder

#### calculate_comprehensive_metrics(y_true, y_pred) → Dict
6 metrik hesaplar:
- mse, rmse, mae: Standart regresyon metrikleri
- mape: Yüzde hata (sıfır değerler atlanır)
- r2: Açıklayıcılık oranı
- directional_accuracy: np.diff ile yön doğruluğu (%)

#### calculate_error_distribution(y_true, y_pred) → Dict
- errors: Tahmin - gerçek farklar
- mean_error, std_error
- percentiles: 25., 50., 75. yüzdelik

#### plot_training_history(history_df) → plt.Figure
2 subplot: Loss grafiği (train/val) ve MAE grafiği (train/val)

#### plot_predictions_vs_actual(y_true, y_pred, dates) → plt.Figure
2 subplot: Zaman serisi karşılaştırması + Scatter plot (mükemmel tahmin çizgisi)

#### plot_error_distribution(errors) → plt.Figure
50 bin histogram + sıfır çizgisi + ortalama hata çizgisi

#### plot_metrics_comparison(metrics_dict) → plt.Figure
6 subplot'ta farklı model versiyonlarının karşılaştırması (bar chart)

#### save_model_report(model_name, metrics, save_path)
JSON formatında kaydeder: model_name, timestamp, metrics

#### load_model_reports(reports_dir) → Dict
Dizindeki tüm .json raporlarını yükler

#### select_best_model(metrics_dict, primary_metric, minimize) → str
Verilen metriğe göre en iyi modeli seçer

---

## 6. TEST ALTYAPISI

### 6.1 Test Yapısı
Proje hem birim testler (basic) hem özellik tabanlı testler (properties) içerir.
Özellik tabanlı testler için Hypothesis kütüphanesi kullanılır.

### 6.2 Test Dosyaları ve Kapsam

| Test Dosyası | Kapsam |
|---|---|
| test_app_basic.py | Plotly görselleştirme fonksiyonları |
| test_app_properties.py | Görselleştirme özellik testleri |
| test_backtest_basic.py | Geriye dönük test fonksiyonları |
| test_backtest_properties.py | Backtest özellik testleri |
| test_features_basic.py | Özellik hesaplama |
| test_features_properties.py | Özellik özellik testleri |
| test_integration_e2e.py | Uçtan uca entegrasyon |
| test_model_evaluation.py | Model değerlendirme |
| test_model_evaluation_properties.py | Değerlendirme özellik testleri |
| test_model_lstm_basic.py | LSTM modeli |
| test_model_lstm_properties.py | LSTM özellik testleri |
| test_model_xgboost_basic.py | XGBoost modeli |
| test_model_xgboost_properties.py | XGBoost özellik testleri |
| test_news_fetcher_basic.py | Haber toplama |
| test_news_fetcher_properties.py | Haber toplama özellik testleri |
| test_sentiment_basic.py | Duygu analizi |
| test_sentiment_properties.py | Duygu analizi özellik testleri |

Toplam: 17 test dosyası

### 6.3 Detaylı Test Kapsamı

#### test_app_basic.py — TestPlotPricePredictions
- Temel grafik oluşturma (2 trace, isim kontrolü)
- Tarih dizisi ile grafik
- Tarifsiz grafik (index ekseni)
- Tek nokta grafik
- 1000 veri noktalı büyük dataset
- Layout özellikleri (başlık, eksen, yükseklik)
- Trace özellikleri (renk, çizgi stili, genişlik)

#### test_app_basic.py — TestPlotSentimentScores
- Temel duygu grafiği (3 trace: Pozitif, Negatif, Compound)
- Aynı günde birden fazla haber (günlük ortalama)
- Tek günlük veri
- Layout özellikleri
- Çift Y ekseni (compound için sağ eksen, -1/1 aralığı)
- Sıralanmamış tarihler

#### test_app_basic.py — TestPlotFeatureImportance
- Yatay bar grafiği, Top N özellik
- Top N'den az özellik
- Tek özellik
- Dinamik yükseklik
- Viridis renk skalası
- Ters sıralama (en önemli en üstte)

#### test_app_basic.py — TestEdgeCases
- Boş array'ler, farklı uzunluklar, sıfır değerler, negatif değerler, çok büyük değerler
- Eksik sütunlar, sıfır önem değerleri, negatif önem değerleri

#### test_integration_e2e.py — TestEndToEndIntegration
- Tam analiz pipeline'ı (mock verilerle)
- Veri kalıcılık testi (CSV ve JSON kaydet/yükle)
- Model kalıcılık testi (LSTM kaydet/yükle)
- Duygu analizi entegrasyonu
- Özellik mühendisliği pipeline'ı

#### test_sentiment_basic.py
- Boş metin, çok uzun metin, özel karakterler
- Model yükleme hatası fallback (lexicon'a geçiş)
- ImportError fallback
- Boş liste, tek metin, çoklu metin batch
- Exception handling
- Pozitif, negatif, nötr, karışık metinler
- String olmayan girdi
- Büyük/küçük harf duyarsızlığı

### 6.4 Testleri Çalıştırma
```bash
pytest tests/                           # Tüm testler
pytest tests/ -k property               # Sadece özellik testleri
pytest tests/test_sentiment_basic.py    # Belirli bir dosya
pytest tests/ --run (vitest yoksa)      # Tek seferlik çalıştırma
```

---

## 7. EĞİTİLMİŞ MODELLER (models/training_history/)

Projede hazır 5 eğitilmiş LSTM modeli bulunmaktadır:

| Model Dosyası | Hisse | Tarih |
|---|---|---|
| lstm_THYAO_20260301_125056_best.keras | THYAO | 01.03.2026 12:50 |
| lstm_THYAO_20260301_132223_best.keras | THYAO | 01.03.2026 13:22 |
| lstm_THYAO_20260301_134744_best.keras | THYAO | 01.03.2026 13:47 |
| lstm_THYAO_20260301_140500_best.keras | THYAO | 01.03.2026 14:05 |
| lstm_SASA_20260301_125616_best.keras  | SASA  | 01.03.2026 12:56 |

Her model için üç dosya:
- _best.keras: En iyi checkpoint (Keras formatı)
- _history.csv: Epoch bazlı eğitim loss/MAE geçmişi
- _report.json: Performans metrikleri (RMSE, MAE, MAPE, R², Yönsel Doğruluk)

---

## 8. WEB ARAYÜZÜ ÖZELLİKLERİ VE KULLANIM

### 8.1 Uygulamayı Başlatma
```bash
py -m streamlit run src/app.py
# veya
streamlit run src/app.py
```
Varsayılan adres: http://localhost:8501

### 8.2 Analiz Adımları
1. Ana menüden "Tahmin Uygulamasına Git" seçimi
2. Sidebar'dan hisse sembolü seçimi (dropdown veya manuel)
3. Tarih aralığı belirleme (min 90 gün önerilir)
4. Gelişmiş ayarlar (opsiyonel):
   - Haber filtresini aç/kapat
   - Minimum ilişki skorunu ayarla
   - RSS kaynak kategorilerini seç
   - LSTM epoch ve lookback parametreleri
   - BERT veya Lexicon seçimi
5. "Analiz Et" butonuna tıkla
6. 6 aşamalı süreç otomatik çalışır
7. Sonuçları 2 sekmede incele: Analiz + Model Performansı

### 8.3 Haber Okuma
1. Ana menüden "Haber Okumasına Git"
2. "Hisse Haberleri" veya "Genel Piyasa Haberleri"
3. Hisse Haberleri: dropdown veya manuel giriş
4. Google News RSS ile anlık haber çekme
5. Lexicon duygu analizi
6. Kart formatında gösterim

### 8.4 Performans Beklentileri
| İşlem | Süre (CPU) | Süre (GPU) |
|---|---|---|
| BERT modeli indirme (ilk kez) | 5–10 dk | 5–10 dk |
| 100 haber toplama | ~30 sn | ~30 sn |
| 100 haber duygu analizi (BERT) | 2–3 dk | ~30 sn |
| 100 haber duygu analizi (Lexicon) | saniyeler | saniyeler |
| LSTM eğitimi (50 epoch) | 5–10 dk | 1–2 dk |
| XGBoost eğitimi (100 ağaç) | 10–30 sn | 10–30 sn |

---

## 9. VERİ AKIŞI (END-TO-END PIPELINE)

```
[RSS KAYNAKLARI]
       ↓
[fetch_rss_news]
  - Per feed limit: 100
  - Anahtar kelime erken filtresi
  - Toplam limit: 200
       ↓
[filter_news_by_stock]
  - Başlık ağırlıklı skor
  - Sektör haberleri (× 0.4 ağırlık)
  - min_relevance eşiği
       ↓
[SentimentAnalyzer.analyze_batch]
  - BERT veya Lexicon
  - Sonuç: positive, negative, neutral, compound
       ↓
[fetch_stock_data]
  - yfinance API
  - OHLCV verisi
       ↓
[calculate_technical_indicators]
  - EMA 10/20/50
  - RSI 14
  - MACD (12/26/9)
  - Volume MA 20
       ↓
[merge_sentiment_with_prices]
  - Günlük ortalama duygu
  - Left join (fiyat verisi korunur)
       ↓
[prepare_training_data]
  - MinMaxScaler normalizasyon
  - Sliding window (lookback=60)
  - X: (samples, 60, features)
  - y: (samples,)
       ↓
    ┌──────────────────────┐
    ↓                      ↓
[LSTM Model]         [XGBoost Model]
 Fiyat tahmini       Yön tahmini
 Regresyon           Sınıflandırma
    ↓                      ↓
[predict_lstm]       [predict_direction]
    ↓                      ↓
    └──────────────────────┘
               ↓
[simple_backtest]
  - Al-Sat stratejisi
  - Getiri hesaplama
               ↓
[calculate_metrics / calculate_comprehensive_metrics]
  - MSE, RMSE, MAE, MAPE, R², Yönsel Doğruluk
               ↓
[Görselleştirme]
  - Fiyat grafiği (Plotly)
  - Duygu grafiği (Plotly)
  - Özellik önemi (Plotly)
  - Metrik kartları
```

---

## 10. RSS FİLTRELEME SİSTEMİ — İKİ AŞAMALI MİMARİ

### Aşama 1: Haber Çekme Aşamasında Filtre (Geniş)
- Hisse + sektör anahtar kelimeleri fetch_rss_news'e geçirilir
- Her haberin başlık+içeriği kontrol edilir
- En az 1 eşleşme gerekli (OR mantığı)
- Verimlilik: Ham 1400 haber → ~100–150 ilgili haber

### Aşama 2: Hassas Filtreleme (Dar)
- filter_news_by_stock ile skor bazlı filtreleme
- Başlık eşleşmesi öncelikli (0.5 base skor)
- min_relevance eşiği uygulanır
- Verimlilik: ~100–150 → ~30–50 yüksek ilişkili haber

### İyileştirme Öncesi vs Sonrası
- Önceki sistem: 15 kaynak → 200 haber → ~6 ilgili (%3 verimlilik)
- Yeni sistem: 14 kaynak → filtreli çekim → ~30–50 haber (%30–50 verimlilik)
- İyileşme: 15–20 kat daha verimli

---

## 11. HABER OKUMA SAYFASI MİMARİSİ

### Google News Entegrasyonu
Hisse bazlı haberler için Google News RSS kullanılır:
- URL şablonu: https://news.google.com/rss/search?q={encoded_query}&hl=tr&gl=TR&ceid=TR:tr
- Her hisse için özel arama terimi (STOCK_SEARCH_TERMS sözlüğü)
- Örnek: THYAO → "THYAO türk hava yolları"
- Gerçek zamanlı haber çekimi

### Kategori → RSS Eşleşmesi
borsa_odakli → config.RSS_FEEDS['borsa_odakli']
turizm → config.RSS_FEEDS['sektorel']['turizm']
analiz_yorum → config.RSS_FEEDS['analiz_yorum']
genel_ekonomi → config.RSS_FEEDS['genel_ekonomi']

### Duygu Renk Kodlaması
🟢 Compound >= 0.3  → Pozitif haber
🔴 Compound <= -0.3 → Negatif haber
🟡 Arası           → Nötr haber

---

## 12. KURULUM VE GEREKSİNİMLER

### Sistem Gereksinimleri
- Python 3.10 veya üzeri
- 2GB+ RAM (BERT modeli için)
- İnternet bağlantısı (ilk çalıştırmada model indirme)
- GPU (opsiyonel, BERT ve LSTM için hız artışı)

### Kurulum Adımları
```bash
# 1. Repoyu klonla
git clone <repo-url>
cd turkish_sentiment_stock

# 2. Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. BERT modelini önceden indir (opsiyonel)
python -c "from transformers import AutoTokenizer, AutoModel;
AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased');
AutoModel.from_pretrained('dbmdz/bert-base-turkish-cased')"

# 5. Uygulamayı başlat
py -m streamlit run src/app.py
```

### Bilinen Kurulum Sorunları
- TensorFlow: pip install tensorflow==2.15.0
- XGBoost: pip install xgboost --upgrade
- Streamlit: pip install streamlit --upgrade

---

## 13. BİLİNEN SINIRLAMALAR

### Teknik Sınırlamalar
- Prototip — üretime hazır değil
- Gerçek zamanlı veri akışı yok (sadece geçmiş veri)
- Tek kullanıcı desteği
- Veritabanı yok (CSV/JSON dosya tabanlı depolama)
- Minimal güvenlik önlemleri
- Kapsamlı hata yönetimi eksik

### Veri Sınırlamaları
- RSS kaynakları zaman zaman erişilemez olabilir
- KAP RSS formatı hatalı — devre dışı
- 100'den az haberle düşük performans
- Minimum 90 günlük veri önerilir
- Tatil ve hafta sonu günlerinde veri eksikliği
- Türkçe karakter encoding sorunları olabilir

### Model Sınırlamaları
- BERT modeli ~500MB indirme
- LSTM eğitimi CPU'da yavaş
- Overfitting riski (küçük veri setleriyle)
- BERT için 2GB+ RAM gerekli
- Tarih uyumsuzluğu (haber tarihleri vs hisse fiyat tarihleri)

### Kullanım Uyarıları
- Finansal tavsiye değildir
- Geçmiş performans gelecek performansı garantilemez
- Ani piyasa değişimlerini tahmin edemez
- Politik olaylar, doğal afetler modelde yer almaz

### NaN Değerleri
İlk günlerde bazı teknik göstergeler NaN üretir (otomatik forward/backward fill uygulanır)

---

## 14. PROJE EVRİMİ — YAPILAN BAŞLICA DEĞİŞİKLİKLER

### v1.0 → v2.0: Haber Çekme İyileştirmesi
- Her RSS kaynağından ~13 haber → 100 haber
- Filtreleme sonradan → haber çekme aşamasında
- Verimlilik: %3 → %50–70

### v2.0 → v3.0: RSS Kaynak Gruplandırması
- Düz liste → 4 kategorili hiyerarşik yapı
- 15 kaynak → 24 kaynak (%60 artış)
- Sektörel kategoriler (turizm, enerji, teknoloji)
- Kullanıcı seçilebilir kaynak kategorileri

### Kaldırılan Özellik: "Sektör Haberlerini Dahil Et" Checkbox
- RSS Kaynak Seçimi özelliğiyle çakışıyordu
- UI basitleştirildi (3 değişken → 2 değişken)
- %33 daha az kod karmaşıklığı

### Eklenen Özellikler
- Ana menü (tahmin + haber okuma modülleri)
- Haber okuma sayfası (Google News entegrasyonu)
- Model Performansı sekmesi (karşılaştırma, en iyi model seçimi)
- Model checkpoint ve rapor kaydetme sistemi
- İki aşamalı haber filtreleme
- Bilinmeyen hisse için otomatik keyword üretimi

---

## 15. KIRO IDE AYARLARI (.kiro/)

### Hook'lar
1. **always-turkish-response.kiro.hook**: Her AI yanıtının Türkçe olmasını zorunlu kılar
2. **wait-for-command-completion.kiro.hook**: Komut tamamlanmadan devam etmemeyi sağlar

### Specs
- turkish-sentiment-stock-prediction: Proje spec dosyası (özellik geliştirme rehberi)

---

## 16. ÖRNEK KULLANIM SENARYOLARI

### Senaryo 1: THYAO Hisse Analizi
```
Hisse: THYAO.IS
Tarih: 2023-01-01 – 2024-01-01
RSS Kaynakları: Borsa Odaklı + Turizm + Analiz
Haber Filtresi: Aktif, Min Skor: 0.2
LSTM: 50 epoch, Lookback: 60
Duygu: BERT
Beklenen haber: ~30–50
Beklenen süre: 5–10 dk (CPU)
```

### Senaryo 2: TUPRS Enerji Analizi
```
Hisse: TUPRS.IS
RSS Kaynakları: Borsa Odaklı + Enerji + Analiz
Beklenen haber: ~20–35
```

### Senaryo 3: Hızlı Test (Lexicon ile)
```
Hisse: GARAN.IS
RSS Kaynakları: Sadece Borsa Odaklı
LSTM: 30 epoch, Lookback: 30
Duygu: Lexicon
Beklenen süre: 2–3 dk
```

### Senaryo 4: Bilinmeyen Hisse
```
Hisse: SASA.IS
Sistem otomatik keyword üretir: ['sasa', 'sasa']
Sektör belirlenmez, sadece hisse özel kelimelerle filtreleme
```

---

## 17. API VE FONKSİYON REFERANSI ÖZETI

### Haber Modülleri
```python
from src.news_fetcher import fetch_rss_news, save_news, load_news
from src.news_filter import filter_news_by_stock, get_stock_info, get_relevance_summary
```

### Duygu Analizi
```python
from src.sentiment import SentimentAnalyzer
analyzer = SentimentAnalyzer(method='bert')  # veya 'lexicon'
score = analyzer.analyze("metin")
df = analyzer.analyze_batch(["metin1", "metin2"])
```

### Özellik Mühendisliği
```python
from src.features import (
    fetch_stock_data,
    calculate_technical_indicators,
    merge_sentiment_with_prices,
    prepare_training_data
)
```

### LSTM Modeli
```python
from src.model_lstm import build_lstm_model, train_lstm_model, predict_lstm, save_lstm_model, load_lstm_model
```

### XGBoost Modeli
```python
from src.model_xgboost import (
    prepare_classification_data,
    train_xgboost_model,
    get_feature_importance,
    predict_direction,
    save_model, load_model
)
```

### Backtest ve Değerlendirme
```python
from src.backtest import simple_backtest, calculate_metrics
from src.model_evaluation import (
    calculate_comprehensive_metrics,
    save_model_report,
    load_model_reports,
    select_best_model
)
```

---

## 18. SONUÇ

Bu proje, Türkçe doğal dil işleme (NLP) ve finansal makine öğrenmesini birleştiren
kapsamlı bir araştırma prototipidir. Temel değeri:

1. Türkçe BERT ile yüksek kaliteli duygu analizi
2. Çok kaynaklı RSS haber toplama ve akıllı filtreleme
3. Klasik teknik göstergeler + duygu verisi birleştirme
4. LSTM (regresyon) + XGBoost (sınıflandırma) ikili model yaklaşımı
5. Streamlit ile kullanıcı dostu web arayüzü
6. Kapsamlı test altyapısı (birim + özellik + entegrasyon)

Sistem tamamen modülerdir — her bileşen bağımsız kullanılabilir.
Gerçek dünya kullanımı için kapsamlı hata yönetimi, veritabanı entegrasyonu
ve güvenlik önlemleri eklenmesi gerekir.

---

Bu belge projenin tüm bileşenlerini, fonksiyonlarını, konfigürasyonlarını
ve kullanım senaryolarını kapsamaktadır.
Son güncelleme: Temmuz 2026
