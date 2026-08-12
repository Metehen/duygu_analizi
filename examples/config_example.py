"""
Örnek Yapılandırma Dosyası

Bu dosya, config.py için örnek ayarları içerir.
Kendi ihtiyaçlarınıza göre özelleştirebilirsiniz.
"""

# ============================================================================
# RSS KAYNAKLARI
# ============================================================================

# Türkçe ekonomi haberleri için RSS beslemeleri - Gruplandırılmış Yapı
RSS_FEEDS = {
    # Borsa Odaklı Kaynaklar
    'borsa_odakli': [
        'https://www.bloomberght.com/rss',
        'https://www.paraanaliz.com/feed/',
        'https://www.investing.com/rss/news_285.rss',
        'https://www.foreks.com/rss',
        'https://www.bigpara.com/rss',
        # 'https://www.kap.org.tr/tr/rss',  # KAP - Yavaş olabilir
    ],
    
    # Sektörel Kaynaklar
    'sektorel': {
        # Turizm Sektörü (THYAO, MAALT için)
        'turizm': [
            'https://www.turizmguncel.com/rss',
            # 'https://www.turizmhaberleri.com/rss',
        ],
        
        # Enerji Sektörü (TUPRS, PETKM için)
        'enerji': [
            'https://www.enerjigundem.com/rss',
            # 'https://www.enerjiportali.com/rss',
        ],
        
        # Teknoloji Sektörü (TCELL, TTKOM için)
        'teknoloji': [
            'https://www.teknolojituru.com/rss',
            # 'https://www.teknolojioku.com/rss',
        ],
    },
    
    # Analiz ve Yorum Kaynakları
    'analiz_yorum': [
        'https://www.dunya.com/rss',
        'https://www.ekonomim.com/rss',
        # 'https://www.analizmarket.com/rss',
        # 'https://www.borsagundem.com/rss',
    ],
    
    # Genel Ekonomi Haberleri
    'genel_ekonomi': [
        'https://www.aa.com.tr/tr/rss/default?cat=ekonomi',
        'https://www.haberturk.com/rss/ekonomi.xml',
        'https://www.hurriyet.com.tr/rss/ekonomi',
        # 'https://www.sabah.com.tr/rss/ekonomi.xml',
        # 'https://www.milliyet.com.tr/rss/rssnew/ekonomirssnew.xml',
        # 'https://www.ntv.com.tr/ekonomi.rss',
        # 'https://www.cnnturk.com/feed/rss/ekonomi/news',
    ],
}

# Tüm RSS kaynaklarını düz liste olarak al
def get_all_rss_feeds():
    """Tüm RSS kaynaklarını tek bir liste olarak döndürür."""
    feeds = []
    feeds.extend(RSS_FEEDS['borsa_odakli'])
    for sektor_feeds in RSS_FEEDS['sektorel'].values():
        feeds.extend(sektor_feeds)
    feeds.extend(RSS_FEEDS['analiz_yorum'])
    feeds.extend(RSS_FEEDS['genel_ekonomi'])
    return feeds

RSS_FEEDS_LIST = get_all_rss_feeds()

# Maksimum çekilecek haber sayısı (toplam)
MAX_ARTICLES_PER_FEED = 100

# ============================================================================
# HİSSE SENETLERİ
# ============================================================================

# Varsayılan hisse sembolleri (Borsa Istanbul)
# Not: Borsa Istanbul sembolleri .IS uzantısı ile kullanılır
DEFAULT_SYMBOLS = [
    'THYAO.IS',  # Türk Hava Yolları
    'GARAN.IS',  # Garanti Bankası
    'ISCTR.IS',  # İş Bankası (C)
    'AKBNK.IS',  # Akbank
    'EREGL.IS',  # Ereğli Demir Çelik
    'TUPRS.IS',  # Tüpraş
    'BIMAS.IS',  # BİM
    'SAHOL.IS',  # Sabancı Holding
]

# Varsayılan hisse sembolü (web arayüzü için)
DEFAULT_SYMBOL = 'THYAO.IS'

# ============================================================================
# TARİH AYARLARI
# ============================================================================

# Varsayılan tarih aralıkları
DEFAULT_START_DATE = '2024-01-01'
DEFAULT_END_DATE = '2025-01-01'

# Minimum gerekli veri günü sayısı
MIN_DATA_DAYS = 90

# ============================================================================
# LSTM MODEL PARAMETRELERİ
# ============================================================================

LSTM_CONFIG = {
    # Model mimarisi
    'lstm_units': [50, 50],  # Her LSTM katmanındaki birim sayısı
    'dropout': 0.2,          # Dropout oranı (overfitting önleme)
    
    # Eğitim parametreleri
    'epochs': 50,            # Maksimum epoch sayısı
    'batch_size': 32,        # Batch boyutu
    'validation_split': 0.2, # Validasyon verisi oranı
    
    # Early stopping
    'early_stopping_patience': 10,  # Kaç epoch iyileşme olmazsa dur
    
    # Veri hazırlama
    'lookback': 60,          # Geçmiş gün sayısı (pencere boyutu)
    'train_test_split': 0.8, # Eğitim/test ayrımı oranı
}

# ============================================================================
# XGBOOST MODEL PARAMETRELERİ
# ============================================================================

XGBOOST_CONFIG = {
    # Model parametreleri
    'max_depth': 6,          # Ağaç derinliği (3-10 arası önerilir)
    'learning_rate': 0.1,    # Öğrenme oranı (0.01-0.3 arası)
    'n_estimators': 100,     # Ağaç sayısı
    'min_child_weight': 1,   # Minimum child weight
    'subsample': 0.8,        # Subsample oranı
    'colsample_bytree': 0.8, # Feature subsample oranı
    
    # Diğer ayarlar
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42,
    
    # Eğitim
    'train_test_split': 0.8,
}

# ============================================================================
# DUYGU ANALİZİ AYARLARI
# ============================================================================

# Duygu analizi yöntemi: 'bert' veya 'lexicon'
# - 'bert': Daha doğru ama yavaş (GPU önerilir)
# - 'lexicon': Hızlı ama basit
SENTIMENT_METHOD = 'bert'

# Türkçe BERT modeli
BERT_MODEL = 'dbmdz/bert-base-turkish-cased'
# Alternatif: 'savasy/bert-base-turkish-sentiment-cased'

# BERT batch size (GPU belleğine göre ayarlayın)
BERT_BATCH_SIZE = 16

# Maksimum metin uzunluğu (token sayısı)
MAX_TEXT_LENGTH = 512

# ============================================================================
# TEKNİK GÖSTERGE PARAMETRELERİ
# ============================================================================

TECHNICAL_INDICATORS = {
    # EMA (Exponential Moving Average) periyotları
    'ema_periods': [10, 20, 50],
    
    # RSI (Relative Strength Index) periyodu
    'rsi_period': 14,
    
    # MACD (Moving Average Convergence Divergence) parametreleri
    # (fast_period, slow_period, signal_period)
    'macd_params': (12, 26, 9),
    
    # Volume Moving Average periyodu
    'volume_ma_period': 20,
}

# ============================================================================
# DOSYA YOLLARI
# ============================================================================

# Ana dizinler
DATA_DIR = 'data/'
MODELS_DIR = 'models/'
LOGS_DIR = 'logs/'

# Veri dosyaları
NEWS_FILE = 'data/news.csv'
SENTIMENT_FILE = 'data/sentiment.csv'
STOCK_DATA_FILE = 'data/stock_data.csv'
MERGED_DATA_FILE = 'data/merged_data.csv'

# Model dosyaları
LSTM_MODEL_FILE = 'models/lstm_model.keras'
XGBOOST_MODEL_FILE = 'models/xgboost_model.pkl'
SCALER_FILE = 'models/scaler.pkl'

# ============================================================================
# GERIYE DÖNÜK TEST AYARLARI
# ============================================================================

BACKTEST_CONFIG = {
    'initial_capital': 10000.0,  # Başlangıç sermayesi (TL)
    'commission': 0.001,          # İşlem komisyonu (0.1%)
    'slippage': 0.0005,           # Slippage (0.05%)
}

# ============================================================================
# STREAMLIT ARAYÜZ AYARLARI
# ============================================================================

STREAMLIT_CONFIG = {
    'page_title': 'Türkçe Ekonomi Haberleri Duygu Analizi & Hisse Tahmin',
    'page_icon': '📈',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}

# ============================================================================
# LOGGING AYARLARI
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/app.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5,
}

# ============================================================================
# PERFORMANS AYARLARI
# ============================================================================

# Paralel işleme
USE_MULTIPROCESSING = False  # Çoklu işlemci kullanımı
NUM_WORKERS = 4              # İşçi sayısı

# Cache ayarları
USE_CACHE = True             # Önbellekleme kullan
CACHE_DIR = 'cache/'         # Önbellek dizini
CACHE_EXPIRY_HOURS = 24      # Önbellek geçerlilik süresi (saat)

# ============================================================================
# GELİŞTİRME AYARLARI
# ============================================================================

# Debug modu
DEBUG = False

# Test modu (küçük veri setleriyle hızlı test)
TEST_MODE = False
TEST_MODE_MAX_ARTICLES = 50
TEST_MODE_MAX_DAYS = 30

# Verbose çıktı
VERBOSE = True

# ============================================================================
# NOTLAR
# ============================================================================

"""
Yapılandırma İpuçları:

1. LSTM Lookback:
   - Küçük değerler (30-60): Hızlı eğitim, kısa vadeli tahmin
   - Büyük değerler (60-120): Yavaş eğitim, uzun vadeli tahmin

2. LSTM Units:
   - Küçük değerler ([25, 25]): Hızlı, basit model
   - Büyük değerler ([100, 100]): Yavaş, karmaşık model

3. Dropout:
   - Düşük değerler (0.1-0.2): Az regularization
   - Yüksek değerler (0.3-0.5): Çok regularization

4. XGBoost Max Depth:
   - Küçük değerler (3-5): Basit ağaçlar, az overfitting
   - Büyük değerler (7-10): Karmaşık ağaçlar, çok overfitting riski

5. Sentiment Method:
   - 'bert': Daha doğru, GPU gerektirir, yavaş
   - 'lexicon': Daha hızlı, CPU yeterli, basit

6. Veri Miktarı:
   - Minimum 90 gün veri önerilir
   - Daha fazla veri genellikle daha iyi sonuç verir
   - Çok fazla veri (>2 yıl) eğitimi yavaşlatabilir
"""
