"""
Merkezi yapılandırma dosyası.
Tüm sistem parametreleri ve ayarları burada tanımlanır.
"""

# RSS Kaynakları - Gruplandırılmış Yapı
RSS_FEEDS = {
    # Borsa Odaklı Kaynaklar
    'borsa_odakli': [
        'https://www.bloomberght.com/rss',
        'https://www.paraanaliz.com/feed/',
        'https://www.investing.com/rss/news_285.rss',  # Türkiye haberleri
        'https://tr.tradingeconomics.com/rss/turkey.xml',
        'https://www.foreks.com/rss',
        'https://www.bigpara.com/rss',
        # 'https://www.kap.org.tr/tr/rss',  # KAP - RSS formatı hatalı, devre dışı
    ],
    
    # Sektörel Kaynaklar
    'sektorel': {
        # Turizm Sektörü
        'turizm': [
            'https://www.turizmguncel.com/rss',
            'https://www.turizmhaberleri.com/rss',
        ],
        
        # Enerji Sektörü
        'enerji': [
            'https://www.enerjigundem.com/rss',
            'https://www.enerjiportali.com/rss',
        ],
        
        # Teknoloji Sektörü
        'teknoloji': [
            'https://www.teknolojituru.com/rss',
            'https://www.teknolojioku.com/rss',
        ],
    },
    
    # Analiz ve Yorum Kaynakları
    'analiz_yorum': [
        'https://www.analizmarket.com/rss',
        'https://www.borsagundem.com/rss',
        'https://www.dunya.com/rss',
        'https://www.ekonomim.com/rss',
    ],
    
    # Genel Ekonomi Haberleri
    'genel_ekonomi': [
        'https://www.mynet.com/haber/rss/ekonomi',
        'https://www.haberturk.com/rss/ekonomi.xml',
        'https://www.hurriyet.com.tr/rss/ekonomi',
        'https://www.sabah.com.tr/rss/ekonomi.xml',
        'https://www.milliyet.com.tr/rss/rssnew/ekonomirssnew.xml',
        'https://www.aa.com.tr/tr/rss/default?cat=ekonomi',
        'https://www.ntv.com.tr/ekonomi.rss',
        'https://www.cnnturk.com/feed/rss/ekonomi/news',
    ],
}

# RSS kaynaklarını düz liste olarak da tut (geriye dönük uyumluluk için)
def get_all_rss_feeds():
    """Tüm RSS kaynaklarını tek bir liste olarak döndürür."""
    feeds = []
    
    # Borsa odaklı
    feeds.extend(RSS_FEEDS['borsa_odakli'])
    
    # Sektörel (tüm alt kategoriler)
    for sektor_feeds in RSS_FEEDS['sektorel'].values():
        feeds.extend(sektor_feeds)
    
    # Analiz ve yorum
    feeds.extend(RSS_FEEDS['analiz_yorum'])
    
    # Genel ekonomi
    feeds.extend(RSS_FEEDS['genel_ekonomi'])
    
    return feeds

# Geriye dönük uyumluluk için düz liste
RSS_FEEDS_LIST = get_all_rss_feeds()

# Hisse Sembolleri (Borsa Istanbul)
DEFAULT_SYMBOLS = [
    'THYAO.IS',  # Türk Hava Yolları
    'GARAN.IS',  # Garanti Bankası
    'ISCTR.IS',  # İş Bankası
    'AKBNK.IS',  # Akbank
    'TUPRS.IS',  # Tüpraş
]

# LSTM Model Parametreleri
LSTM_CONFIG = {
    'lstm_units': [50, 50],
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'lookback': 60,
    'validation_split': 0.2,
    'early_stopping_patience': 10
}

# XGBoost Model Parametreleri
XGBOOST_CONFIG = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42
}

# Duygu Analizi Ayarları
SENTIMENT_METHOD = 'bert'  # 'bert' veya 'lexicon'
BERT_MODEL = 'dbmdz/bert-base-turkish-cased'
# Alternatif: 'savasy/bert-base-turkish-sentiment-cased'

# Veri Yolları
DATA_DIR = 'data/'
MODELS_DIR = 'models/'
NEWS_FILE = 'data/news.csv'
SENTIMENT_FILE = 'data/sentiment.csv'

# Tarih Aralıkları
DEFAULT_START_DATE = '2023-01-01'
DEFAULT_END_DATE = '2024-01-01'

# Teknik Gösterge Parametreleri
TECHNICAL_INDICATORS = {
    'ema_periods': [10, 20, 50],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'volume_ma_period': 20
}

# Geriye Dönük Test Ayarları
BACKTEST_CONFIG = {
    'initial_capital': 10000.0,
    'transaction_cost': 0.001,  # %0.1 işlem maliyeti
    'test_split': 0.2  # Son %20 test için
}

# Haber Toplama Ayarları
NEWS_CONFIG = {
    'max_articles': 200,  # Daha fazla haber çek
    'timeout': 30,  # saniye
    'retry_attempts': 3
}

# Logging Ayarları
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'app.log'
}
