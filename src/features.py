"""
Özellik Hazırlama Modülü

Bu modül, hisse senedi fiyat verilerini çekme, teknik göstergeler hesaplama,
duygu skorlarını birleştirme ve model eğitimi için veri hazırlama işlemlerini içerir.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import logging

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    yfinance kullanarak hisse verilerini çeker.
    
    Args:
        symbol: Hisse sembolü (örn: 'THYAO.IS')
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date: Bitiş tarihi (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        ValueError: Geçersiz sembol veya tarih aralığı
        ConnectionError: Bağlantı hatası
    """
    try:
        logger.info(f"Hisse verisi çekiliyor: {symbol} ({start_date} - {end_date})")
        
        # Tarih kontrolü
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if start >= end:
            raise ValueError("Başlangıç tarihi bitiş tarihinden önce olmalıdır")
        
        # yfinance ile veri çekme
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"Hisse verisi çekilemedi: {symbol}. Sembol doğru mu?")
        
        # Sütun isimlerini küçük harfe çevir
        df.columns = df.columns.str.lower()
        
        # Sadece OHLCV sütunlarını tut
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        df = df[required_columns]
        
        # Index'i date sütununa çevir
        df = df.reset_index()
        df.columns = ['date'] + required_columns
        
        logger.info(f"Başarıyla {len(df)} satır veri çekildi")
        
        return df
        
    except ValueError as e:
        logger.error(f"Değer hatası: {e}")
        raise
    except Exception as e:
        logger.error(f"Hisse verisi çekme hatası: {e}")
        raise ConnectionError(f"Hisse verisi çekilemedi: {symbol}. Bağlantı hatası: {e}")



def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Teknik göstergeleri hesaplar.
    
    Args:
        df: OHLCV verileri içeren DataFrame
    
    Returns:
        Teknik göstergeler eklenmiş DataFrame
    
    Indicators:
        - EMA_10, EMA_20, EMA_50
        - RSI_14
        - MACD, MACD_signal
        - Volume_MA_20
    """
    logger.info("Teknik göstergeler hesaplanıyor...")
    
    # DataFrame kopyası oluştur
    df = df.copy()
    
    # EMA (Exponential Moving Average) hesaplama
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI (Relative Strength Index) hesaplama
    df['rsi_14'] = _calculate_rsi(df['close'], period=14)
    
    # MACD (Moving Average Convergence Divergence) hesaplama
    df['macd'], df['macd_signal'] = _calculate_macd(df['close'])
    
    # Volume Moving Average hesaplama
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    logger.info("Teknik göstergeler başarıyla hesaplandı")
    
    return df


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI (Relative Strength Index) hesaplar.
    
    Args:
        prices: Fiyat serisi
        period: RSI periyodu (varsayılan: 14)
    
    Returns:
        RSI değerleri (0-100 aralığında)
    """
    # Fiyat değişimlerini hesapla
    delta = prices.diff()
    
    # Kazanç ve kayıpları ayır
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Ortalama kazanç ve kayıp
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    # RS (Relative Strength) hesapla
    rs = avg_gain / avg_loss
    
    # RSI hesapla
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence) hesaplar.
    
    Args:
        prices: Fiyat serisi
        fast: Hızlı EMA periyodu (varsayılan: 12)
        slow: Yavaş EMA periyodu (varsayılan: 26)
        signal: Sinyal çizgisi periyodu (varsayılan: 9)
    
    Returns:
        (MACD, MACD_signal) tuple
    """
    # Hızlı ve yavaş EMA'ları hesapla
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    # MACD çizgisi
    macd = ema_fast - ema_slow
    
    # Sinyal çizgisi
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    
    return macd, macd_signal



def merge_sentiment_with_prices(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Duygu skorlarını fiyat verileriyle birleştirir.
    
    Args:
        price_df: Fiyat ve teknik göstergeler
        sentiment_df: Duygu skorları (date, positive, negative, neutral, compound sütunları)
    
    Returns:
        Birleştirilmiş DataFrame (günlük ortalama duygu skorları)
    """
    logger.info("Duygu skorları fiyat verileriyle birleştiriliyor...")
    
    # DataFrame kopyaları oluştur
    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()
    
    # Tarih sütunlarını datetime'a çevir
    if 'date' in price_df.columns:
        price_df['date'] = pd.to_datetime(price_df['date'])
    else:
        # Eğer date index ise, sütuna çevir
        if isinstance(price_df.index, pd.DatetimeIndex):
            price_df = price_df.reset_index()
            price_df.columns = ['date'] + list(price_df.columns[1:])
    
    if 'date' in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    else:
        # Eğer date index ise, sütuna çevir
        if isinstance(sentiment_df.index, pd.DatetimeIndex):
            sentiment_df = sentiment_df.reset_index()
            sentiment_df.columns = ['date'] + list(sentiment_df.columns[1:])
    
    # Tarihleri sadece gün seviyesine indir (saat bilgisini kaldır)
    price_df['date'] = price_df['date'].dt.date
    sentiment_df['date'] = sentiment_df['date'].dt.date
    
    # Duygu skorlarını günlük ortalama olarak hesapla
    sentiment_columns = ['positive', 'negative', 'neutral', 'compound']
    
    # Sadece gerekli sütunları seç
    available_sentiment_cols = [col for col in sentiment_columns if col in sentiment_df.columns]
    
    if not available_sentiment_cols:
        logger.warning("Duygu skoru sütunları bulunamadı, boş skorlar ekleniyor")
        # Boş duygu skorları ekle
        for col in sentiment_columns:
            price_df[f'sentiment_{col}'] = 0.0
        return price_df
    
    # Günlük ortalama duygu skorlarını hesapla
    daily_sentiment = sentiment_df.groupby('date')[available_sentiment_cols].mean().reset_index()
    
    # Sütun isimlerini yeniden adlandır (sentiment_ prefix ekle)
    rename_dict = {col: f'sentiment_{col}' for col in available_sentiment_cols}
    daily_sentiment = daily_sentiment.rename(columns=rename_dict)
    
    # Fiyat verileriyle birleştir (left join - tüm fiyat verilerini koru)
    merged_df = price_df.merge(daily_sentiment, on='date', how='left')
    
    # Eksik duygu skorlarını 0 ile doldur (haberin olmadığı günler için)
    sentiment_cols_in_merged = [f'sentiment_{col}' for col in available_sentiment_cols]
    merged_df[sentiment_cols_in_merged] = merged_df[sentiment_cols_in_merged].fillna(0.0)

    # Haber olan / olmayan günleri ayırt eden flag sütunu ekle
    # Bu sayede model "duygu skoru sıfır" ile "haber yok" arasını öğrenebilir
    merged_df['has_sentiment'] = merged_df[sentiment_cols_in_merged[0]].notna().astype(float)
    # Not: fillna sonrası NaN kalmadı, bu yüzden birleştirme öncesi durumu işaretliyoruz
    # Doğru yaklaşım: birleştirme öncesi hangi tarihlerde haber var bilgisini tut
    merged_df['has_sentiment'] = merged_df['date'].isin(
        pd.to_datetime(daily_sentiment['date'])
    ).astype(float)
    
    # Tarihi tekrar datetime'a çevir
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    logger.info(f"Birleştirme tamamlandı: {len(merged_df)} satır")
    
    return merged_df



def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = 'close',
    lookback: int = 60,
    feature_columns: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Model eğitimi için veri hazırlar.
    
    Args:
        df: Özellikler içeren DataFrame
        target_column: Hedef değişken (varsayılan: 'close')
        lookback: Geçmiş gün sayısı (varsayılan: 60)
        feature_columns: Kullanılacak özellik sütunları (None ise otomatik seçilir)
    
    Returns:
        (X, y, scaler) tuple for training
        - X: shape (samples, lookback, features)
        - y: shape (samples,)
        - scaler: Normalizasyon için kullanılan MinMaxScaler
        
    Raises:
        ValueError: Yetersiz veri veya geçersiz parametreler
    """
    logger.info(f"Eğitim verisi hazırlanıyor (lookback={lookback})...")
    
    # DataFrame kopyası oluştur
    df = df.copy()
    
    # Minimum veri kontrolü
    min_required = lookback + 1
    if len(df) < min_required:
        raise ValueError(
            f"Yetersiz veri: En az {min_required} günlük veri gerekli. "
            f"Mevcut: {len(df)} satır"
        )
    
    # Hedef sütun kontrolü
    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun bulunamadı: {target_column}")
    
    # Özellik sütunlarını belirle
    if feature_columns is None:
        # Tarih ve hedef sütunu hariç tüm numeric sütunları kullan
        exclude_cols = ['date', target_column]
        feature_columns = [col for col in df.columns 
                          if col not in exclude_cols and 
                          pd.api.types.is_numeric_dtype(df[col])]
    
    if not feature_columns:
        raise ValueError("Hiç özellik sütunu bulunamadı")
    
    logger.info(f"Kullanılan özellikler ({len(feature_columns)}): {feature_columns}")
    
    # Sadece özellik sütunlarını seç
    feature_df = df[feature_columns].copy()
    
    # NaN değerleri kontrol et ve doldur
    if feature_df.isnull().any().any():
        logger.warning("NaN değerler tespit edildi, forward fill uygulanıyor")
        feature_df = feature_df.ffill().bfill()
    
    # Hedef değişkeni al ve normalize et
    target_raw = df[target_column].values.reshape(-1, 1)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaled = target_scaler.fit_transform(target_raw).flatten()
    
    # Özellikler için scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(feature_df)
    
    # Lookback window ile X, y oluştur
    X = []
    y = []
    
    for i in range(lookback, len(scaled_features)):
        # Son lookback günlük veriyi al
        X.append(scaled_features[i-lookback:i])
        # Hedef: normalize edilmiş kapanış fiyatı
        y.append(target_scaled[i])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Eğitim verisi hazırlandı: X shape={X.shape}, y shape={y.shape}")
    
    # target_scaler'ı döndür (tahminleri gerçek fiyata çevirmek için)
    return X, y, target_scaler
