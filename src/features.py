# Copyright (C) 2026  Metehen
# Licensed under GNU AGPL v3. See LICENSE file for details.
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
from src.logger import get_logger

logger = get_logger(__name__)


def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    yfinance kullanarak hisse verilerini çeker.

    Args:
        symbol:     Hisse sembolü (örn: 'THYAO.IS')
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date:   Bitiş tarihi (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data

    Raises:
        StockDataError:        Sembol bulunamadı veya veri dönmedi
        InsufficientDataError: Çok az veri döndü (model için yetersiz)
        ConfigError:           Geçersiz tarih formatı
    """
    from src.exceptions import StockDataError, InsufficientDataError, ConfigError

    # ── Girdi doğrulama ────────────────────────────────────
    if not symbol or not symbol.strip():
        raise StockDataError("Hisse sembolü boş olamaz.", symbol=symbol)

    symbol = symbol.strip().upper()

    try:
        start = pd.to_datetime(start_date)
        end   = pd.to_datetime(end_date)
    except Exception:
        raise ConfigError(
            f"Geçersiz tarih formatı: start={start_date!r}, end={end_date!r}. "
            "YYYY-MM-DD formatı kullanın.",
            hint="Örnek: 2023-01-01"
        )

    if start >= end:
        raise ConfigError(
            f"Başlangıç tarihi ({start_date}) bitiş tarihinden ({end_date}) önce olmalıdır.",
            hint="Başlangıç tarihini daha eski bir tarih olarak ayarlayın."
        )

    min_days = 30
    if (end - start).days < min_days:
        raise InsufficientDataError(
            f"Tarih aralığı çok kısa: {(end - start).days} gün. En az {min_days} gün gerekli.",
            hint=f"Başlangıç tarihini en az {min_days} gün öncesine alın.",
            days=(end - start).days
        )

    # ── Veri çekme ─────────────────────────────────────────
    logger.info("Hisse verisi çekiliyor: %s (%s - %s)", symbol, start_date, end_date)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
    except Exception as e:
        raise StockDataError(
            f"{symbol} için veri çekilemedi: {e}",
            hint="İnternet bağlantınızı ve Yahoo Finance'e erişimi kontrol edin.",
            symbol=symbol
        )

    if df.empty:
        raise StockDataError(
            f"{symbol} için hiç veri dönmedi. Sembol yanlış veya bu tarih aralığında işlem görmüyor.",
            hint="Sembolün '.IS' ile bittiğinden ve BIST'te işlem gördüğünden emin olun.",
            symbol=symbol
        )

    # ── Minimum satır kontrolü ─────────────────────────────
    MIN_ROWS = 30
    if len(df) < MIN_ROWS:
        raise InsufficientDataError(
            f"{symbol} için yalnızca {len(df)} günlük veri bulundu. Model için en az {MIN_ROWS} gün gerekli.",
            hint="Tarih aralığını genişletin.",
            symbol=symbol, rows=len(df)
        )

    # ── Veri temizleme ─────────────────────────────────────
    df.columns = df.columns.str.lower()
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    df = df[required_columns]
    df = df.reset_index()
    df.columns = ['date'] + required_columns

    logger.info("Hisse verisi çekildi: %s satır (%s)", len(df), symbol)
    return df



def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Teknik göstergeleri hesaplar.

    Args:
        df: OHLCV verileri içeren DataFrame

    Returns:
        Teknik göstergeler eklenmiş DataFrame

    Indicators:
        Trend      : EMA_10, EMA_20, EMA_50
        Momentum   : RSI_14, MACD, MACD_signal, MACD_hist
                     Stochastic_%K, Stochastic_%D
        Volatilite : Bollinger Upper/Middle/Lower, BB_width, BB_pct_b
                     ATR_14
        Hacim      : Volume_MA_20, OBV
    """
    logger.info("Teknik göstergeler hesaplanıyor...")

    df = df.copy()

    # ── Trend: EMA ────────────────────────────────────────────────
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # ── Momentum: RSI ─────────────────────────────────────────────
    df['rsi_14'] = _calculate_rsi(df['close'], period=14)

    # ── Momentum: MACD ───────────────────────────────────────────
    df['macd'], df['macd_signal'] = _calculate_macd(df['close'])
    df['macd_hist'] = df['macd'] - df['macd_signal']  # histogram = momentum değişim hızı

    # ── Momentum: Stochastic Oscillator ───────────────────────────
    df['stoch_k'], df['stoch_d'] = _calculate_stochastic(
        df['high'], df['low'], df['close'],
        k_period=14, d_period=3
    )

    # ── Volatilite: Bollinger Bands ───────────────────────────────
    bb_upper, bb_middle, bb_lower = _calculate_bollinger_bands(
        df['close'], period=20, std_dev=2.0
    )
    df['bb_upper']  = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower']  = bb_lower

    # Türetilmiş Bollinger özellikleri
    bb_range = bb_upper - bb_lower
    df['bb_width'] = np.where(
        bb_middle > 0,
        bb_range / bb_middle,   # bantların göreceli genişliği
        0.0
    )
    df['bb_pct_b'] = np.where(
        bb_range > 0,
        (df['close'] - bb_lower) / bb_range,  # fiyatın bantlar içindeki konumu (0-1)
        0.5
    )

    # ── Volatilite: ATR ───────────────────────────────────────────
    df['atr_14'] = _calculate_atr(df['high'], df['low'], df['close'], period=14)

    # ATR normalize: mutlak TL değeri yerine fiyata oranı (model karşılaştırması için)
    df['atr_pct'] = np.where(
        df['close'] > 0,
        df['atr_14'] / df['close'] * 100,
        0.0
    )

    # ── Hacim: Volume MA ve OBV ───────────────────────────────────
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['obv']          = _calculate_obv(df['close'], df['volume'])

    # OBV normalize: ham OBV çok büyük sayılar, log ölçeğe al
    # log1p kullan: OBV negatif olabilir → shift ile pozitif yap
    obv_min = df['obv'].min()
    if obv_min < 0:
        df['obv_norm'] = np.log1p(df['obv'] - obv_min)
    else:
        df['obv_norm'] = np.log1p(df['obv'])

    logger.info(
        "Teknik göstergeler hesaplandı: %d gösterge eklendi",
        len([c for c in df.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume']])
    )

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



def _calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands hesaplar.

    Üç bant döndürür:
        - Orta bant  : period günlük SMA
        - Üst bant   : SMA + std_dev × standart sapma
        - Alt bant   : SMA − std_dev × standart sapma

    LSTM için eklenen sütunlar:
        bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b

    bb_width  = (üst − alt) / orta  → volatilite ölçümü
    bb_pct_b  = (fiyat − alt) / (üst − alt)  → 0-1, bantlar içindeki konum

    Args:
        prices:  Kapanış fiyatı serisi
        period:  SMA ve std periyodu (varsayılan: 20)
        std_dev: Standart sapma çarpanı (varsayılan: 2.0)

    Returns:
        (upper, middle, lower) tuple
    """
    middle = prices.rolling(window=period).mean()
    std    = prices.rolling(window=period).std()
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std
    return upper, middle, lower


def _calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K ve %D) hesaplar.

    %K = (close − lowest_low) / (highest_high − lowest_low) × 100
    %D = %K'nın d_period günlük SMA'sı (sinyal çizgisi)

    Yorumlama:
        > 80 → aşırı alım bölgesi
        < 20 → aşırı satım bölgesi
        %K, %D'yi yukarı keserse alım sinyali

    Args:
        high, low, close: OHLC serileri
        k_period:         %K pencere boyutu (varsayılan: 14)
        d_period:         %D smoothing periyodu (varsayılan: 3)

    Returns:
        (stoch_k, stoch_d) tuple
    """
    lowest_low   = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    denom = highest_high - lowest_low
    # Sıfıra bölme koruması
    stoch_k = np.where(
        denom > 0,
        (close - lowest_low) / denom * 100,
        50.0  # Düz piyasada nötr değer
    )
    stoch_k = pd.Series(stoch_k, index=close.index)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def _calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    ATR (Average True Range) hesaplar — volatilite ölçümü.

    True Range = max(
        high − low,
        |high − prev_close|,
        |low  − prev_close|
    )
    ATR = TR'nin period günlük EMA'sı

    Yorumlama:
        Yüksek ATR → yüksek volatilite → geniş güven bandı beklenmeli
        Düşük ATR  → düşük volatilite → fiyat sıkıştı

    MC Dropout güven bandıyla birlikte: yüksek ATR günlerinde
    güven aralığı doğal olarak genişler.

    Args:
        high, low, close: OHLC serileri
        period:           EMA periyodu (varsayılan: 14)

    Returns:
        ATR serisi
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def _calculate_obv(
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    OBV (On-Balance Volume) hesaplar — hacim bazlı momentum.

    Kural:
        close[t] > close[t-1] → OBV += volume[t]
        close[t] < close[t-1] → OBV -= volume[t]
        close[t] == close[t-1]→ OBV değişmez

    Yorumlama:
        OBV artıyor, fiyat artmıyorsa → boğa sapması (fiyat yükselebilir)
        OBV düşüyor, fiyat düşmüyorsa → ayı sapması (fiyat düşebilir)

    Args:
        close:  Kapanış fiyatı serisi
        volume: İşlem hacmi serisi

    Returns:
        OBV serisi (LSTM için normalize edilecek)
    """
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    return obv


def find_optimal_sentiment_lag(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    max_lag: int = 7,
    target_column: str = 'close'
) -> dict:
    """
    Cross-correlation analizi ile haber duygu skorunun fiyata olan
    optimal gecikme (lag) süresini bulur.

    Mantık:
        Bir haberin etkisi aynı gün değil, 1-3 gün sonra fiyata
        yansıyabilir. Bu fonksiyon 0'dan max_lag'e kadar her gecikme
        değeri için |korelasyon|'u hesaplar ve en yüksek korelasyonlu
        lag'i döndürür.

    Args:
        price_df:      OHLCV fiyat verisi (date, close sütunları gerekli)
        sentiment_df:  Haber duygu skorları (date, compound sütunları gerekli)
        max_lag:       Test edilecek maksimum gecikme (gün). Varsayılan: 7
        target_column: Fiyat sütunu adı. Varsayılan: 'close'

    Returns:
        {
            'optimal_lag':    int   — en yüksek korelasyonlu gecikme (gün),
            'correlations':   dict  — {lag: correlation} tüm lag değerleri,
            'best_corr':      float — optimal lag'deki korelasyon değeri,
            'interpretation': str   — Türkçe yorum
        }

    Örnek:
        result = find_optimal_sentiment_lag(price_df, sentiment_df, max_lag=5)
        lag = result['optimal_lag']  # örn: 2 → haber 2 gün sonra fiyata yansıyor
    """
    logger.info("Cross-correlation analizi başlıyor (max_lag=%d)...", max_lag)

    # ── Tarih normalizasyonu ───────────────────────────────────────
    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()

    price_df['date'] = pd.to_datetime(price_df['date']).dt.normalize()
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()

    # ── Günlük ortalama compound skoru ────────────────────────────
    if 'compound' not in sentiment_df.columns:
        logger.warning("'compound' sütunu bulunamadı, lag analizi atlanıyor")
        return {'optimal_lag': 0, 'correlations': {}, 'best_corr': 0.0,
                'interpretation': 'Duygu verisi yetersiz, lag=0 kullanıldı.'}

    daily_sent = (
        sentiment_df.groupby('date')['compound']
        .mean()
        .reset_index()
        .rename(columns={'compound': 'sentiment'})
    )

    # ── Fiyat değişimi (%) ────────────────────────────────────────
    if target_column not in price_df.columns:
        logger.warning("'%s' sütunu bulunamadı, lag analizi atlanıyor", target_column)
        return {'optimal_lag': 0, 'correlations': {}, 'best_corr': 0.0,
                'interpretation': 'Fiyat verisi yetersiz, lag=0 kullanıldı.'}

    price_chg = price_df[['date', target_column]].copy()
    price_chg['price_change'] = price_chg[target_column].pct_change()

    # ── Her lag için Pearson korelasyonu ──────────────────────────
    correlations: dict[int, float] = {}

    for lag in range(0, max_lag + 1):
        # Sentiment'i lag gün öne kaydır: bugünkü haber → lag gün sonraki fiyat
        shifted = daily_sent.copy()
        shifted['date'] = shifted['date'] + pd.Timedelta(days=lag)

        merged = price_chg.merge(shifted, on='date', how='inner')
        merged = merged.dropna(subset=['price_change', 'sentiment'])

        if len(merged) < 10:
            # Yeterli örtüşme yok
            correlations[lag] = 0.0
            continue

        corr = merged['price_change'].corr(merged['sentiment'])
        correlations[lag] = round(float(corr) if not np.isnan(corr) else 0.0, 4)

        logger.debug("lag=%d  |  n=%d  |  corr=%.4f", lag, len(merged), correlations[lag])

    # ── En iyi lag: mutlak korelasyon maksimumu ───────────────────
    optimal_lag = max(correlations, key=lambda k: abs(correlations[k]))
    best_corr   = correlations[optimal_lag]

    # ── Yorum ─────────────────────────────────────────────────────
    if abs(best_corr) < 0.05:
        interp = (
            f"Duygu skorları ile fiyat arasında anlamlı bir korelasyon bulunamadı "
            f"(en yüksek |r|={abs(best_corr):.3f}). lag=0 önerilir."
        )
    elif optimal_lag == 0:
        interp = (
            f"Haberler aynı gün fiyata yansıyor (lag=0, r={best_corr:+.3f}). "
            "Mevcut birleştirme doğru."
        )
    else:
        direction = "pozitif" if best_corr > 0 else "negatif"
        interp = (
            f"Haberler {optimal_lag} gün sonra fiyata yansıyor "
            f"(lag={optimal_lag}, r={best_corr:+.3f}, {direction} korelasyon). "
            "Sentiment skorları bu kadar ötelenirse model doğruluğu artabilir."
        )

    logger.info(
        "Lag analizi tamamlandı: optimal_lag=%d, corr=%.4f | %s",
        optimal_lag, best_corr, interp
    )

    return {
        'optimal_lag':    optimal_lag,
        'correlations':   correlations,
        'best_corr':      best_corr,
        'interpretation': interp
    }


def merge_sentiment_with_prices(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    lag: int = 0
) -> pd.DataFrame:
    """
    Duygu skorlarını fiyat verileriyle birleştirir.

    Args:
        price_df:     Fiyat ve teknik göstergeler
        sentiment_df: Duygu skorları (date, positive, negative, neutral, compound sütunları)
        lag:          Sentiment kaydırma (gün). 0 = aynı gün, 1 = bir gün sonraki fiyata etkisi.
                      find_optimal_sentiment_lag() çıktısından alınabilir.

    Returns:
        Birleştirilmiş DataFrame (günlük ortalama duygu skorları)
    """
    logger.info("Duygu skorları fiyat verileriyle birleştiriliyor (lag=%d)...", lag)

    # DataFrame kopyaları oluştur
    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()

    # Tarih sütunlarını datetime'a çevir
    if 'date' in price_df.columns:
        price_df['date'] = pd.to_datetime(price_df['date'])
    else:
        if isinstance(price_df.index, pd.DatetimeIndex):
            price_df = price_df.reset_index()
            price_df.columns = ['date'] + list(price_df.columns[1:])

    if 'date' in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    else:
        if isinstance(sentiment_df.index, pd.DatetimeIndex):
            sentiment_df = sentiment_df.reset_index()
            sentiment_df.columns = ['date'] + list(sentiment_df.columns[1:])

    # Tarihleri sadece gün seviyesine indir (saat bilgisini kaldır)
    price_df['date']     = price_df['date'].dt.date
    sentiment_df['date'] = sentiment_df['date'].dt.date

    # Duygu skorlarını günlük ortalama olarak hesapla
    sentiment_columns = ['positive', 'negative', 'neutral', 'compound']
    available_sentiment_cols = [col for col in sentiment_columns if col in sentiment_df.columns]

    if not available_sentiment_cols:
        logger.warning("Duygu skoru sütunları bulunamadı, boş skorlar ekleniyor")
        for col in sentiment_columns:
            price_df[f'sentiment_{col}'] = 0.0
        return price_df

    daily_sentiment = sentiment_df.groupby('date')[available_sentiment_cols].mean().reset_index()

    # ── LAG uygula: sentiment tarihlerini ileriye kaydır ──────────
    # lag=1 → bugünkü haber yarının fiyatıyla eşleştirilir
    if lag > 0:
        daily_sentiment['date'] = (
            pd.to_datetime(daily_sentiment['date']) + pd.Timedelta(days=lag)
        ).dt.date
        logger.info("Sentiment %d gün ötelendi (lag=%d)", lag, lag)

    # Sütun isimlerini yeniden adlandır (sentiment_ prefix ekle)
    rename_dict = {col: f'sentiment_{col}' for col in available_sentiment_cols}
    daily_sentiment = daily_sentiment.rename(columns=rename_dict)

    # Fiyat verileriyle birleştir (left join — tüm borsa günleri korunur)
    merged_df = price_df.merge(daily_sentiment, on='date', how='left')

    # Eksik duygu skorlarını 0 ile doldur (haberin olmadığı günler için)
    sentiment_cols_in_merged = [f'sentiment_{col}' for col in available_sentiment_cols]
    merged_df[sentiment_cols_in_merged] = merged_df[sentiment_cols_in_merged].fillna(0.0)

    # Haber olan / olmayan günleri flag'le (model 0 skoru ile haber yokluğunu ayırt eder)
    merged_df['has_sentiment'] = merged_df['date'].isin(
        daily_sentiment['date']
    ).astype(float)

    # Tarihi tekrar datetime'a çevir
    merged_df['date'] = pd.to_datetime(merged_df['date'])

    logger.info("Birleştirme tamamlandı: %d satır (lag=%d)", len(merged_df), lag)
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
