"""
Özellik Tabanlı Testler: Özellik Hazırlama Modülü

Bu dosya, features.py modülündeki fonksiyonlar için özellik tabanlı testleri içerir.
Hypothesis kütüphanesi kullanılarak evrensel doğruluk özellikleri test edilir.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from hypothesis import given, strategies as st, settings, assume
import tempfile

# src dizinini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import (
    calculate_technical_indicators,
    merge_sentiment_with_prices,
    prepare_training_data
)


# Yardımcı fonksiyonlar: Random DataFrame oluşturma

def generate_random_ohlcv_dataframe(num_samples: int) -> pd.DataFrame:
    """
    Random OHLCV DataFrame oluşturur.
    
    Args:
        num_samples: Satır sayısı
    
    Returns:
        OHLCV verileri içeren DataFrame
    """
    dates = pd.date_range(start='2023-01-01', periods=num_samples, freq='D')
    
    # Gerçekçi fiyat verileri oluştur
    base_price = 100.0
    close_prices = []
    
    for i in range(num_samples):
        # Random walk ile fiyat oluştur
        change = np.random.uniform(-2, 2)
        base_price = max(base_price + change, 1.0)  # Fiyat pozitif kalmalı
        close_prices.append(base_price)
    
    close_prices = np.array(close_prices)
    
    df = pd.DataFrame({
        'date': dates,
        'open': close_prices * np.random.uniform(0.98, 1.02, num_samples),
        'high': close_prices * np.random.uniform(1.00, 1.05, num_samples),
        'low': close_prices * np.random.uniform(0.95, 1.00, num_samples),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, num_samples)
    })
    
    return df


def generate_random_sentiment_dataframe(num_samples: int) -> pd.DataFrame:
    """
    Random duygu skoru DataFrame oluşturur.
    
    Args:
        num_samples: Satır sayısı
    
    Returns:
        Duygu skorları içeren DataFrame
    """
    dates = pd.date_range(start='2023-01-01', periods=num_samples, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'positive': np.random.uniform(0, 1, num_samples),
        'negative': np.random.uniform(0, 1, num_samples),
        'neutral': np.random.uniform(0, 1, num_samples),
        'compound': np.random.uniform(-1, 1, num_samples)
    })
    
    return df


def generate_random_feature_dataframe(num_samples: int, num_features: int) -> pd.DataFrame:
    """
    Random özellik DataFrame oluşturur.
    
    Args:
        num_samples: Satır sayısı
        num_features: Özellik sayısı
    
    Returns:
        Özellikler içeren DataFrame
    """
    dates = pd.date_range(start='2023-01-01', periods=num_samples, freq='D')
    
    data = {'date': dates}
    
    # Hedef sütun (close)
    data['close'] = np.random.uniform(90, 110, num_samples)
    
    # Diğer özellikler
    for i in range(num_features):
        data[f'feature_{i}'] = np.random.uniform(0, 100, num_samples)
    
    return pd.DataFrame(data)


# Property 4: Teknik Gösterge Hesaplama
# **Validates: Requirements 3.2**

@given(st.integers(min_value=100, max_value=500))
@settings(max_examples=100, deadline=None)
def test_property_4_technical_indicators_structure(num_samples):
    """
    Feature: turkish-sentiment-stock-prediction, Property 4:
    Herhangi bir geçerli OHLCV fiyat DataFrame'i için, teknik gösterge 
    hesaplama fonksiyonu çağrıldığında, dönen DataFrame şu ek sütunları 
    içermelidir: ema_10, ema_20, ema_50, rsi_14, macd, macd_signal, 
    volume_ma_20. Tüm değerler numeric olmalı ve RSI değerleri 0-100 
    aralığında olmalıdır.
    
    **Validates: Requirements 3.2**
    """
    # Random OHLCV DataFrame oluştur
    df = generate_random_ohlcv_dataframe(num_samples)
    
    # Teknik göstergeleri hesapla
    result = calculate_technical_indicators(df)
    
    # Gerekli sütunların varlığını kontrol et
    required_columns = ['ema_10', 'ema_20', 'ema_50', 'rsi_14', 
                       'macd', 'macd_signal', 'volume_ma_20']
    
    for col in required_columns:
        assert col in result.columns, f"Sütun bulunamadı: {col}"
    
    # Tüm değerlerin numeric olduğunu kontrol et
    for col in required_columns:
        assert pd.api.types.is_numeric_dtype(result[col]), \
            f"Sütun numeric değil: {col}"
    
    # RSI değerlerinin 0-100 aralığında olduğunu kontrol et
    rsi_values = result['rsi_14'].dropna()
    if len(rsi_values) > 0:
        assert (rsi_values >= 0).all(), "RSI değerleri 0'dan küçük"
        assert (rsi_values <= 100).all(), "RSI değerleri 100'den büyük"


@given(st.integers(min_value=100, max_value=500))
@settings(max_examples=100, deadline=None)
def test_property_4_technical_indicators_no_data_loss(num_samples):
    """
    Property 4 ek test: Teknik gösterge hesaplama, orijinal satır sayısını korumalıdır.
    
    **Validates: Requirements 3.2**
    """
    # Random OHLCV DataFrame oluştur
    df = generate_random_ohlcv_dataframe(num_samples)
    
    # Teknik göstergeleri hesapla
    result = calculate_technical_indicators(df)
    
    # Satır sayısının korunduğunu kontrol et
    assert len(result) == len(df), \
        f"Satır sayısı değişti: {len(df)} -> {len(result)}"
    
    # Orijinal sütunların korunduğunu kontrol et
    original_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in original_columns:
        assert col in result.columns, f"Orijinal sütun kayboldu: {col}"


# Property 5: Duygu-Fiyat Birleştirme
# **Validates: Requirements 3.3**

@given(
    st.integers(min_value=50, max_value=200),  # price_samples
    st.integers(min_value=20, max_value=100)   # sentiment_samples
)
@settings(max_examples=100, deadline=None)
def test_property_5_sentiment_price_merge_structure(price_samples, sentiment_samples):
    """
    Feature: turkish-sentiment-stock-prediction, Property 5:
    Herhangi bir fiyat DataFrame'i ve duygu DataFrame'i için, birleştirme 
    fonksiyonu çağrıldığında, dönen DataFrame hem fiyat sütunlarını hem de 
    günlük ortalama duygu skorlarını (sentiment_positive, sentiment_negative, 
    sentiment_neutral, sentiment_compound) içermelidir. Tarih indeksi her iki 
    kaynaktan da verileri içermelidir.
    
    **Validates: Requirements 3.3**
    """
    # Random fiyat ve duygu DataFrame'leri oluştur
    price_df = generate_random_ohlcv_dataframe(price_samples)
    sentiment_df = generate_random_sentiment_dataframe(sentiment_samples)
    
    # Birleştirme yap
    result = merge_sentiment_with_prices(price_df, sentiment_df)
    
    # Fiyat sütunlarının varlığını kontrol et
    price_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in price_columns:
        assert col in result.columns, f"Fiyat sütunu bulunamadı: {col}"
    
    # Duygu sütunlarının varlığını kontrol et
    sentiment_columns = ['sentiment_positive', 'sentiment_negative', 
                        'sentiment_neutral', 'sentiment_compound']
    for col in sentiment_columns:
        assert col in result.columns, f"Duygu sütunu bulunamadı: {col}"
    
    # Tüm fiyat verilerinin korunduğunu kontrol et (left join)
    assert len(result) == len(price_df), \
        f"Fiyat verileri kayboldu: {len(price_df)} -> {len(result)}"
    
    # Tarih sütununun varlığını kontrol et
    assert 'date' in result.columns, "Tarih sütunu bulunamadı"


@given(st.integers(min_value=50, max_value=200))
@settings(max_examples=100, deadline=None)
def test_property_5_sentiment_merge_handles_missing_data(num_samples):
    """
    Property 5 ek test: Birleştirme, eksik duygu verilerini uygun şekilde 
    ele almalıdır (0 ile doldurma).
    
    **Validates: Requirements 3.3**
    """
    # Fiyat verisi oluştur
    price_df = generate_random_ohlcv_dataframe(num_samples)
    
    # Sadece ilk yarısı için duygu verisi oluştur
    sentiment_samples = num_samples // 2
    sentiment_df = generate_random_sentiment_dataframe(sentiment_samples)
    
    # Birleştirme yap
    result = merge_sentiment_with_prices(price_df, sentiment_df)
    
    # Tüm satırların varlığını kontrol et
    assert len(result) == len(price_df)
    
    # Duygu sütunlarının NaN içermediğini kontrol et (0 ile doldurulmalı)
    sentiment_columns = ['sentiment_positive', 'sentiment_negative', 
                        'sentiment_neutral', 'sentiment_compound']
    for col in sentiment_columns:
        assert not result[col].isnull().any(), \
            f"Duygu sütunu NaN içeriyor: {col}"


# Property 6: Eğitim Verisi Boyutları
# **Validates: Requirements 3.4**

@given(
    st.integers(min_value=100, max_value=500),  # num_samples
    st.integers(min_value=3, max_value=10),     # num_features
    st.integers(min_value=10, max_value=60)     # lookback
)
@settings(max_examples=100, deadline=None)
def test_property_6_training_data_dimensions(num_samples, num_features, lookback):
    """
    Feature: turkish-sentiment-stock-prediction, Property 6:
    Herhangi bir geçerli özellik DataFrame'i ve lookback parametresi için, 
    eğitim verisi hazırlama fonksiyonu çağrıldığında, dönen X array'inin 
    şekli (samples, lookback, features) ve y array'inin şekli (samples,) 
    olmalıdır. samples sayısı, orijinal DataFrame uzunluğundan lookback 
    kadar az olmalıdır.
    
    **Validates: Requirements 3.4**
    """
    # Yeterli veri olduğundan emin ol
    assume(num_samples > lookback)
    
    # Random özellik DataFrame oluştur
    df = generate_random_feature_dataframe(num_samples, num_features)
    
    # Eğitim verisi hazırla
    X, y, scaler = prepare_training_data(df, target_column='close', lookback=lookback)
    
    # Beklenen sample sayısı
    expected_samples = num_samples - lookback
    
    # X boyutlarını kontrol et
    assert X.shape[0] == expected_samples, \
        f"X sample sayısı yanlış: beklenen {expected_samples}, alınan {X.shape[0]}"
    assert X.shape[1] == lookback, \
        f"X lookback boyutu yanlış: beklenen {lookback}, alınan {X.shape[1]}"
    assert X.shape[2] == num_features, \
        f"X feature sayısı yanlış: beklenen {num_features}, alınan {X.shape[2]}"
    
    # y boyutlarını kontrol et
    assert y.shape[0] == expected_samples, \
        f"y sample sayısı yanlış: beklenen {expected_samples}, alınan {y.shape[0]}"
    assert len(y.shape) == 1, \
        f"y boyutu yanlış: beklenen 1D, alınan {len(y.shape)}D"


@given(
    st.integers(min_value=100, max_value=500),
    st.integers(min_value=3, max_value=10),
    st.integers(min_value=10, max_value=60)
)
@settings(max_examples=100, deadline=None)
def test_property_6_training_data_normalization(num_samples, num_features, lookback):
    """
    Property 6 ek test: Eğitim verisi normalizasyonu 0-1 aralığında olmalıdır.
    
    **Validates: Requirements 3.4**
    """
    # Yeterli veri olduğundan emin ol
    assume(num_samples > lookback)
    
    # Random özellik DataFrame oluştur
    df = generate_random_feature_dataframe(num_samples, num_features)
    
    # Eğitim verisi hazırla
    X, y, scaler = prepare_training_data(df, target_column='close', lookback=lookback)
    
    # X değerlerinin 0-1 aralığında olduğunu kontrol et (küçük tolerans ile)
    tolerance = 1e-10
    assert X.min() >= -tolerance, f"X minimum değeri 0'dan küçük: {X.min()}"
    assert X.max() <= 1 + tolerance, f"X maksimum değeri 1'den büyük: {X.max()}"
    
    # Scaler'ın MinMaxScaler olduğunu kontrol et
    from sklearn.preprocessing import MinMaxScaler
    assert isinstance(scaler, MinMaxScaler), \
        f"Scaler tipi yanlış: {type(scaler)}"


@given(
    st.integers(min_value=100, max_value=500),
    st.integers(min_value=3, max_value=10),
    st.integers(min_value=10, max_value=60)
)
@settings(max_examples=100, deadline=None)
def test_property_6_training_data_no_nan(num_samples, num_features, lookback):
    """
    Property 6 ek test: Eğitim verisi NaN içermemelidir.
    
    **Validates: Requirements 3.4**
    """
    # Yeterli veri olduğundan emin ol
    assume(num_samples > lookback)
    
    # Random özellik DataFrame oluştur
    df = generate_random_feature_dataframe(num_samples, num_features)
    
    # Eğitim verisi hazırla
    X, y, scaler = prepare_training_data(df, target_column='close', lookback=lookback)
    
    # X ve y'nin NaN içermediğini kontrol et
    assert not np.isnan(X).any(), "X NaN içeriyor"
    assert not np.isnan(y).any(), "y NaN içeriyor"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
