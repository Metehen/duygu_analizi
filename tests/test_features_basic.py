"""
Birim Testleri: Özellik Hazırlama Modülü

Bu dosya, features.py modülündeki fonksiyonlar için birim testleri içerir.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# src dizinini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import (
    fetch_stock_data,
    calculate_technical_indicators,
    merge_sentiment_with_prices,
    prepare_training_data
)


class TestFetchStockData:
    """fetch_stock_data fonksiyonu için testler."""
    
    def test_fetch_stock_data_invalid_date_range(self):
        """Geçersiz tarih aralığı testi - bitiş tarihi başlangıçtan önce."""
        with pytest.raises(ValueError, match="Başlangıç tarihi bitiş tarihinden önce olmalıdır"):
            fetch_stock_data("THYAO.IS", "2024-01-01", "2023-01-01")
    
    def test_fetch_stock_data_same_dates(self):
        """Geçersiz tarih aralığı testi - aynı başlangıç ve bitiş tarihi."""
        with pytest.raises(ValueError):
            fetch_stock_data("THYAO.IS", "2024-01-01", "2024-01-01")


class TestCalculateTechnicalIndicators:
    """calculate_technical_indicators fonksiyonu için testler."""
    
    def test_technical_indicators_structure(self):
        """Teknik gösterge sütunlarının varlığı testi."""
        # Örnek veri oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(95, 115, 100),
            'low': np.random.uniform(85, 105, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        result = calculate_technical_indicators(df)
        
        # Gerekli sütunların varlığını kontrol et
        assert 'ema_10' in result.columns
        assert 'ema_20' in result.columns
        assert 'ema_50' in result.columns
        assert 'rsi_14' in result.columns
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'volume_ma_20' in result.columns
    
    def test_rsi_range(self):
        """RSI değerlerinin 0-100 aralığında olduğunu test et."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(95, 115, 100),
            'low': np.random.uniform(85, 105, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        result = calculate_technical_indicators(df)
        
        # NaN olmayan RSI değerlerini kontrol et
        rsi_values = result['rsi_14'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
    
    def test_technical_indicators_single_row(self):
        """Tek satırlık veri için teknik gösterge testi."""
        df = pd.DataFrame({
            'date': [datetime.now()],
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000000]
        })
        
        result = calculate_technical_indicators(df)
        
        # Sütunlar oluşturulmalı (NaN olabilir)
        assert 'ema_10' in result.columns
        assert 'rsi_14' in result.columns


class TestMergeSentimentWithPrices:
    """merge_sentiment_with_prices fonksiyonu için testler."""
    
    def test_merge_basic(self):
        """Temel birleştirme testi."""
        # Fiyat verisi
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        price_df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(90, 110, 10),
            'high': np.random.uniform(95, 115, 10),
            'low': np.random.uniform(85, 105, 10),
            'close': np.random.uniform(90, 110, 10),
            'volume': np.random.randint(1000000, 10000000, 10)
        })
        
        # Duygu verisi
        sentiment_df = pd.DataFrame({
            'date': dates[:5],  # Sadece ilk 5 gün
            'positive': np.random.uniform(0, 1, 5),
            'negative': np.random.uniform(0, 1, 5),
            'neutral': np.random.uniform(0, 1, 5),
            'compound': np.random.uniform(-1, 1, 5)
        })
        
        result = merge_sentiment_with_prices(price_df, sentiment_df)
        
        # Tüm fiyat verileri korunmalı
        assert len(result) == len(price_df)
        
        # Duygu sütunları eklenmeli
        assert 'sentiment_positive' in result.columns
        assert 'sentiment_negative' in result.columns
        assert 'sentiment_neutral' in result.columns
        assert 'sentiment_compound' in result.columns
    
    def test_merge_empty_sentiment(self):
        """Boş duygu verisi ile birleştirme testi."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        price_df = pd.DataFrame({
            'date': dates,
            'close': [100, 101, 102, 103, 104]
        })
        
        sentiment_df = pd.DataFrame({
            'date': [],
            'positive': [],
            'negative': [],
            'neutral': [],
            'compound': []
        })
        
        result = merge_sentiment_with_prices(price_df, sentiment_df)
        
        # Fiyat verileri korunmalı
        assert len(result) == len(price_df)
    
    def test_merge_single_row_price(self):
        """Tek satırlık fiyat verisi ile birleştirme testi."""
        price_df = pd.DataFrame({
            'date': [datetime.now()],
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000000]
        })
        
        sentiment_df = pd.DataFrame({
            'date': [datetime.now()],
            'positive': [0.8],
            'negative': [0.1],
            'neutral': [0.1],
            'compound': [0.7]
        })
        
        result = merge_sentiment_with_prices(price_df, sentiment_df)
        
        # Tek satır korunmalı
        assert len(result) == 1
        assert 'sentiment_positive' in result.columns


class TestPrepareTrainingData:
    """prepare_training_data fonksiyonu için testler."""
    
    def test_prepare_training_data_dimensions(self):
        """Eğitim verisi boyutları testi."""
        # Örnek veri oluştur
        num_samples = 100
        lookback = 60
        
        dates = pd.date_range(start='2023-01-01', periods=num_samples, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'close': np.random.uniform(90, 110, num_samples),
            'volume': np.random.randint(1000000, 10000000, num_samples),
            'ema_10': np.random.uniform(90, 110, num_samples),
            'rsi_14': np.random.uniform(0, 100, num_samples)
        })
        
        X, y, scaler = prepare_training_data(df, target_column='close', lookback=lookback)
        
        # Boyutları kontrol et
        expected_samples = num_samples - lookback
        assert X.shape[0] == expected_samples
        assert X.shape[1] == lookback
        assert X.shape[2] > 0  # En az bir özellik
        assert y.shape[0] == expected_samples
    
    def test_prepare_training_data_insufficient_data(self):
        """Yetersiz veri testi - lookback'ten az veri."""
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'close': np.random.uniform(90, 110, 10),
            'volume': np.random.randint(1000000, 10000000, 10)
        })
        
        with pytest.raises(ValueError, match="Yetersiz veri"):
            prepare_training_data(df, lookback=60)
    
    def test_prepare_training_data_exactly_minimum_data(self):
        """Yetersiz veri testi - tam minimum veri (lookback + 1)."""
        lookback = 10
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=lookback + 1, freq='D'),
            'close': np.random.uniform(90, 110, lookback + 1),
            'volume': np.random.randint(1000000, 10000000, lookback + 1)
        })
        
        # Bu durumda sadece 1 sample oluşmalı
        X, y, scaler = prepare_training_data(df, lookback=lookback)
        assert X.shape[0] == 1
        assert y.shape[0] == 1
    
    def test_prepare_training_data_one_less_than_minimum(self):
        """Yetersiz veri testi - minimum'dan 1 eksik veri."""
        lookback = 10
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=lookback, freq='D'),
            'close': np.random.uniform(90, 110, lookback),
            'volume': np.random.randint(1000000, 10000000, lookback)
        })
        
        with pytest.raises(ValueError, match="Yetersiz veri"):
            prepare_training_data(df, lookback=lookback)
    
    def test_prepare_training_data_invalid_target(self):
        """Geçersiz hedef sütun testi."""
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'close': np.random.uniform(90, 110, 100)
        })
        
        with pytest.raises(ValueError, match="Hedef sütun bulunamadı"):
            prepare_training_data(df, target_column='invalid_column', lookback=10)
    
    def test_prepare_training_data_normalization(self):
        """Normalizasyon testi (0-1 aralığı)."""
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'close': np.random.uniform(90, 110, 100),
            'feature1': np.random.uniform(0, 1000, 100)
        })
        
        X, y, scaler = prepare_training_data(df, target_column='close', lookback=10)
        
        # X değerleri 0-1 aralığında olmalı
        assert X.min() >= 0
        assert X.max() <= 1
    
    def test_prepare_training_data_single_row(self):
        """Tek satırlık veri testi - yetersiz veri hatası vermeli."""
        df = pd.DataFrame({
            'date': [datetime.now()],
            'close': [100],
            'volume': [1000000]
        })
        
        with pytest.raises(ValueError, match="Yetersiz veri"):
            prepare_training_data(df, lookback=10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
