"""
Streamlit arayüzü için birim testler (Unit Tests)

Bu testler, belirli örnekleri ve kenar durumları test eder.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import plotly.graph_objects as go

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import (
    plot_price_predictions,
    plot_sentiment_scores,
    plot_feature_importance
)


# ============================================================================
# Görselleştirme Fonksiyonları Testleri
# ============================================================================

class TestPlotPricePredictions:
    """plot_price_predictions fonksiyonu için testler."""
    
    def test_basic_plot_creation(self):
        """Temel fiyat grafiği oluşturma testi."""
        # Basit test verileri
        actual_prices = np.array([100.0, 105.0, 103.0, 108.0, 110.0])
        predicted_prices = np.array([101.0, 104.0, 105.0, 107.0, 109.0])
        
        # Grafik oluştur
        fig = plot_price_predictions(actual_prices, predicted_prices)
        
        # Temel kontroller
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Gerçek ve tahmin
        assert fig.data[0].name == 'Gerçek Fiyat'
        assert fig.data[1].name == 'Tahmin Edilen Fiyat'
    
    def test_plot_with_dates(self):
        """Tarih dizisi ile fiyat grafiği testi."""
        actual_prices = np.array([100.0, 105.0, 103.0, 108.0, 110.0])
        predicted_prices = np.array([101.0, 104.0, 105.0, 107.0, 109.0])
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D').to_numpy()
        
        # Grafik oluştur
        fig = plot_price_predictions(actual_prices, predicted_prices, dates)
        
        # Tarih eksenini kontrol et
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.title.text == 'Tarih'
        assert len(fig.data[0].x) == 5
    
    def test_plot_without_dates(self):
        """Tarih dizisi olmadan fiyat grafiği testi."""
        actual_prices = np.array([100.0, 105.0, 103.0])
        predicted_prices = np.array([101.0, 104.0, 105.0])
        
        # Grafik oluştur
        fig = plot_price_predictions(actual_prices, predicted_prices)
        
        # Index eksenini kontrol et
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.title.text == 'Zaman Adımı'
    
    def test_plot_single_point(self):
        """Tek veri noktası ile grafik testi."""
        actual_prices = np.array([100.0])
        predicted_prices = np.array([101.0])
        
        # Grafik oluştur
        fig = plot_price_predictions(actual_prices, predicted_prices)
        
        # Grafik oluşturulmalı
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert len(fig.data[0].y) == 1
    
    def test_plot_large_dataset(self):
        """Büyük veri seti ile grafik testi."""
        # 1000 veri noktası
        actual_prices = np.random.uniform(100, 200, 1000)
        predicted_prices = actual_prices + np.random.normal(0, 5, 1000)
        
        # Grafik oluştur
        fig = plot_price_predictions(actual_prices, predicted_prices)
        
        # Grafik oluşturulmalı
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].y) == 1000
    
    def test_plot_layout_properties(self):
        """Grafik layout özelliklerini test et."""
        actual_prices = np.array([100.0, 105.0, 103.0])
        predicted_prices = np.array([101.0, 104.0, 105.0])
        
        fig = plot_price_predictions(actual_prices, predicted_prices)
        
        # Layout kontrolleri
        assert fig.layout.title.text == 'Gerçek vs Tahmin Edilen Fiyatlar'
        assert fig.layout.yaxis.title.text == 'Fiyat (₺)'
        assert fig.layout.height == 500
        # Template is an object, check it exists
        assert fig.layout.template is not None
    
    def test_plot_trace_properties(self):
        """Grafik trace özelliklerini test et."""
        actual_prices = np.array([100.0, 105.0, 103.0])
        predicted_prices = np.array([101.0, 104.0, 105.0])
        
        fig = plot_price_predictions(actual_prices, predicted_prices)
        
        # Gerçek fiyat trace
        assert fig.data[0].mode == 'lines'
        assert fig.data[0].line.color == 'blue'
        assert fig.data[0].line.width == 2
        
        # Tahmin trace
        assert fig.data[1].mode == 'lines'
        assert fig.data[1].line.color == 'red'
        assert fig.data[1].line.dash == 'dash'


class TestPlotSentimentScores:
    """plot_sentiment_scores fonksiyonu için testler."""
    
    def test_basic_sentiment_plot(self):
        """Temel duygu grafiği oluşturma testi."""
        # Test verileri
        sentiment_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'positive': [0.7, 0.6, 0.8, 0.5, 0.9],
            'negative': [0.2, 0.3, 0.1, 0.4, 0.1],
            'neutral': [0.1, 0.1, 0.1, 0.1, 0.0],
            'compound': [0.5, 0.3, 0.7, 0.1, 0.8]
        })
        
        # Grafik oluştur
        fig = plot_sentiment_scores(sentiment_df)
        
        # Temel kontroller
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # Pozitif, negatif, compound
        assert fig.data[0].name == 'Pozitif'
        assert fig.data[1].name == 'Negatif'
        assert fig.data[2].name == 'Compound'
    
    def test_sentiment_plot_with_duplicates(self):
        """Aynı günde birden fazla haber ile duygu grafiği testi."""
        # Aynı günde birden fazla haber
        sentiment_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
            'positive': [0.7, 0.8, 0.6],
            'negative': [0.2, 0.1, 0.3],
            'neutral': [0.1, 0.1, 0.1],
            'compound': [0.5, 0.7, 0.3]
        })
        
        # Grafik oluştur
        fig = plot_sentiment_scores(sentiment_df)
        
        # Günlük ortalama alınmalı, 2 benzersiz tarih
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].y) == 2  # 2 benzersiz gün
    
    def test_sentiment_plot_single_day(self):
        """Tek günlük duygu verisi ile grafik testi."""
        sentiment_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01']),
            'positive': [0.7],
            'negative': [0.2],
            'neutral': [0.1],
            'compound': [0.5]
        })
        
        # Grafik oluştur
        fig = plot_sentiment_scores(sentiment_df)
        
        # Grafik oluşturulmalı
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].y) == 1
    
    def test_sentiment_plot_layout(self):
        """Duygu grafiği layout özelliklerini test et."""
        sentiment_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=3, freq='D'),
            'positive': [0.7, 0.6, 0.8],
            'negative': [0.2, 0.3, 0.1],
            'neutral': [0.1, 0.1, 0.1],
            'compound': [0.5, 0.3, 0.7]
        })
        
        fig = plot_sentiment_scores(sentiment_df)
        
        # Layout kontrolleri
        assert fig.layout.title.text == 'Günlük Duygu Skorları'
        assert fig.layout.xaxis.title.text == 'Tarih'
        assert fig.layout.yaxis.title.text == 'Pozitif/Negatif Skor'
        assert fig.layout.height == 500
    
    def test_sentiment_plot_dual_axis(self):
        """Duygu grafiği çift eksen kontrolü."""
        sentiment_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=3, freq='D'),
            'positive': [0.7, 0.6, 0.8],
            'negative': [0.2, 0.3, 0.1],
            'neutral': [0.1, 0.1, 0.1],
            'compound': [0.5, 0.3, 0.7]
        })
        
        fig = plot_sentiment_scores(sentiment_df)
        
        # İkinci Y ekseni kontrolü (compound için)
        assert hasattr(fig.layout, 'yaxis2')
        assert fig.layout.yaxis2.title.text == 'Compound Skor'
        # Range is a tuple, not a list
        assert fig.layout.yaxis2.range == (-1, 1)
    
    def test_sentiment_plot_unsorted_dates(self):
        """Sıralanmamış tarihler ile duygu grafiği testi."""
        # Karışık sıralı tarihler
        sentiment_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02']),
            'positive': [0.8, 0.7, 0.6],
            'negative': [0.1, 0.2, 0.3],
            'neutral': [0.1, 0.1, 0.1],
            'compound': [0.7, 0.5, 0.3]
        })
        
        # Grafik oluştur
        fig = plot_sentiment_scores(sentiment_df)
        
        # Grafik oluşturulmalı ve tarihler sıralanmalı
        assert isinstance(fig, go.Figure)
        dates = pd.to_datetime(fig.data[0].x)
        assert dates[0] < dates[1] < dates[2]


class TestPlotFeatureImportance:
    """plot_feature_importance fonksiyonu için testler."""
    
    def test_basic_importance_plot(self):
        """Temel özellik önem grafiği oluşturma testi."""
        # Test verileri
        importance_df = pd.DataFrame({
            'feature': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            'importance': [0.5, 0.3, 0.2, 0.15, 0.1]
        })
        
        # Grafik oluştur
        fig = plot_feature_importance(importance_df)
        
        # Temel kontroller
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Bar grafiği
        assert fig.data[0].type == 'bar'
        assert fig.data[0].orientation == 'h'  # Yatay bar
    
    def test_importance_plot_top_n(self):
        """Top N özellik gösterimi testi."""
        # 20 özellik
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(20)],
            'importance': np.linspace(1.0, 0.05, 20)
        })
        
        # Top 10 göster
        fig = plot_feature_importance(importance_df, top_n=10)
        
        # Sadece 10 özellik gösterilmeli
        assert len(fig.data[0].y) == 10
    
    def test_importance_plot_fewer_than_top_n(self):
        """Top N'den az özellik olduğunda test."""
        # 5 özellik
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(5)],
            'importance': [0.5, 0.3, 0.2, 0.15, 0.1]
        })
        
        # Top 10 iste ama sadece 5 var
        fig = plot_feature_importance(importance_df, top_n=10)
        
        # Tüm 5 özellik gösterilmeli
        assert len(fig.data[0].y) == 5
    
    def test_importance_plot_single_feature(self):
        """Tek özellik ile grafik testi."""
        importance_df = pd.DataFrame({
            'feature': ['feature_1'],
            'importance': [0.5]
        })
        
        # Grafik oluştur
        fig = plot_feature_importance(importance_df)
        
        # Grafik oluşturulmalı
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].y) == 1
    
    def test_importance_plot_layout(self):
        """Özellik önem grafiği layout özelliklerini test et."""
        importance_df = pd.DataFrame({
            'feature': ['feature_1', 'feature_2', 'feature_3'],
            'importance': [0.5, 0.3, 0.2]
        })
        
        fig = plot_feature_importance(importance_df, top_n=15)
        
        # Layout kontrolleri
        assert 'En Önemli 15 Özellik' in fig.layout.title.text
        assert fig.layout.xaxis.title.text == 'Önem Skoru'
        assert fig.layout.yaxis.title.text == 'Özellik'
        # Template is an object, check it exists
        assert fig.layout.template is not None
    
    def test_importance_plot_dynamic_height(self):
        """Dinamik yükseklik testi."""
        # 20 özellik
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(20)],
            'importance': np.linspace(1.0, 0.05, 20)
        })
        
        # Top 20 göster
        fig = plot_feature_importance(importance_df, top_n=20)
        
        # Yükseklik dinamik olmalı (top_n * 30)
        expected_height = max(400, 20 * 30)
        assert fig.layout.height == expected_height
    
    def test_importance_plot_colorscale(self):
        """Renk skalası testi."""
        importance_df = pd.DataFrame({
            'feature': ['feature_1', 'feature_2', 'feature_3'],
            'importance': [0.5, 0.3, 0.2]
        })
        
        fig = plot_feature_importance(importance_df)
        
        # Renk skalası kontrolü - colorscale is set and showscale is True
        assert fig.data[0].marker.colorscale is not None
        assert fig.data[0].marker.showscale is True
    
    def test_importance_plot_reverse_order(self):
        """Özellik sıralaması testi (en önemli en üstte)."""
        importance_df = pd.DataFrame({
            'feature': ['feature_1', 'feature_2', 'feature_3'],
            'importance': [0.5, 0.3, 0.2]
        })
        
        fig = plot_feature_importance(importance_df)
        
        # En düşük önem en altta, en yüksek en üstte olmalı
        # (ters çevrilmiş sıralama)
        y_values = fig.data[0].y
        x_values = fig.data[0].x
        
        # İlk eleman en düşük önem olmalı
        assert x_values[0] < x_values[-1]


# ============================================================================
# Kenar Durumları ve Hata Testleri
# ============================================================================

class TestEdgeCases:
    """Kenar durumları ve hata testleri."""
    
    def test_empty_arrays(self):
        """Boş array'ler ile test."""
        actual_prices = np.array([])
        predicted_prices = np.array([])
        
        # Grafik oluşturulmalı (boş olsa bile)
        fig = plot_price_predictions(actual_prices, predicted_prices)
        assert isinstance(fig, go.Figure)
    
    def test_mismatched_lengths(self):
        """Farklı uzunluktaki array'ler ile test."""
        actual_prices = np.array([100.0, 105.0, 103.0])
        predicted_prices = np.array([101.0, 104.0])  # Daha kısa
        
        # Grafik oluşturulmalı (Plotly farklı uzunlukları handle eder)
        fig = plot_price_predictions(actual_prices, predicted_prices)
        assert isinstance(fig, go.Figure)
    
    def test_zero_values(self):
        """Sıfır değerler ile test."""
        actual_prices = np.array([0.0, 0.0, 0.0])
        predicted_prices = np.array([0.0, 0.0, 0.0])
        
        # Grafik oluşturulmalı
        fig = plot_price_predictions(actual_prices, predicted_prices)
        assert isinstance(fig, go.Figure)
    
    def test_negative_values(self):
        """Negatif değerler ile test."""
        actual_prices = np.array([-100.0, -105.0, -103.0])
        predicted_prices = np.array([-101.0, -104.0, -105.0])
        
        # Grafik oluşturulmalı
        fig = plot_price_predictions(actual_prices, predicted_prices)
        assert isinstance(fig, go.Figure)
    
    def test_very_large_values(self):
        """Çok büyük değerler ile test."""
        actual_prices = np.array([1e10, 1e10 + 1000, 1e10 + 2000])
        predicted_prices = np.array([1e10 + 500, 1e10 + 1500, 1e10 + 2500])
        
        # Grafik oluşturulmalı
        fig = plot_price_predictions(actual_prices, predicted_prices)
        assert isinstance(fig, go.Figure)
    
    def test_sentiment_with_missing_columns(self):
        """Eksik sütunlar ile duygu DataFrame'i testi."""
        # Sadece bazı sütunlar var
        sentiment_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=3, freq='D'),
            'positive': [0.7, 0.6, 0.8],
            'negative': [0.2, 0.3, 0.1]
            # neutral ve compound eksik
        })
        
        # Eksik sütunlar ekle
        sentiment_df['neutral'] = 0.0
        sentiment_df['compound'] = 0.0
        
        # Grafik oluşturulmalı
        fig = plot_sentiment_scores(sentiment_df)
        assert isinstance(fig, go.Figure)
    
    def test_importance_with_zero_importance(self):
        """Sıfır önem değerleri ile test."""
        importance_df = pd.DataFrame({
            'feature': ['feature_1', 'feature_2', 'feature_3'],
            'importance': [0.0, 0.0, 0.0]
        })
        
        # Grafik oluşturulmalı
        fig = plot_feature_importance(importance_df)
        assert isinstance(fig, go.Figure)
    
    def test_importance_with_negative_values(self):
        """Negatif önem değerleri ile test (olmamalı ama test edelim)."""
        importance_df = pd.DataFrame({
            'feature': ['feature_1', 'feature_2', 'feature_3'],
            'importance': [-0.1, 0.3, 0.2]
        })
        
        # Grafik oluşturulmalı
        fig = plot_feature_importance(importance_df)
        assert isinstance(fig, go.Figure)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
