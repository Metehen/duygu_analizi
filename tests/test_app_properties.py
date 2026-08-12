"""
Streamlit arayüzü için özellik tabanlı testler (Property-Based Tests)

Bu testler, tasarım dokümanındaki doğruluk özelliklerini doğrular.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
import plotly.graph_objects as go

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import (
    plot_price_predictions,
    plot_sentiment_scores,
    plot_feature_importance,
    display_results
)


# Yardımcı stratejiler
@composite
def price_data_strategy(draw, min_size=10, max_size=100):
    """
    Geçerli bir fiyat verisi oluşturur.
    
    Returns:
        (actual_prices, predicted_prices, dates) tuple
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Pozitif fiyatlar (10.0 - 1000.0 arası)
    actual_prices = draw(st.lists(
        st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    
    predicted_prices = draw(st.lists(
        st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    
    # Tarih dizisi oluştur (opsiyonel)
    use_dates = draw(st.booleans())
    if use_dates:
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
        dates = dates.to_numpy()
    else:
        dates = None
    
    return np.array(actual_prices), np.array(predicted_prices), dates


@composite
def sentiment_dataframe_strategy(draw, min_size=10, max_size=100):
    """
    Geçerli bir duygu DataFrame'i oluşturur.
    
    Returns:
        DataFrame with columns: date, positive, negative, neutral, compound
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Tarih dizisi
    dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
    
    # Duygu skorları (0.0 - 1.0 arası)
    positive = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    
    negative = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    
    neutral = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    
    # Compound skorları (-1.0 - 1.0 arası)
    compound = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    
    df = pd.DataFrame({
        'date': dates,
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'compound': compound
    })
    
    return df


@composite
def feature_importance_dataframe_strategy(draw, min_size=5, max_size=30):
    """
    Geçerli bir özellik önem DataFrame'i oluşturur.
    
    Returns:
        DataFrame with columns: feature, importance
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Özellik isimleri
    feature_names = [f'feature_{i}' for i in range(size)]
    
    # Önem skorları (0.0 - 1.0 arası, pozitif)
    importance_scores = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    
    # Önem skoruna göre sırala (azalan)
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return df


@composite
def analysis_results_strategy(draw):
    """
    Geçerli bir analiz sonuçları dictionary'si oluşturur.
    
    Returns:
        Dictionary with analysis results
    """
    size = draw(st.integers(min_value=20, max_value=100))
    
    # Fiyat verileri
    actual_prices = draw(st.lists(
        st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    
    lstm_predictions = draw(st.lists(
        st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    
    xgb_predictions = draw(st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=size,
        max_size=size
    ))
    
    # Tarihler
    test_dates = pd.date_range(start='2023-01-01', periods=size, freq='D').to_numpy()
    
    # Duygu DataFrame'i
    sentiment_df = draw(sentiment_dataframe_strategy(min_size=size, max_size=size))
    
    # LSTM metrikleri
    lstm_metrics = {
        'rmse': draw(st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)),
        'mae': draw(st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)),
        'r2': draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    }
    
    # Geriye dönük test sonuçları
    backtest_results = {
        'total_return': draw(st.floats(min_value=-50.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        'num_trades': draw(st.integers(min_value=0, max_value=size))
    }
    
    # XGBoost doğruluğu
    xgb_accuracy = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    
    # Özellik önemleri
    feature_importance = draw(feature_importance_dataframe_strategy())
    
    results = {
        'actual_prices': np.array(actual_prices),
        'lstm_predictions': np.array(lstm_predictions),
        'xgb_predictions': np.array(xgb_predictions),
        'test_dates': test_dates,
        'sentiment_df': sentiment_df,
        'lstm_metrics': lstm_metrics,
        'backtest_results': backtest_results,
        'xgb_accuracy': xgb_accuracy,
        'feature_importance': feature_importance
    }
    
    return results


# Property 12: Streamlit Görselleştirme Varlığı
@given(data=price_data_strategy())
@settings(max_examples=100, deadline=5000)
def test_property_plot_price_predictions_exists(data):
    """
    **Validates: Requirements 6.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 12:
    Herhangi bir başarılı analiz çalıştırması için, fiyat grafiği fonksiyonu
    çağrıldığında, geçerli bir Plotly figure nesnesi döndürmelidir.
    """
    actual_prices, predicted_prices, dates = data
    
    # Fiyat grafiği oluştur
    fig = plot_price_predictions(actual_prices, predicted_prices, dates)
    
    # Figure'ın Plotly objesi olduğunu kontrol et
    assert isinstance(fig, go.Figure), \
        "plot_price_predictions geçerli bir Plotly Figure objesi döndürmeli"
    
    # Figure'ın veri içerdiğini kontrol et
    assert len(fig.data) > 0, \
        "Figure en az bir trace içermeli"
    
    # En az iki trace olmalı (gerçek ve tahmin)
    assert len(fig.data) >= 2, \
        "Figure en az iki trace içermeli (gerçek ve tahmin)"
    
    # Layout'un title içerdiğini kontrol et
    assert fig.layout.title is not None, \
        "Figure bir title içermeli"
    
    # X ve Y eksen başlıklarının olduğunu kontrol et
    assert fig.layout.xaxis.title is not None, \
        "Figure X ekseni başlığı içermeli"
    assert fig.layout.yaxis.title is not None, \
        "Figure Y ekseni başlığı içermeli"


@given(sentiment_df=sentiment_dataframe_strategy())
@settings(max_examples=100, deadline=5000)
def test_property_plot_sentiment_scores_exists(sentiment_df):
    """
    **Validates: Requirements 6.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 12:
    Herhangi bir başarılı analiz çalıştırması için, duygu grafiği fonksiyonu
    çağrıldığında, geçerli bir Plotly figure nesnesi döndürmelidir.
    """
    # Duygu grafiği oluştur
    fig = plot_sentiment_scores(sentiment_df)
    
    # Figure'ın Plotly objesi olduğunu kontrol et
    assert isinstance(fig, go.Figure), \
        "plot_sentiment_scores geçerli bir Plotly Figure objesi döndürmeli"
    
    # Figure'ın veri içerdiğini kontrol et
    assert len(fig.data) > 0, \
        "Figure en az bir trace içermeli"
    
    # En az üç trace olmalı (positive, negative, compound)
    assert len(fig.data) >= 3, \
        "Figure en az üç trace içermeli (positive, negative, compound)"
    
    # Layout'un title içerdiğini kontrol et
    assert fig.layout.title is not None, \
        "Figure bir title içermeli"
    
    # X ve Y eksen başlıklarının olduğunu kontrol et
    assert fig.layout.xaxis.title is not None, \
        "Figure X ekseni başlığı içermeli"
    assert fig.layout.yaxis.title is not None, \
        "Figure Y ekseni başlığı içermeli"


@given(importance_df=feature_importance_dataframe_strategy())
@settings(max_examples=100, deadline=5000)
def test_property_plot_feature_importance_exists(importance_df):
    """
    **Validates: Requirements 6.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 12:
    Herhangi bir başarılı analiz çalıştırması için, özellik önem grafiği fonksiyonu
    çağrıldığında, geçerli bir Plotly figure nesnesi döndürmelidir.
    """
    # Özellik önem grafiği oluştur
    fig = plot_feature_importance(importance_df)
    
    # Figure'ın Plotly objesi olduğunu kontrol et
    assert isinstance(fig, go.Figure), \
        "plot_feature_importance geçerli bir Plotly Figure objesi döndürmeli"
    
    # Figure'ın veri içerdiğini kontrol et
    assert len(fig.data) > 0, \
        "Figure en az bir trace içermeli"
    
    # Layout'un title içerdiğini kontrol et
    assert fig.layout.title is not None, \
        "Figure bir title içermeli"
    
    # X ve Y eksen başlıklarının olduğunu kontrol et
    assert fig.layout.xaxis.title is not None, \
        "Figure X ekseni başlığı içermeli"
    assert fig.layout.yaxis.title is not None, \
        "Figure Y ekseni başlığı içermeli"


@given(data=price_data_strategy())
@settings(max_examples=50, deadline=5000)
def test_property_plot_price_predictions_data_consistency(data):
    """
    **Validates: Requirements 6.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 12 (Extended):
    Fiyat grafiği, girdi verisi ile tutarlı sayıda veri noktası içermelidir.
    """
    actual_prices, predicted_prices, dates = data
    
    # Fiyat grafiği oluştur
    fig = plot_price_predictions(actual_prices, predicted_prices, dates)
    
    # Her trace'in veri noktası sayısını kontrol et
    for trace in fig.data:
        # Trace'in x ve y verisi olmalı
        assert hasattr(trace, 'x'), "Trace X verisi içermeli"
        assert hasattr(trace, 'y'), "Trace Y verisi içermeli"
        
        # Veri noktası sayısı girdi ile tutarlı olmalı
        assert len(trace.y) == len(actual_prices), \
            f"Trace veri noktası sayısı girdi ile tutarlı olmalı: " \
            f"beklenen={len(actual_prices)}, alınan={len(trace.y)}"


@given(sentiment_df=sentiment_dataframe_strategy())
@settings(max_examples=50, deadline=5000)
def test_property_plot_sentiment_scores_data_consistency(sentiment_df):
    """
    **Validates: Requirements 6.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 12 (Extended):
    Duygu grafiği, girdi DataFrame'i ile tutarlı sayıda veri noktası içermelidir.
    """
    # Duygu grafiği oluştur
    fig = plot_sentiment_scores(sentiment_df)
    
    # Günlük ortalama hesaplandığı için, benzersiz tarih sayısını al
    unique_dates = sentiment_df['date'].nunique()
    
    # Her trace'in veri noktası sayısını kontrol et
    for trace in fig.data:
        # Trace'in x ve y verisi olmalı
        assert hasattr(trace, 'x'), "Trace X verisi içermeli"
        assert hasattr(trace, 'y'), "Trace Y verisi içermeli"
        
        # Veri noktası sayısı benzersiz tarih sayısı ile tutarlı olmalı
        assert len(trace.y) == unique_dates, \
            f"Trace veri noktası sayısı benzersiz tarih sayısı ile tutarlı olmalı: " \
            f"beklenen={unique_dates}, alınan={len(trace.y)}"


@given(importance_df=feature_importance_dataframe_strategy(), top_n=st.integers(min_value=5, max_value=20))
@settings(max_examples=50, deadline=5000)
def test_property_plot_feature_importance_top_n(importance_df, top_n):
    """
    **Validates: Requirements 6.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 12 (Extended):
    Özellik önem grafiği, top_n parametresine göre doğru sayıda özellik göstermelidir.
    """
    # Özellik önem grafiği oluştur
    fig = plot_feature_importance(importance_df, top_n=top_n)
    
    # Beklenen özellik sayısı (DataFrame'deki özellik sayısı ile top_n'in minimumu)
    expected_features = min(top_n, len(importance_df))
    
    # Bar grafiğinin veri noktası sayısını kontrol et
    assert len(fig.data) > 0, "Figure en az bir trace içermeli"
    
    # İlk trace'in (bar grafiği) veri noktası sayısını kontrol et
    bar_trace = fig.data[0]
    assert len(bar_trace.y) == expected_features, \
        f"Bar grafiği {expected_features} özellik göstermeli, " \
        f"alınan: {len(bar_trace.y)}"


@given(data=price_data_strategy())
@settings(max_examples=50, deadline=5000)
def test_property_plot_price_predictions_no_nan_inf(data):
    """
    **Validates: Requirements 6.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 12 (Extended):
    Fiyat grafiği, NaN veya sonsuz değerler içermemelidir.
    """
    actual_prices, predicted_prices, dates = data
    
    # Fiyat grafiği oluştur
    fig = plot_price_predictions(actual_prices, predicted_prices, dates)
    
    # Her trace'in verilerini kontrol et
    for trace in fig.data:
        y_data = np.array(trace.y)
        
        # NaN kontrolü
        assert not np.any(np.isnan(y_data)), \
            "Grafik verisi NaN içermemeli"
        
        # Sonsuz değer kontrolü
        assert not np.any(np.isinf(y_data)), \
            "Grafik verisi sonsuz değer içermemeli"


@given(sentiment_df=sentiment_dataframe_strategy())
@settings(max_examples=50, deadline=5000)
def test_property_plot_sentiment_scores_no_nan_inf(sentiment_df):
    """
    **Validates: Requirements 6.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 12 (Extended):
    Duygu grafiği, NaN veya sonsuz değerler içermemelidir.
    """
    # Duygu grafiği oluştur
    fig = plot_sentiment_scores(sentiment_df)
    
    # Her trace'in verilerini kontrol et
    for trace in fig.data:
        y_data = np.array(trace.y)
        
        # NaN kontrolü
        assert not np.any(np.isnan(y_data)), \
            "Grafik verisi NaN içermemeli"
        
        # Sonsuz değer kontrolü
        assert not np.any(np.isinf(y_data)), \
            "Grafik verisi sonsuz değer içermemeli"


@given(importance_df=feature_importance_dataframe_strategy())
@settings(max_examples=50, deadline=5000)
def test_property_plot_feature_importance_no_nan_inf(importance_df):
    """
    **Validates: Requirements 6.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 12 (Extended):
    Özellik önem grafiği, NaN veya sonsuz değerler içermemelidir.
    """
    # Özellik önem grafiği oluştur
    fig = plot_feature_importance(importance_df)
    
    # Bar trace'in verilerini kontrol et
    bar_trace = fig.data[0]
    x_data = np.array(bar_trace.x)
    
    # NaN kontrolü
    assert not np.any(np.isnan(x_data)), \
        "Grafik verisi NaN içermemeli"
    
    # Sonsuz değer kontrolü
    assert not np.any(np.isinf(x_data)), \
        "Grafik verisi sonsuz değer içermemeli"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
