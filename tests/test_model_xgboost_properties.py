"""
Özellik Tabanlı Testler: XGBoost Model Modülü

Bu dosya, model_xgboost.py modülündeki fonksiyonlar için özellik tabanlı testleri içerir.
Hypothesis kütüphanesi kullanılarak evrensel doğruluk özellikleri test edilir.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, assume
import xgboost as xgb

# src dizinini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_xgboost import (
    prepare_classification_data,
    train_xgboost_model,
    get_feature_importance,
    predict_direction,
    save_model,
    load_model
)


# Yardımcı fonksiyonlar

def generate_random_price_dataframe(num_rows: int, num_features: int):
    """
    Random fiyat ve özellik DataFrame'i oluşturur.
    
    Args:
        num_rows: Satır sayısı
        num_features: Özellik sayısı
    
    Returns:
        DataFrame with 'close' column and random features
    """
    data = {
        'close': np.random.uniform(50, 200, num_rows)
    }
    
    # Ek özellikler ekle
    for i in range(num_features):
        data[f'feature_{i}'] = np.random.uniform(0, 100, num_rows)
    
    return pd.DataFrame(data)


def train_simple_xgboost_model(num_samples: int = 50, num_features: int = 5):
    """
    Basit bir XGBoost modeli eğitir (test için).
    
    Args:
        num_samples: Sample sayısı
        num_features: Özellik sayısı
    
    Returns:
        (model, X_train, y_train) tuple
    """
    # Veri oluştur
    df = generate_random_price_dataframe(num_samples + 1, num_features)
    X, y = prepare_classification_data(df)
    
    # Model eğit (hızlı parametreler)
    params = {
        'max_depth': 3,
        'n_estimators': 10,
        'learning_rate': 0.1,
        'random_state': 42
    }
    model = train_xgboost_model(X, y, params=params)
    
    return model, X, y


# Property 9: XGBoost Yön Tahmini Geçerliliği
# **Validates: Requirements 4.5**

@given(
    st.integers(min_value=10, max_value=50),   # num_samples
    st.integers(min_value=3, max_value=10)     # num_features
)
@settings(max_examples=20, deadline=None)
def test_property_9_xgboost_direction_prediction_validity(num_samples, num_features):
    """
    Feature: turkish-sentiment-stock-prediction, Property 9:
    Herhangi bir eğitilmiş XGBoost modeli ve geçerli girdi verisi için, tahmin 
    fonksiyonu çağrıldığında, dönen tahminler binary değerler (0 veya 1) olmalı 
    ve olasılık değerleri 0.0-1.0 aralığında olmalıdır.
    
    **Validates: Requirements 4.5**
    """
    # Model eğit
    model, X_train, y_train = train_simple_xgboost_model(num_samples, num_features)
    
    # Test verisi oluştur (eğitim verisinin bir kısmını kullan)
    test_size = min(10, len(X_train))
    X_test = X_train.iloc[:test_size]
    
    # Tahmin yap
    predictions, probabilities = predict_direction(model, X_test)
    
    # Tahminlerin binary olduğunu kontrol et (0 veya 1)
    assert isinstance(predictions, np.ndarray), \
        f"Tahminler numpy array olmalı, alınan: {type(predictions)}"
    assert set(predictions).issubset({0, 1}), \
        f"Tahminler sadece 0 veya 1 olmalı, alınan: {set(predictions)}"
    
    # Olasılıkların geçerli olduğunu kontrol et
    assert isinstance(probabilities, np.ndarray), \
        f"Olasılıklar numpy array olmalı, alınan: {type(probabilities)}"
    assert probabilities.shape[0] == test_size, \
        f"Olasılık sayısı girdi ile eşleşmiyor: beklenen {test_size}, alınan {probabilities.shape[0]}"
    
    # Olasılıkların 0.0-1.0 aralığında olduğunu kontrol et
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0)), \
        f"Olasılıklar 0.0-1.0 aralığında olmalı, min: {probabilities.min()}, max: {probabilities.max()}"
    
    # NaN veya Inf içermediğini kontrol et
    assert not np.isnan(predictions).any(), "Tahminler NaN içeriyor"
    assert not np.isnan(probabilities).any(), "Olasılıklar NaN içeriyor"
    assert not np.isinf(probabilities).any(), "Olasılıklar Inf içeriyor"


@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=3, max_value=10)
)
@settings(max_examples=20, deadline=None)
def test_property_9_xgboost_prediction_count_matches_input(num_samples, num_features):
    """
    Property 9 ek test: Tahmin sayısı girdi sample sayısı ile eşleşmelidir.
    
    **Validates: Requirements 4.5**
    """
    # Model eğit
    model, X_train, y_train = train_simple_xgboost_model(num_samples, num_features)
    
    # Test verisi oluştur
    test_size = min(15, len(X_train))
    X_test = X_train.iloc[:test_size]
    
    # Tahmin yap
    predictions, probabilities = predict_direction(model, X_test)
    
    # Sayıların eşleştiğini kontrol et
    assert len(predictions) == test_size, \
        f"Tahmin sayısı girdi ile eşleşmiyor: beklenen {test_size}, alınan {len(predictions)}"
    assert len(probabilities) == test_size, \
        f"Olasılık sayısı girdi ile eşleşmiyor: beklenen {test_size}, alınan {len(probabilities)}"


@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=3, max_value=10)
)
@settings(max_examples=20, deadline=None)
def test_property_9_xgboost_prediction_deterministic(num_samples, num_features):
    """
    Property 9 ek test: Aynı girdi için aynı tahminler üretilmelidir 
    (deterministik davranış).
    
    **Validates: Requirements 4.5**
    """
    # Model eğit
    model, X_train, y_train = train_simple_xgboost_model(num_samples, num_features)
    
    # Test verisi oluştur
    test_size = min(10, len(X_train))
    X_test = X_train.iloc[:test_size]
    
    # İki kez tahmin yap
    predictions_1, probabilities_1 = predict_direction(model, X_test)
    predictions_2, probabilities_2 = predict_direction(model, X_test)
    
    # Tahminlerin eşit olduğunu kontrol et
    np.testing.assert_array_equal(
        predictions_1,
        predictions_2,
        err_msg="Aynı girdi için farklı tahminler üretildi"
    )
    
    np.testing.assert_allclose(
        probabilities_1,
        probabilities_2,
        rtol=1e-5,
        err_msg="Aynı girdi için farklı olasılıklar üretildi"
    )


# Property 10: Özellik Önem Toplamı
# **Validates: Requirements 4.6**

@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=3, max_value=10)
)
@settings(max_examples=20, deadline=None)
def test_property_10_feature_importance_sum(num_samples, num_features):
    """
    Feature: turkish-sentiment-stock-prediction, Property 10:
    Herhangi bir eğitilmiş XGBoost modeli için, özellik önem fonksiyonu 
    çağrıldığında, dönen DataFrame tüm özellikleri içermeli ve önem değerleri 
    negatif olmamalıdır. Tüm önem değerlerinin toplamı pozitif bir sayı olmalıdır.
    
    **Validates: Requirements 4.6**
    """
    # Model eğit
    model, X_train, y_train = train_simple_xgboost_model(num_samples, num_features)
    
    # Özellik önemlerini al
    importance_df = get_feature_importance(model, feature_names=X_train.columns.tolist())
    
    # DataFrame yapısını kontrol et
    assert isinstance(importance_df, pd.DataFrame), \
        f"Sonuç DataFrame olmalı, alınan: {type(importance_df)}"
    assert 'feature' in importance_df.columns, \
        "DataFrame 'feature' sütunu içermeli"
    assert 'importance' in importance_df.columns, \
        "DataFrame 'importance' sütunu içermeli"
    
    # Tüm özelliklerin dahil olduğunu kontrol et
    assert len(importance_df) == num_features, \
        f"Tüm özellikler dahil olmalı: beklenen {num_features}, alınan {len(importance_df)}"
    
    # Önem değerlerinin negatif olmadığını kontrol et
    assert (importance_df['importance'] >= 0).all(), \
        f"Önem değerleri negatif olmamalı, min: {importance_df['importance'].min()}"
    
    # Toplamın pozitif olduğunu kontrol et
    total_importance = importance_df['importance'].sum()
    assert total_importance > 0, \
        f"Önem değerlerinin toplamı pozitif olmalı, alınan: {total_importance}"
    
    # NaN veya Inf içermediğini kontrol et
    assert not importance_df['importance'].isna().any(), \
        "Önem değerleri NaN içeriyor"
    assert not np.isinf(importance_df['importance']).any(), \
        "Önem değerleri Inf içeriyor"


@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=3, max_value=10)
)
@settings(max_examples=20, deadline=None)
def test_property_10_feature_importance_sorted(num_samples, num_features):
    """
    Property 10 ek test: Özellik önemleri azalan sırada sıralanmış olmalıdır.
    
    **Validates: Requirements 4.6**
    """
    # Model eğit
    model, X_train, y_train = train_simple_xgboost_model(num_samples, num_features)
    
    # Özellik önemlerini al
    importance_df = get_feature_importance(model, feature_names=X_train.columns.tolist())
    
    # Sıralamanın doğru olduğunu kontrol et (azalan)
    importances = importance_df['importance'].values
    assert np.all(importances[:-1] >= importances[1:]), \
        "Özellik önemleri azalan sırada sıralanmış olmalı"


@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=3, max_value=10)
)
@settings(max_examples=20, deadline=None)
def test_property_10_feature_importance_consistency(num_samples, num_features):
    """
    Property 10 ek test: Aynı model için özellik önemleri tutarlı olmalıdır.
    
    **Validates: Requirements 4.6**
    """
    # Model eğit
    model, X_train, y_train = train_simple_xgboost_model(num_samples, num_features)
    
    # İki kez özellik önemlerini al
    importance_df_1 = get_feature_importance(model, feature_names=X_train.columns.tolist())
    importance_df_2 = get_feature_importance(model, feature_names=X_train.columns.tolist())
    
    # Değerlerin eşit olduğunu kontrol et
    pd.testing.assert_frame_equal(
        importance_df_1,
        importance_df_2,
        check_exact=False,
        rtol=1e-5
    )


# Property 8: Model Serileştirme Round-Trip (XGBoost için)
# **Validates: Requirements 4.7**

@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=3, max_value=10)
)
@settings(max_examples=20, deadline=None)
def test_property_8_xgboost_model_serialization_roundtrip(num_samples, num_features):
    """
    Feature: turkish-sentiment-stock-prediction, Property 8:
    Herhangi bir eğitilmiş model (LSTM veya XGBoost) için, modeli kaydedip 
    sonra yüklemek, aynı girdi verileri için eşdeğer tahminler üreten bir 
    model oluşturmalıdır (küçük numerik tolerans içinde).
    
    **Validates: Requirements 4.7**
    """
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Model eğit
        model, X_train, y_train = train_simple_xgboost_model(num_samples, num_features)
        
        # Test verisi oluştur
        test_size = min(10, len(X_train))
        X_test = X_train.iloc[:test_size]
        
        # Orijinal tahminler
        predictions_original, probabilities_original = predict_direction(model, X_test)
        
        # Model kaydet
        filepath = os.path.join(temp_dir, 'test_xgboost_model.pkl')
        save_model(model, filepath)
        
        # Dosyanın oluşturulduğunu kontrol et
        assert os.path.exists(filepath), "Model dosyası oluşturulmadı"
        
        # Model yükle
        loaded_model = load_model(filepath)
        
        # Yüklenen modelden tahminler
        predictions_loaded, probabilities_loaded = predict_direction(loaded_model, X_test)
        
        # Tahminlerin eşdeğer olduğunu kontrol et
        np.testing.assert_array_equal(
            predictions_original,
            predictions_loaded,
            err_msg="Kaydedilen ve yüklenen model farklı tahminler üretti"
        )
        
        np.testing.assert_allclose(
            probabilities_original,
            probabilities_loaded,
            rtol=1e-5,
            err_msg="Kaydedilen ve yüklenen model farklı olasılıklar üretti"
        )
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=3, max_value=10)
)
@settings(max_examples=20, deadline=None)
def test_property_8_xgboost_feature_importance_preserved(num_samples, num_features):
    """
    Property 8 ek test: Özellik önemleri kaydet-yükle sonrası korunmalıdır.
    
    **Validates: Requirements 4.7**
    """
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Model eğit
        model, X_train, y_train = train_simple_xgboost_model(num_samples, num_features)
        
        # Orijinal özellik önemleri
        importance_original = get_feature_importance(model, feature_names=X_train.columns.tolist())
        
        # Model kaydet
        filepath = os.path.join(temp_dir, 'test_xgboost_model.pkl')
        save_model(model, filepath)
        
        # Model yükle
        loaded_model = load_model(filepath)
        
        # Yüklenen modelden özellik önemleri
        importance_loaded = get_feature_importance(loaded_model, feature_names=X_train.columns.tolist())
        
        # Özellik önemlerinin korunduğunu kontrol et
        pd.testing.assert_frame_equal(
            importance_original,
            importance_loaded,
            check_exact=False,
            rtol=1e-5
        )
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=3, max_value=10)
)
@settings(max_examples=20, deadline=None)
def test_property_8_xgboost_multiple_save_load_cycles(num_samples, num_features):
    """
    Property 8 ek test: Birden fazla kaydet-yükle döngüsü tahminleri 
    bozmamalıdır.
    
    **Validates: Requirements 4.7**
    """
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Model eğit
        model, X_train, y_train = train_simple_xgboost_model(num_samples, num_features)
        
        # Test verisi oluştur
        test_size = min(10, len(X_train))
        X_test = X_train.iloc[:test_size]
        
        # Orijinal tahminler
        predictions_original, probabilities_original = predict_direction(model, X_test)
        
        # 3 kez kaydet-yükle döngüsü
        current_model = model
        for i in range(3):
            filepath = os.path.join(temp_dir, f'xgboost_model_{i}.pkl')
            save_model(current_model, filepath)
            current_model = load_model(filepath)
        
        # Son modelden tahminler
        predictions_final, probabilities_final = predict_direction(current_model, X_test)
        
        # Tahminlerin hala eşdeğer olduğunu kontrol et
        np.testing.assert_array_equal(
            predictions_original,
            predictions_final,
            err_msg="Birden fazla kaydet-yükle döngüsü tahminleri bozdu"
        )
        
        np.testing.assert_allclose(
            probabilities_original,
            probabilities_final,
            rtol=1e-5,
            err_msg="Birden fazla kaydet-yükle döngüsü olasılıkları bozdu"
        )
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
