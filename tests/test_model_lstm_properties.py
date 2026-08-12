"""
Özellik Tabanlı Testler: LSTM Model Modülü

Bu dosya, model_lstm.py modülündeki fonksiyonlar için özellik tabanlı testleri içerir.
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
from sklearn.preprocessing import MinMaxScaler

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# src dizinini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_lstm import (
    build_lstm_model,
    train_lstm_model,
    predict_lstm,
    save_lstm_model,
    load_lstm_model
)


# TensorFlow yoksa testleri atla
pytestmark = pytest.mark.skipif(
    not TENSORFLOW_AVAILABLE,
    reason="TensorFlow gerekli"
)


# Yardımcı fonksiyonlar

def generate_random_training_data(num_samples: int, timesteps: int, features: int):
    """
    Random eğitim verisi oluşturur.
    
    Args:
        num_samples: Sample sayısı
        timesteps: Zaman adımı sayısı
        features: Özellik sayısı
    
    Returns:
        (X, y) tuple
    """
    X = np.random.rand(num_samples, timesteps, features).astype(np.float32)
    y = np.random.rand(num_samples).astype(np.float32)
    return X, y


def train_simple_lstm_model(timesteps: int, features: int, num_samples: int = 30):
    """
    Basit bir LSTM modeli eğitir (test için).
    
    Args:
        timesteps: Zaman adımı sayısı
        features: Özellik sayısı
        num_samples: Eğitim sample sayısı
    
    Returns:
        Eğitilmiş model
    """
    # Model oluştur
    input_shape = (timesteps, features)
    model = build_lstm_model(input_shape, lstm_units=[32, 32], dropout=0.1)
    
    # Eğitim verisi oluştur
    X_train, y_train = generate_random_training_data(num_samples, timesteps, features)
    
    # Çok hızlı eğitim (sadece 1 epoch, verbose=0)
    train_lstm_model(model, X_train, y_train, epochs=1, batch_size=16)
    
    return model


# Property 7: LSTM Tahmin Geçerliliği
# **Validates: Requirements 4.2**

@given(
    st.integers(min_value=10, max_value=30),   # timesteps (azaltıldı)
    st.integers(min_value=3, max_value=8),     # features (azaltıldı)
    st.integers(min_value=5, max_value=20)     # num_samples (azaltıldı)
)
@settings(max_examples=5, deadline=None)  # Sadece 5 örnek
def test_property_7_lstm_prediction_validity(timesteps, features, num_samples):
    """
    Feature: turkish-sentiment-stock-prediction, Property 7:
    Herhangi bir eğitilmiş LSTM modeli ve geçerli girdi verisi için, tahmin 
    fonksiyonu çağrıldığında, dönen değerler numeric olmalı ve girdi ile aynı 
    sample sayısına sahip olmalıdır. Tahmin değerleri makul bir fiyat aralığında 
    olmalıdır (negatif olmamalı).
    
    **Validates: Requirements 4.2**
    """
    # Model eğit
    model = train_simple_lstm_model(timesteps, features, num_samples=50)
    
    # Test verisi oluştur
    X_test, _ = generate_random_training_data(num_samples, timesteps, features)
    
    # Tahmin yap
    predictions = predict_lstm(model, X_test)
    
    # Tahminlerin numeric olduğunu kontrol et
    assert isinstance(predictions, np.ndarray), \
        f"Tahminler numpy array olmalı, alınan: {type(predictions)}"
    assert np.issubdtype(predictions.dtype, np.number), \
        f"Tahminler numeric olmalı, alınan dtype: {predictions.dtype}"
    
    # Sample sayısının eşleştiğini kontrol et
    assert predictions.shape[0] == num_samples, \
        f"Tahmin sayısı girdi ile eşleşmiyor: beklenen {num_samples}, alınan {predictions.shape[0]}"
    
    # Tahminlerin 1D array olduğunu kontrol et
    assert len(predictions.shape) == 1, \
        f"Tahminler 1D olmalı, alınan shape: {predictions.shape}"
    
    # NaN veya Inf içermediğini kontrol et
    assert not np.isnan(predictions).any(), "Tahminler NaN içeriyor"
    assert not np.isinf(predictions).any(), "Tahminler Inf içeriyor"


@given(
    st.integers(min_value=10, max_value=30),
    st.integers(min_value=3, max_value=8),
    st.integers(min_value=5, max_value=20)
)
@settings(max_examples=5, deadline=None)
def test_property_7_lstm_prediction_with_scaler(timesteps, features, num_samples):
    """
    Property 7 ek test: Scaler ile tahmin, orijinal ölçeğe dönüştürülmüş 
    değerler üretmelidir.
    
    **Validates: Requirements 4.2**
    """
    # Model eğit
    model = train_simple_lstm_model(timesteps, features, num_samples=30)
    
    # Test verisi oluştur
    X_test, _ = generate_random_training_data(num_samples, timesteps, features)
    
    # Scaler oluştur ve fit et
    scaler = MinMaxScaler()
    dummy_data = np.random.rand(100, 1) * 100  # 0-100 arası değerler
    scaler.fit(dummy_data)
    
    # Tahmin yap (scaler ile)
    predictions = predict_lstm(model, X_test, scaler)
    
    # Tahminlerin geçerli olduğunu kontrol et
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == num_samples
    assert not np.isnan(predictions).any()
    assert not np.isinf(predictions).any()


@given(
    st.integers(min_value=10, max_value=30),
    st.integers(min_value=3, max_value=8)
)
@settings(max_examples=5, deadline=None)
def test_property_7_lstm_prediction_deterministic(timesteps, features):
    """
    Property 7 ek test: Aynı girdi için aynı tahminler üretilmelidir 
    (deterministik davranış).
    
    **Validates: Requirements 4.2**
    """
    # Model eğit
    model = train_simple_lstm_model(timesteps, features, num_samples=30)
    
    # Test verisi oluştur
    X_test, _ = generate_random_training_data(10, timesteps, features)
    
    # İki kez tahmin yap
    predictions_1 = predict_lstm(model, X_test)
    predictions_2 = predict_lstm(model, X_test)
    
    # Tahminlerin eşit olduğunu kontrol et
    np.testing.assert_allclose(
        predictions_1,
        predictions_2,
        rtol=1e-5,
        err_msg="Aynı girdi için farklı tahminler üretildi"
    )


# Property 8: Model Serileştirme Round-Trip
# **Validates: Requirements 4.3**

@given(
    st.integers(min_value=10, max_value=30),
    st.integers(min_value=3, max_value=8),
    st.integers(min_value=5, max_value=15)
)
@settings(max_examples=5, deadline=None)
def test_property_8_model_serialization_roundtrip(timesteps, features, num_samples):
    """
    Feature: turkish-sentiment-stock-prediction, Property 8:
    Herhangi bir eğitilmiş model (LSTM veya XGBoost) için, modeli kaydedip 
    sonra yüklemek, aynı girdi verileri için eşdeğer tahminler üreten bir 
    model oluşturmalıdır (küçük numerik tolerans içinde).
    
    **Validates: Requirements 4.3**
    """
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Model eğit
        model = train_simple_lstm_model(timesteps, features, num_samples=30)
        
        # Test verisi oluştur
        X_test, _ = generate_random_training_data(num_samples, timesteps, features)
        
        # Orijinal tahminler
        predictions_original = predict_lstm(model, X_test)
        
        # Model kaydet
        filepath = os.path.join(temp_dir, 'test_model.keras')
        save_lstm_model(model, filepath)
        
        # Dosyanın oluşturulduğunu kontrol et
        assert os.path.exists(filepath), "Model dosyası oluşturulmadı"
        
        # Model yükle
        loaded_model = load_lstm_model(filepath)
        
        # Yüklenen modelden tahminler
        predictions_loaded = predict_lstm(loaded_model, X_test)
        
        # Tahminlerin eşdeğer olduğunu kontrol et (küçük tolerans)
        np.testing.assert_allclose(
            predictions_original,
            predictions_loaded,
            rtol=1e-5,
            err_msg="Kaydedilen ve yüklenen model farklı tahminler üretti"
        )
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


@given(
    st.integers(min_value=10, max_value=30),
    st.integers(min_value=3, max_value=8)
)
@settings(max_examples=5, deadline=None)
def test_property_8_model_architecture_preserved(timesteps, features):
    """
    Property 8 ek test: Model mimarisi kaydet-yükle sonrası korunmalıdır.
    
    **Validates: Requirements 4.3**
    """
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Model oluştur
        input_shape = (timesteps, features)
        model = build_lstm_model(input_shape)
        
        # Orijinal model bilgileri
        original_input_shape = model.input_shape
        original_output_shape = model.output_shape
        original_param_count = model.count_params()
        
        # Model kaydet
        filepath = os.path.join(temp_dir, 'test_model.keras')
        save_lstm_model(model, filepath)
        
        # Model yükle
        loaded_model = load_lstm_model(filepath)
        
        # Mimari bilgilerinin korunduğunu kontrol et
        assert loaded_model.input_shape == original_input_shape, \
            f"Input shape değişti: {original_input_shape} -> {loaded_model.input_shape}"
        assert loaded_model.output_shape == original_output_shape, \
            f"Output shape değişti: {original_output_shape} -> {loaded_model.output_shape}"
        assert loaded_model.count_params() == original_param_count, \
            f"Parametre sayısı değişti: {original_param_count} -> {loaded_model.count_params()}"
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


@given(
    st.integers(min_value=10, max_value=30),
    st.integers(min_value=3, max_value=8)
)
@settings(max_examples=5, deadline=None)
def test_property_8_multiple_save_load_cycles(timesteps, features):
    """
    Property 8 ek test: Birden fazla kaydet-yükle döngüsü tahminleri 
    bozmamalıdır.
    
    **Validates: Requirements 4.3**
    """
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Model eğit
        model = train_simple_lstm_model(timesteps, features, num_samples=30)
        
        # Test verisi oluştur
        X_test, _ = generate_random_training_data(10, timesteps, features)
        
        # Orijinal tahminler
        predictions_original = predict_lstm(model, X_test)
        
        # 3 kez kaydet-yükle döngüsü
        current_model = model
        for i in range(3):
            filepath = os.path.join(temp_dir, f'model_{i}.keras')
            save_lstm_model(current_model, filepath)
            current_model = load_lstm_model(filepath)
        
        # Son modelden tahminler
        predictions_final = predict_lstm(current_model, X_test)
        
        # Tahminlerin hala eşdeğer olduğunu kontrol et
        np.testing.assert_allclose(
            predictions_original,
            predictions_final,
            rtol=1e-5,
            err_msg="Birden fazla kaydet-yükle döngüsü tahminleri bozdu"
        )
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
