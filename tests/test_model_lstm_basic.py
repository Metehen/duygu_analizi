"""
Birim Testleri: LSTM Model Modülü
LSTM model oluşturma, eğitim, tahmin ve kaydetme/yükleme fonksiyonları için testler.
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

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


class TestBuildLSTMModel:
    """build_lstm_model fonksiyonu için testler."""
    
    def test_build_model_default_params(self):
        """Varsayılan parametrelerle model oluşturma."""
        input_shape = (60, 10)  # 60 timesteps, 10 features
        model = build_lstm_model(input_shape)
        
        assert model is not None
        assert isinstance(model, keras.Model)
        assert model.input_shape == (None, 60, 10)
        assert model.output_shape == (None, 1)
    
    def test_build_model_custom_params(self):
        """Özel parametrelerle model oluşturma."""
        input_shape = (30, 5)
        lstm_units = [64, 32]
        dropout = 0.3
        
        model = build_lstm_model(input_shape, lstm_units, dropout)
        
        assert model is not None
        assert model.input_shape == (None, 30, 5)
        assert model.output_shape == (None, 1)
    
    def test_build_model_compiled(self):
        """Model'in compile edildiğini kontrol et."""
        input_shape = (60, 10)
        model = build_lstm_model(input_shape)
        
        # Optimizer ve loss fonksiyonu ayarlanmış olmalı
        assert model.optimizer is not None
        assert model.loss is not None
    
    def test_build_model_parameter_count(self):
        """Model parametrelerinin pozitif olduğunu kontrol et."""
        input_shape = (60, 10)
        model = build_lstm_model(input_shape)
        
        param_count = model.count_params()
        assert param_count > 0


class TestTrainLSTMModel:
    """train_lstm_model fonksiyonu için testler."""
    
    def test_train_model_basic(self):
        """Basit eğitim testi."""
        # Küçük model ve veri oluştur
        input_shape = (10, 5)
        model = build_lstm_model(input_shape)
        
        # Dummy veri
        X_train = np.random.rand(50, 10, 5).astype(np.float32)
        y_train = np.random.rand(50).astype(np.float32)
        
        # Eğit (az epoch ile)
        history = train_lstm_model(
            model, X_train, y_train,
            epochs=2,
            batch_size=16
        )
        
        assert history is not None
        assert 'loss' in history.history
        assert len(history.history['loss']) > 0
    
    def test_train_model_with_validation(self):
        """Validasyon verisi ile eğitim."""
        input_shape = (10, 5)
        model = build_lstm_model(input_shape)
        
        X_train = np.random.rand(50, 10, 5).astype(np.float32)
        y_train = np.random.rand(50).astype(np.float32)
        X_val = np.random.rand(20, 10, 5).astype(np.float32)
        y_val = np.random.rand(20).astype(np.float32)
        
        history = train_lstm_model(
            model, X_train, y_train, X_val, y_val,
            epochs=2,
            batch_size=16
        )
        
        assert 'val_loss' in history.history
    
    def test_train_model_invalid_shape_X(self):
        """Geçersiz X_train boyutu."""
        input_shape = (10, 5)
        model = build_lstm_model(input_shape)
        
        # 2D array (hatalı)
        X_train = np.random.rand(50, 10).astype(np.float32)
        y_train = np.random.rand(50).astype(np.float32)
        
        with pytest.raises(ValueError, match="X_train boyutu"):
            train_lstm_model(model, X_train, y_train, epochs=1)
    
    def test_train_model_invalid_shape_y(self):
        """Geçersiz y_train boyutu."""
        input_shape = (10, 5)
        model = build_lstm_model(input_shape)
        
        X_train = np.random.rand(50, 10, 5).astype(np.float32)
        # 2D array (hatalı)
        y_train = np.random.rand(50, 1).astype(np.float32)
        
        with pytest.raises(ValueError, match="y_train boyutu"):
            train_lstm_model(model, X_train, y_train, epochs=1)
    
    def test_train_model_mismatched_samples(self):
        """X ve y sample sayıları uyumsuz."""
        input_shape = (10, 5)
        model = build_lstm_model(input_shape)
        
        X_train = np.random.rand(50, 10, 5).astype(np.float32)
        y_train = np.random.rand(40).astype(np.float32)  # Farklı sample sayısı
        
        with pytest.raises(ValueError, match="sample sayıları eşleşmiyor"):
            train_lstm_model(model, X_train, y_train, epochs=1)


class TestPredictLSTM:
    """predict_lstm fonksiyonu için testler."""
    
    def test_predict_basic(self):
        """Basit tahmin testi."""
        input_shape = (10, 5)
        model = build_lstm_model(input_shape)
        
        # Dummy veri ile eğit
        X_train = np.random.rand(50, 10, 5).astype(np.float32)
        y_train = np.random.rand(50).astype(np.float32)
        train_lstm_model(model, X_train, y_train, epochs=1, batch_size=16)
        
        # Tahmin yap
        X_test = np.random.rand(10, 10, 5).astype(np.float32)
        predictions = predict_lstm(model, X_test)
        
        assert predictions is not None
        assert predictions.shape == (10,)
        assert predictions.dtype == np.float32 or predictions.dtype == np.float64
    
    def test_predict_with_scaler(self):
        """Scaler ile tahmin."""
        from sklearn.preprocessing import MinMaxScaler
        
        input_shape = (10, 5)
        model = build_lstm_model(input_shape)
        
        # Eğit
        X_train = np.random.rand(50, 10, 5).astype(np.float32)
        y_train = np.random.rand(50).astype(np.float32)
        train_lstm_model(model, X_train, y_train, epochs=1, batch_size=16)
        
        # Scaler oluştur
        scaler = MinMaxScaler()
        scaler.fit(np.random.rand(100, 1))
        
        # Tahmin yap
        X_test = np.random.rand(10, 10, 5).astype(np.float32)
        predictions = predict_lstm(model, X_test, scaler)
        
        assert predictions is not None
        assert predictions.shape == (10,)
    
    def test_predict_invalid_shape(self):
        """Geçersiz girdi boyutu."""
        input_shape = (10, 5)
        model = build_lstm_model(input_shape)
        
        # 2D array (hatalı)
        X_test = np.random.rand(10, 5).astype(np.float32)
        
        with pytest.raises(ValueError, match="X boyutu"):
            predict_lstm(model, X_test)
    
    def test_predict_output_shape(self):
        """Tahmin çıktı boyutunu kontrol et."""
        input_shape = (10, 5)
        model = build_lstm_model(input_shape)
        
        # Eğit
        X_train = np.random.rand(50, 10, 5).astype(np.float32)
        y_train = np.random.rand(50).astype(np.float32)
        train_lstm_model(model, X_train, y_train, epochs=1, batch_size=16)
        
        # Farklı sample sayıları ile tahmin
        for n_samples in [1, 5, 20]:
            X_test = np.random.rand(n_samples, 10, 5).astype(np.float32)
            predictions = predict_lstm(model, X_test)
            assert predictions.shape == (n_samples,)


class TestSaveLoadLSTMModel:
    """save_lstm_model ve load_lstm_model fonksiyonları için testler."""
    
    def test_save_and_load_model(self):
        """Model kaydetme ve yükleme."""
        # Geçici dizin oluştur
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Model oluştur
            input_shape = (10, 5)
            model = build_lstm_model(input_shape)
            
            # Kaydet
            filepath = os.path.join(temp_dir, 'test_model.keras')
            save_lstm_model(model, filepath)
            
            # Dosya oluşturuldu mu?
            assert os.path.exists(filepath)
            
            # Yükle
            loaded_model = load_lstm_model(filepath)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, keras.Model)
            assert loaded_model.input_shape == model.input_shape
            assert loaded_model.output_shape == model.output_shape
            
        finally:
            # Temizle
            shutil.rmtree(temp_dir)
    
    def test_save_and_load_predictions_match(self):
        """Kaydedilen ve yüklenen model aynı tahminleri üretmeli."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Model oluştur ve eğit
            input_shape = (10, 5)
            model = build_lstm_model(input_shape)
            
            X_train = np.random.rand(50, 10, 5).astype(np.float32)
            y_train = np.random.rand(50).astype(np.float32)
            train_lstm_model(model, X_train, y_train, epochs=2, batch_size=16)
            
            # Test verisi
            X_test = np.random.rand(10, 10, 5).astype(np.float32)
            
            # Orijinal tahminler
            predictions_original = predict_lstm(model, X_test)
            
            # Kaydet ve yükle
            filepath = os.path.join(temp_dir, 'test_model.keras')
            save_lstm_model(model, filepath)
            loaded_model = load_lstm_model(filepath)
            
            # Yüklenen modelden tahminler
            predictions_loaded = predict_lstm(loaded_model, X_test)
            
            # Tahminler eşit olmalı (küçük tolerans)
            np.testing.assert_allclose(
                predictions_original,
                predictions_loaded,
                rtol=1e-5
            )
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_nonexistent_model(self):
        """Var olmayan model dosyası."""
        with pytest.raises(FileNotFoundError):
            load_lstm_model('nonexistent_model.keras')
    
    def test_save_creates_directory(self):
        """Kaydetme dizini yoksa oluşturulmalı."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            input_shape = (10, 5)
            model = build_lstm_model(input_shape)
            
            # Var olmayan alt dizin
            filepath = os.path.join(temp_dir, 'subdir', 'model.keras')
            save_lstm_model(model, filepath)
            
            assert os.path.exists(filepath)
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
