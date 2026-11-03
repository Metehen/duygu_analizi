"""
LSTM Model Modülü
Zaman serisi tahmini için LSTM mimarisi, eğitim ve tahmin fonksiyonları.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional, Any
import os

# Config'den parametreleri al
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_CONFIG, MODELS_DIR
from src.logger import get_logger

# Logging
logger = get_logger(__name__)


def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: list = None,
    dropout: float = None
) -> keras.Model:
    """
    LSTM modelini oluşturur.
    
    Args:
        input_shape: (timesteps, features) - Girdi verisi boyutu
        lstm_units: Her LSTM katmanındaki birim sayısı listesi (varsayılan: [50, 50])
        dropout: Dropout oranı (varsayılan: 0.2)
    
    Returns:
        Compiled Keras model
    
    Mimari:
        Input Layer (timesteps, features)
            ↓
        LSTM Layer (50 units, return_sequences=True)
            ↓
        Dropout (0.2)
            ↓
        LSTM Layer (50 units)
            ↓
        Dropout (0.2)
            ↓
        Dense Layer (25 units, ReLU)
            ↓
        Dense Layer (1 unit, Linear)
    """
    # Varsayılan parametreleri kullan
    if lstm_units is None:
        lstm_units = LSTM_CONFIG['lstm_units']
    if dropout is None:
        dropout = LSTM_CONFIG['dropout']
    
    logger.info(f"LSTM modeli oluşturuluyor: input_shape={input_shape}, lstm_units={lstm_units}, dropout={dropout}")
    
    # Model mimarisi
    model = keras.Sequential([
        # İlk LSTM katmanı (return_sequences=True çünkü ikinci LSTM katmanı var)
        layers.LSTM(
            lstm_units[0],
            return_sequences=True,
            input_shape=input_shape,
            name='lstm_1'
        ),
        layers.Dropout(dropout, name='dropout_1'),
        
        # İkinci LSTM katmanı
        layers.LSTM(
            lstm_units[1],
            return_sequences=False,
            name='lstm_2'
        ),
        layers.Dropout(dropout, name='dropout_2'),
        
        # Dense katmanlar
        layers.Dense(25, activation='relu', name='dense_1'),
        layers.Dense(1, activation='linear', name='output')
    ])
    
    # Modeli compile et
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    logger.info("LSTM modeli başarıyla oluşturuldu ve compile edildi")
    logger.info(f"Model parametreleri: {model.count_params()}")
    
    return model



def train_lstm_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = None,
    batch_size: int = None,
    early_stopping_patience: int = None,
    model_save_path: Optional[str] = None
) -> keras.callbacks.History:
    """
    LSTM modelini eğitir.
    
    Args:
        model: Keras modeli (build_lstm_model ile oluşturulmuş)
        X_train: Eğitim girdi verileri, shape: (samples, timesteps, features)
        y_train: Eğitim hedef verileri, shape: (samples,)
        X_val: Validasyon girdi verileri (opsiyonel)
        y_val: Validasyon hedef verileri (opsiyonel)
        epochs: Epoch sayısı (varsayılan: config'den)
        batch_size: Batch boyutu (varsayılan: config'den)
        early_stopping_patience: Early stopping patience (varsayılan: config'den)
        model_save_path: Model checkpoint kayıt yolu (opsiyonel)
    
    Returns:
        Training history (keras.callbacks.History)
    
    Raises:
        ValueError: Veri boyutları uyumsuzsa
    """
    # Varsayılan parametreleri kullan
    if epochs is None:
        epochs = LSTM_CONFIG['epochs']
    if batch_size is None:
        batch_size = LSTM_CONFIG['batch_size']
    if early_stopping_patience is None:
        early_stopping_patience = LSTM_CONFIG['early_stopping_patience']
    
    # Veri boyutlarını kontrol et
    if len(X_train.shape) != 3:
        raise ValueError(f"X_train boyutu (samples, timesteps, features) olmalı, alınan: {X_train.shape}")
    if len(y_train.shape) != 1:
        raise ValueError(f"y_train boyutu (samples,) olmalı, alınan: {y_train.shape}")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train ve y_train sample sayıları eşleşmiyor: {X_train.shape[0]} vs {y_train.shape[0]}")
    
    logger.info(f"LSTM modeli eğitimi başlıyor...")
    logger.info(f"Eğitim verisi: X_train={X_train.shape}, y_train={y_train.shape}")
    if X_val is not None and y_val is not None:
        logger.info(f"Validasyon verisi: X_val={X_val.shape}, y_val={y_val.shape}")
    logger.info(f"Parametreler: epochs={epochs}, batch_size={batch_size}, patience={early_stopping_patience}")
    
    # Callbacks oluştur
    callbacks = []
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss' if X_val is not None else 'loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint callback (eğer kayıt yolu verilmişse)
    if model_save_path:
        # Dizini oluştur
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        logger.info(f"Model checkpoint: {model_save_path}")
    
    # Validasyon verisi hazırla
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    # Modeli eğit
    try:
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("LSTM modeli eğitimi tamamlandı")
        logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
        if validation_data:
            logger.info(f"Final val_loss: {history.history['val_loss'][-1]:.4f}")
        
        return history
        
    except Exception as e:
        logger.error(f"Model eğitimi sırasında hata: {str(e)}")
        raise



def predict_lstm_with_uncertainty(
    model: keras.Model,
    X: np.ndarray,
    scaler: Optional[Any] = None,
    n_passes: int = 50,
    confidence: float = 0.90
) -> dict:
    """
    Monte Carlo Dropout ile tahmin yapar ve belirsizlik aralığı hesaplar.

    Nasıl çalışır:
        Dropout katmanları normalde yalnızca eğitim sırasında aktiftir.
        MC Dropout'ta model tahmin modunda bile Dropout'u açık tutarak
        aynı girdi için N kez farklı tahmin üretir. Bu dağılımın
        ortalaması tahmini, standart sapması ise belirsizliği verir.

    Args:
        model:      Eğitilmiş LSTM modeli (Dropout katmanları içermeli)
        X:          Girdi verileri, shape (samples, timesteps, features)
        scaler:     Normalizasyon scalerı — verilmişse tahminler orijinal
                    ölçeğe dönüştürülür (predict_lstm ile aynı davranış)
        n_passes:   Monte Carlo forward pass sayısı. Daha yüksek = daha
                    kararlı belirsizlik tahmini, daha yavaş. Önerilen: 50
        confidence: Güven aralığı genişliği (0.0-1.0). 0.90 → %90 CI.

    Returns:
        {
            'mean':       np.ndarray — ortalama tahmin (orijinal ölçekte)
            'std':        np.ndarray — standart sapma (belirsizlik)
            'lower':      np.ndarray — güven aralığı alt sınırı
            'upper':      np.ndarray — güven aralığı üst sınırı
            'confidence': float      — kullanılan güven düzeyi
            'n_passes':   int        — kullanılan pass sayısı
            'cv':         np.ndarray — varyasyon katsayısı (std/mean, %)
                                       yüksek CV → tahmin güvenilmez
        }

    Örnek:
        result = predict_lstm_with_uncertainty(model, X_test, scaler, n_passes=50)
        mean   = result['mean']    # Ana tahmin
        lower  = result['lower']   # Alt sınır
        upper  = result['upper']   # Üst sınır

        # Düşük güven uyarısı
        high_uncertainty = result['cv'] > 5.0  # %5 üzeri CV
    """
    if len(X.shape) != 3:
        raise ValueError(
            f"X boyutu (samples, timesteps, features) olmalı, alınan: {X.shape}"
        )

    logger.info(
        "MC Dropout belirsizlik analizi başlıyor: n_passes=%d, confidence=%.0f%%",
        n_passes, confidence * 100
    )

    # ── N kez forward pass (training=True → Dropout aktif) ────────
    all_preds = []
    for i in range(n_passes):
        # training=True ile Dropout eğitim modunda çalışır
        preds = model(X, training=True)
        all_preds.append(preds.numpy().flatten())

    # shape: (n_passes, samples)
    all_preds = np.array(all_preds)

    # ── İstatistikler (normalize edilmiş uzayda) ──────────────────
    mean_norm = all_preds.mean(axis=0)
    std_norm  = all_preds.std(axis=0)

    # Güven aralığı z-skoru: 0.90 → 1.645, 0.95 → 1.960, 0.99 → 2.576
    from scipy.stats import norm as _norm
    z = float(_norm.ppf(0.5 + confidence / 2))

    lower_norm = mean_norm - z * std_norm
    upper_norm = mean_norm + z * std_norm

    # ── Scaler ile orijinal ölçeğe dön ───────────────────────────
    def _inverse(arr: np.ndarray) -> np.ndarray:
        if scaler is not None:
            return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()
        return arr

    mean_real  = _inverse(mean_norm)
    lower_real = _inverse(lower_norm)
    upper_real = _inverse(upper_norm)
    std_real   = _inverse(std_norm) - _inverse(np.zeros_like(std_norm))

    # Alt sınır negatif olmasın (fiyat)
    lower_real = np.maximum(lower_real, 0.0)

    # ── Varyasyon katsayısı (%) — güven göstergesi ────────────────
    # Küçük mean'lere karşı sıfıra bölme koruması
    cv = np.where(
        np.abs(mean_real) > 1e-6,
        (np.abs(std_real) / np.abs(mean_real)) * 100,
        0.0
    )

    logger.info(
        "MC Dropout tamamlandı: mean=%.2f±%.2f, CI=[%.2f, %.2f], CV=%.2f%%",
        float(mean_real.mean()),
        float(std_real.mean()),
        float(lower_real.mean()),
        float(upper_real.mean()),
        float(cv.mean())
    )

    return {
        'mean':       mean_real,
        'std':        std_real,
        'lower':      lower_real,
        'upper':      upper_real,
        'confidence': confidence,
        'n_passes':   n_passes,
        'cv':         cv,
    }


def predict_lstm(
    model: keras.Model,
    X: np.ndarray,
    scaler: Optional[Any] = None
) -> np.ndarray:
    """
    LSTM ile tahmin yapar.
    
    Args:
        model: Eğitilmiş LSTM modeli
        X: Girdi verileri, shape: (samples, timesteps, features)
        scaler: Normalizasyon için scaler (opsiyonel, MinMaxScaler veya StandardScaler)
                Eğer verilmişse, tahminler inverse_transform ile geri dönüştürülür
    
    Returns:
        Tahmin edilen değerler, shape: (samples,)
        Eğer scaler verilmişse, orijinal ölçeğe dönüştürülmüş değerler
    
    Raises:
        ValueError: Girdi boyutları uyumsuzsa
    """
    # Veri boyutunu kontrol et
    if len(X.shape) != 3:
        raise ValueError(f"X boyutu (samples, timesteps, features) olmalı, alınan: {X.shape}")
    
    logger.info(f"LSTM tahmini yapılıyor: X={X.shape}")
    
    try:
        # Tahmin yap
        predictions = model.predict(X, verbose=0)
        
        # Tahminler (samples, 1) şeklinde gelir, (samples,) şekline dönüştür
        predictions = predictions.flatten()
        
        # Negatif değerleri kontrol et ve uyar
        if np.any(predictions < 0):
            logger.warning(f"Negatif tahmin değerleri bulundu: {np.sum(predictions < 0)} adet")
            logger.warning("Fiyat tahminleri negatif olmamalı, model performansını kontrol edin")
        
        # Scaler ile geri dönüştür (eğer verilmişse)
        if scaler is not None:
            logger.info("Tahminler orijinal ölçeğe dönüştürülüyor")
            # Scaler genellikle 2D array bekler, reshape et
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions_reshaped).flatten()
        
        logger.info(f"Tahmin tamamlandı: {len(predictions)} adet tahmin")
        logger.info(f"Tahmin istatistikleri: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Tahmin sırasında hata: {str(e)}")
        raise



def save_lstm_model(
    model: keras.Model,
    filepath: str
) -> None:
    """
    LSTM modelini .keras formatında kaydeder.
    
    Args:
        model: Kaydedilecek Keras modeli
        filepath: Kayıt yolu (örn: 'models/lstm_model.keras')
    
    Raises:
        Exception: Model kaydedilemezse
    """
    try:
        # Dizini oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Modeli kaydet (.keras formatı)
        model.save(filepath)
        
        logger.info(f"Model başarıyla kaydedildi: {filepath}")
        logger.info(f"Dosya boyutu: {os.path.getsize(filepath) / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Model kaydedilemedi: {str(e)}")
        raise


def load_lstm_model(
    filepath: str
) -> keras.Model:
    """
    Kaydedilmiş LSTM modelini yükler.
    
    Args:
        filepath: Model dosyası yolu (örn: 'models/lstm_model.keras')
    
    Returns:
        Yüklenmiş Keras modeli
    
    Raises:
        FileNotFoundError: Model dosyası bulunamazsa
        Exception: Model yüklenemezse
    """
    # Dosya varlığını kontrol et
    if not os.path.exists(filepath):
        logger.error(f"Model dosyası bulunamadı: {filepath}")
        raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
    
    try:
        # Modeli yükle
        model = keras.models.load_model(filepath)
        
        logger.info(f"Model başarıyla yüklendi: {filepath}")
        logger.info(f"Model parametreleri: {model.count_params()}")
        
        return model
        
    except Exception as e:
        logger.error(f"Model yüklenemedi: {str(e)}")
        raise


# Yardımcı fonksiyon: Model özeti
def print_model_summary(model: keras.Model) -> None:
    """
    Model mimarisini ve parametrelerini yazdırır.
    
    Args:
        model: Keras modeli
    """
    print("\n" + "="*60)
    print("LSTM MODEL ÖZETİ")
    print("="*60)
    model.summary()
    print("="*60 + "\n")
