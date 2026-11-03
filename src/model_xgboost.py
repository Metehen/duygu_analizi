# Copyright (C) 2026  Metehen
# Licensed under GNU AGPL v3. See LICENSE file for details.
"""
XGBoost modeli modülü.
Hisse senedi fiyat yönü tahmini (yukarı/aşağı) için XGBoost sınıflandırıcı.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from src.logger import get_logger

logger = get_logger(__name__)


def prepare_classification_data(df: pd.DataFrame, target_column: str = 'close') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sınıflandırma için veri hazırlar.
    
    Yön etiketleri oluşturur:
    - 1: Fiyat artışı (Close[t+1] > Close[t])
    - 0: Fiyat düşüşü veya sabit (Close[t+1] <= Close[t])
    
    Args:
        df: Özellikler içeren DataFrame (teknik göstergeler, duygu skorları vb.)
        target_column: Hedef fiyat sütunu (varsayılan: 'close')
    
    Returns:
        (X, y) tuple:
            - X: Özellikler DataFrame (son satır hariç)
            - y: Binary etiketler Series (1: yukarı, 0: aşağı)
    
    Raises:
        ValueError: DataFrame boş veya target_column bulunamadığında
        ValueError: Yetersiz veri (< 2 satır) olduğunda
    """
    if df.empty:
        raise ValueError("DataFrame boş olamaz")
    
    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun '{target_column}' DataFrame'de bulunamadı")
    
    if len(df) < 2:
        raise ValueError("En az 2 satır veri gerekli")
    
    # Hedef sütunu küçük harfe çevir (case-insensitive)
    target_col = None
    for col in df.columns:
        if col.lower() == target_column.lower():
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"Hedef sütun '{target_column}' DataFrame'de bulunamadı")
    
    # Yön etiketlerini oluştur
    # 1: Fiyat artışı, 0: Fiyat düşüşü veya sabit
    df_copy = df.copy()
    df_copy['direction'] = (df_copy[target_col].shift(-1) > df_copy[target_col]).astype(int)
    
    # Son satırı çıkar (gelecek fiyat bilgisi yok)
    df_copy = df_copy.iloc[:-1]
    
    # Özellikleri ve etiketleri ayır
    y = df_copy['direction']
    X = df_copy.drop(columns=['direction'])
    
    # Hedef sütunu da özelliklerden çıkar (data leakage önleme)
    if target_col in X.columns:
        X = X.drop(columns=[target_col])

    # Feature leakage önleme: EMA'lar doğrudan close'tan hesaplandığı için
    # XGBoost close'u dolaylı olarak görür — bu sütunları çıkarıyoruz.
    leaky_cols = [c for c in X.columns if c.startswith('ema_')]
    if leaky_cols:
        X = X.drop(columns=leaky_cols)
        logger.info(f"Leakage önlemi: {leaky_cols} sütunları çıkarıldı")
    
    logger.info(f"Sınıflandırma verisi hazırlandı: {len(X)} örnek, {X.shape[1]} özellik")
    logger.info(f"Sınıf dağılımı - Yukarı: {y.sum()}, Aşağı: {len(y) - y.sum()}")
    
    return X, y



def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict] = None
) -> xgb.XGBClassifier:
    """
    XGBoost modelini eğitir.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim etiketleri (binary: 1=yukarı, 0=aşağı)
        X_val: Validasyon özellikleri (opsiyonel)
        y_val: Validasyon etiketleri (opsiyonel)
        params: Model parametreleri (opsiyonel, varsayılan config.py'den alınır)
    
    Returns:
        Eğitilmiş XGBoost modeli
    
    Raises:
        ValueError: Eğitim verileri boş veya uyumsuz boyutlarda ise
    """
    if X_train.empty or len(y_train) == 0:
        raise ValueError("Eğitim verileri boş olamaz")
    
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train ({len(X_train)}) ve y_train ({len(y_train)}) boyutları uyumsuz")
    
    # Varsayılan parametreler
    default_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    # Kullanıcı parametreleri ile birleştir
    if params is not None:
        default_params.update(params)
    
    logger.info(f"XGBoost modeli eğitiliyor: {len(X_train)} eğitim örneği, {X_train.shape[1]} özellik")
    logger.info(f"Parametreler: {default_params}")
    
    # Model oluştur
    model = xgb.XGBClassifier(**default_params)
    
    # Validasyon verisi varsa eval_set kullan
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        logger.info(f"Validasyon seti kullanılıyor: {len(X_val)} örnek")
    
    # Modeli eğit
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        logger.info("Model eğitimi tamamlandı")
    except Exception as e:
        logger.error(f"Model eğitimi sırasında hata: {e}")
        raise
    
    return model



def get_feature_importance(model: xgb.XGBClassifier, feature_names: Optional[list] = None) -> pd.DataFrame:
    """
    Özellik önemlerini döndürür.
    
    Args:
        model: Eğitilmiş XGBoost modeli
        feature_names: Özellik isimleri listesi (opsiyonel)
    
    Returns:
        DataFrame with columns: 'feature' ve 'importance'
        Önem değerine göre azalan sırada sıralanmış
    
    Raises:
        ValueError: Model eğitilmemişse
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model henüz eğitilmemiş")
    
    # Özellik önemlerini al
    importances = model.feature_importances_
    
    # Özellik isimlerini belirle
    if feature_names is None:
        # Model'den özellik isimlerini almaya çalış
        if hasattr(model, 'get_booster'):
            try:
                feature_names = model.get_booster().feature_names
            except:
                pass
        
        # Hala yoksa varsayılan isimler kullan
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    # DataFrame oluştur
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Önem değerine göre sırala (azalan)
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    logger.info(f"Özellik önemleri hesaplandı: {len(importance_df)} özellik")
    logger.info(f"En önemli 5 özellik: {importance_df.head()['feature'].tolist()}")
    
    return importance_df



def predict_direction(model: xgb.XGBClassifier, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Yön tahmini yapar.
    
    Args:
        model: Eğitilmiş XGBoost modeli
        X: Girdi özellikleri DataFrame
    
    Returns:
        (predictions, probabilities) tuple:
            - predictions: Binary tahminler (1: yukarı, 0: aşağı)
            - probabilities: Olasılık değerleri (yukarı yönü için)
    
    Raises:
        ValueError: Model eğitilmemişse veya girdi boşsa
    """
    if X.empty:
        raise ValueError("Girdi verileri boş olamaz")
    
    if not hasattr(model, 'predict'):
        raise ValueError("Model henüz eğitilmemiş")
    
    try:
        # Binary tahminler (0 veya 1)
        predictions = model.predict(X)
        
        # Olasılık değerleri (her sınıf için)
        probabilities_all = model.predict_proba(X)
        
        # Sadece pozitif sınıf (1: yukarı) olasılığını al
        probabilities = probabilities_all[:, 1]
        
        logger.info(f"Tahmin yapıldı: {len(predictions)} örnek")
        logger.info(f"Yukarı tahmin: {predictions.sum()}, Aşağı tahmin: {len(predictions) - predictions.sum()}")
        
        return predictions, probabilities
    
    except Exception as e:
        logger.error(f"Tahmin sırasında hata: {e}")
        raise



def save_model(model: xgb.XGBClassifier, filepath: str) -> None:
    """
    XGBoost modelini dosyaya kaydeder.
    
    Args:
        model: Eğitilmiş XGBoost modeli
        filepath: Kayıt yolu (örn: 'models/xgboost_model.pkl')
    
    Raises:
        ValueError: Model eğitilmemişse
        IOError: Dosya yazma hatası
    """
    if not hasattr(model, 'predict'):
        raise ValueError("Model henüz eğitilmemiş")
    
    # Dizini oluştur (yoksa)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model kaydedildi: {filepath}")
    except Exception as e:
        logger.error(f"Model kaydetme hatası: {e}")
        raise IOError(f"Model kaydedilemedi: {e}")


def load_model(filepath: str) -> xgb.XGBClassifier:
    """
    Kaydedilmiş XGBoost modelini yükler.
    
    Args:
        filepath: Model dosya yolu
    
    Returns:
        Yüklenmiş XGBoost modeli
    
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        IOError: Dosya okuma veya model yükleme hatası
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model yüklendi: {filepath}")
        return model
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        raise IOError(f"Model yüklenemedi: {e}")
