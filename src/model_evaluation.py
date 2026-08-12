"""
Model Değerlendirme Modülü

Bu modül, LSTM model performansını izleme, değerlendirme ve görselleştirme
fonksiyonlarını içerir.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import tensorflow as tf
except ImportError:
    tf = None


class ModelTracker:
    """LSTM model eğitim ve performans izleme sınıfı."""
    
    def __init__(self, model_name: str, save_dir: str = 'models/training_history/'):
        """
        Args:
            model_name: Model adı (dosya isimlendirme için)
            save_dir: Eğitim geçmişi kayıt dizini
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.history = None
        self.metrics = {}
        
        # Dizini oluştur
        os.makedirs(save_dir, exist_ok=True)
    
    def get_callbacks(self) -> List:
        """
        Model eğitimi için callback'leri döndürür.
        
        Returns:
            List of Keras callbacks:
                - ModelCheckpoint: En iyi modeli kaydeder
                - EarlyStopping: Overfitting'i önler
                - CSVLogger: Eğitim metriklerini CSV'ye kaydeder
        """
        if tf is None:
            raise ImportError("TensorFlow is required for callbacks")
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.save_dir}{self.model_name}_best.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.save_dir}{self.model_name}_training_log.csv',
                separator=',',
                append=False
            )
        ]
        return callbacks
    
    def save_training_history(self, history) -> None:
        """
        Eğitim geçmişini kaydeder.
        
        Args:
            history: Keras training history object
        """
        self.history = history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(
            f'{self.save_dir}{self.model_name}_history.csv',
            index=False
        )


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Kapsamlı performans metrikleri hesaplar.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri
    
    Returns:
        {
            'mse': Mean Squared Error,
            'rmse': Root Mean Squared Error,
            'mae': Mean Absolute Error,
            'mape': Mean Absolute Percentage Error,
            'r2': R-squared Score,
            'directional_accuracy': Yönsel doğruluk (%)
        }
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE hesaplama (sıfır değerleri atla)
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    r2 = r2_score(y_true, y_pred)
    
    # Yönsel doğruluk
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = 0.0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'directional_accuracy': float(directional_accuracy)
    }


def calculate_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Hata dağılımı istatistiklerini hesaplar.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri
    
    Returns:
        {
            'errors': Hata değerleri array,
            'mean_error': Ortalama hata,
            'std_error': Hata standart sapması,
            'percentiles': Hata yüzdelikleri (25, 50, 75)
        }
    """
    errors = y_pred - y_true
    
    return {
        'errors': errors,
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'percentiles': {
            '25': float(np.percentile(errors, 25)),
            '50': float(np.percentile(errors, 50)),
            '75': float(np.percentile(errors, 75))
        }
    }


def plot_training_history(history_df: pd.DataFrame) -> plt.Figure:
    """
    Eğitim geçmişini görselleştirir.
    
    Args:
        history_df: Eğitim geçmişi DataFrame
    
    Returns:
        Matplotlib figure (loss ve metrik grafikleri)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss grafiği
    axes[0].plot(history_df['loss'], label='Training Loss')
    axes[0].plot(history_df['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Metrik grafiği (MAE varsa)
    if 'mae' in history_df.columns:
        axes[1].plot(history_df['mae'], label='Training MAE')
        axes[1].plot(history_df['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Model MAE')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None
) -> plt.Figure:
    """
    Tahmin vs gerçek değer grafiği oluşturur.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri
        dates: Tarih indeksi (opsiyonel)
    
    Returns:
        Matplotlib figure (zaman serisi ve scatter plot)
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Zaman serisi grafiği
    x_axis = dates if dates is not None else range(len(y_true))
    axes[0].plot(x_axis, y_true, label='Gerçek', alpha=0.7)
    axes[0].plot(x_axis, y_pred, label='Tahmin', alpha=0.7)
    axes[0].set_xlabel('Zaman')
    axes[0].set_ylabel('Fiyat')
    axes[0].set_title('Tahmin vs Gerçek Değerler (Zaman Serisi)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5)
    axes[1].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Mükemmel Tahmin')
    axes[1].set_xlabel('Gerçek Değerler')
    axes[1].set_ylabel('Tahmin Değerleri')
    axes[1].set_title('Tahmin vs Gerçek Değerler (Scatter Plot)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig


def plot_error_distribution(errors: np.ndarray) -> plt.Figure:
    """
    Hata dağılımı histogramı oluşturur.
    
    Args:
        errors: Hata değerleri array
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Sıfır Hata')
    ax.axvline(x=np.mean(errors), color='g', linestyle='--', 
               linewidth=2, label=f'Ortalama Hata: {np.mean(errors):.2f}')
    ax.set_xlabel('Hata (Tahmin - Gerçek)')
    ax.set_ylabel('Frekans')
    ax.set_title('Hata Dağılımı')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> plt.Figure:
    """
    Farklı model versiyonlarının metriklerini karşılaştırır.
    
    Args:
        metrics_dict: {model_name: {metric_name: value}}
    
    Returns:
        Matplotlib figure (bar chart)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metric_names = ['rmse', 'mae', 'mape', 'r2', 'directional_accuracy']
    model_names = list(metrics_dict.keys())
    
    for idx, metric in enumerate(metric_names):
        if idx < len(axes):
            values = [metrics_dict[model].get(metric, 0) for model in model_names]
            axes[idx].bar(model_names, values)
            axes[idx].set_title(metric.upper())
            axes[idx].set_ylabel('Değer')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
    
    # Son subplot'u gizle
    if len(metric_names) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    return fig


def save_model_report(
    model_name: str,
    metrics: Dict[str, float],
    save_path: str
) -> None:
    """
    Model performans raporunu JSON formatında kaydeder.
    
    Args:
        model_name: Model adı
        metrics: Performans metrikleri
        save_path: Kayıt yolu
    """
    report = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def load_model_reports(reports_dir: str) -> Dict[str, Dict]:
    """
    Tüm model raporlarını yükler.
    
    Args:
        reports_dir: Raporların bulunduğu dizin
    
    Returns:
        {model_name: report_dict}
    """
    reports = {}
    
    if not os.path.exists(reports_dir):
        return reports
    
    for filename in os.listdir(reports_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(reports_dir, filename), 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    reports[report['model_name']] = report
            except (json.JSONDecodeError, KeyError):
                continue
    
    return reports


def select_best_model(
    metrics_dict: Dict[str, Dict[str, float]],
    primary_metric: str = 'rmse',
    minimize: bool = True
) -> str:
    """
    En iyi modeli seçer.
    
    Args:
        metrics_dict: {model_name: {metric_name: value}}
        primary_metric: Karşılaştırma için kullanılacak metrik
        minimize: True ise düşük değer iyidir, False ise yüksek değer
    
    Returns:
        En iyi model adı
    """
    if minimize:
        best_model = min(metrics_dict.items(), 
                        key=lambda x: x[1].get(primary_metric, float('inf')))
    else:
        best_model = max(metrics_dict.items(), 
                        key=lambda x: x[1].get(primary_metric, float('-inf')))
    
    return best_model[0]
