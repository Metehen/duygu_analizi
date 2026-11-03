# Copyright (C) 2026  Metehen
# Licensed under GNU AGPL v3. See LICENSE file for details.
"""
Geriye Dönük Test Modülü

Bu modül, tahmin modellerinin performansını değerlendirmek için
basit geriye dönük test fonksiyonları sağlar.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def simple_backtest(
    predictions: np.ndarray,
    actual_prices: np.ndarray,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001
) -> Dict[str, float]:
    """
    Basit geriye dönük test yapar.
    
    Strateji:
        - Tahmin yukarı ise (fiyat artacak): Al
        - Tahmin aşağı ise (fiyat düşecek): Sat veya bekle
    
    Args:
        predictions: Model tahminleri (fiyat tahminleri veya yön tahminleri)
        actual_prices: Gerçek fiyatlar
        initial_capital: Başlangıç sermayesi
        transaction_cost: İşlem başına maliyet oranı (varsayılan: %0.1)
    
    Returns:
        Dictionary containing:
            - total_return: Toplam getiri (yüzde)
            - accuracy: Doğruluk oranı (yüzde)
            - num_trades: İşlem sayısı
            - final_capital: Son sermaye
            - total_cost: Toplam işlem maliyeti
    """
    if len(predictions) == 0 or len(actual_prices) == 0:
        return {
            'total_return': 0.0,
            'accuracy': 0.0,
            'num_trades': 0,
            'final_capital': initial_capital,
            'total_cost': 0.0
        }
    
    if len(predictions) != len(actual_prices) - 1:
        raise ValueError(
            f"Predictions length ({len(predictions)}) must be one less than "
            f"actual_prices length ({len(actual_prices)})"
        )
    
    capital = initial_capital
    position = 0  # 0: pozisyon yok, 1: hisse sahibi
    num_trades = 0
    correct_predictions = 0
    total_predictions = 0
    total_cost = 0.0
    shares = 0.0
    
    for i in range(len(predictions)):
        current_price = actual_prices[i]
        next_price = actual_prices[i + 1]
        predicted_price = predictions[i]
        
        # Yön tahmini
        predicted_direction = 1 if predicted_price > current_price else 0
        actual_direction = 1 if next_price > current_price else 0
        
        # Doğruluk hesaplama
        if predicted_direction == actual_direction:
            correct_predictions += 1
        total_predictions += 1
        
        # Al-sat stratejisi — işlem maliyeti dahil
        if predicted_direction == 1 and position == 0:
            # Al: sermayenin %0.1'i komisyon
            cost = capital * transaction_cost
            total_cost += cost
            capital -= cost
            shares = capital / current_price
            position = 1
            num_trades += 1
        elif predicted_direction == 0 and position == 1:
            # Sat: satış tutarının %0.1'i komisyon
            gross = shares * current_price
            cost = gross * transaction_cost
            total_cost += cost
            capital = gross - cost
            position = 0
            shares = 0.0
            num_trades += 1
    
    # Hâlâ pozisyon varsa son fiyattan sat (komisyon dahil)
    if position == 1:
        gross = shares * actual_prices[-1]
        cost = gross * transaction_cost
        total_cost += cost
        capital = gross - cost
    
    total_return = ((capital - initial_capital) / initial_capital) * 100
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
    
    return {
        'total_return': total_return,
        'accuracy': accuracy,
        'num_trades': num_trades,
        'final_capital': capital,
        'total_cost': total_cost
    }



def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Tahmin metrikleri hesaplar.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri
    
    Returns:
        Dictionary containing:
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2: R-squared score
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'r2': 0.0
        }
    
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length ({len(y_true)}) must equal "
            f"y_pred length ({len(y_pred)})"
        )
    
    # MSE hesapla
    mse = mean_squared_error(y_true, y_pred)
    
    # RMSE hesapla
    rmse = np.sqrt(mse)
    
    # MAE hesapla
    mae = mean_absolute_error(y_true, y_pred)
    
    # R2 hesapla
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
