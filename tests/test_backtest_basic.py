"""
Geriye dönük test modülü için birim testler (Unit Tests)

Bu testler, belirli örnekleri ve kenar durumları test eder.
"""

import pytest
import numpy as np
import os
import sys

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest import simple_backtest, calculate_metrics


class TestSimpleBacktest:
    """simple_backtest fonksiyonu için birim testler."""
    
    def test_simple_scenario(self):
        """
        Basit senaryo testi.
        
        Basit bir fiyat serisi ve tahminlerle geriye dönük test yapar.
        Sonuçların beklenen yapıda olduğunu doğrular.
        """
        # Basit bir fiyat serisi: yavaşça artan
        actual_prices = np.array([100.0, 105.0, 110.0, 108.0, 112.0, 115.0])
        
        # Tahminler: çoğunlukla doğru yön tahminleri
        predictions = np.array([106.0, 111.0, 107.0, 113.0, 116.0])
        
        initial_capital = 10000.0
        
        # Geriye dönük test çalıştır
        result = simple_backtest(predictions, actual_prices, initial_capital)
        
        # Sonuç yapısını kontrol et
        assert 'total_return' in result
        assert 'accuracy' in result
        assert 'num_trades' in result
        assert 'final_capital' in result
        
        # Değer tiplerini kontrol et
        assert isinstance(result['total_return'], (int, float))
        assert isinstance(result['accuracy'], (int, float))
        assert isinstance(result['num_trades'], int)
        assert isinstance(result['final_capital'], (int, float))
        
        # Değer aralıklarını kontrol et
        assert 0.0 <= result['accuracy'] <= 100.0
        assert result['num_trades'] >= 0
        assert result['final_capital'] > 0
    
    def test_all_predictions_correct(self):
        """
        Tüm tahminler doğru testi.
        
        Tüm tahminlerin doğru olduğu bir senaryo.
        Doğruluk oranının %100 olması beklenir.
        """
        # Sürekli artan fiyatlar
        actual_prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
        
        # Tüm tahminler doğru: her zaman yukarı tahmin et
        # Her tahmin bir sonraki günün fiyatından yüksek
        predictions = np.array([111.0, 121.0, 131.0, 141.0, 151.0])
        
        initial_capital = 10000.0
        
        # Geriye dönük test çalıştır
        result = simple_backtest(predictions, actual_prices, initial_capital)
        
        # Doğruluk oranı %100 olmalı
        assert result['accuracy'] == 100.0, \
            f"Tüm tahminler doğru olduğunda accuracy %100 olmalı, alınan: {result['accuracy']}"
        
        # En az bir işlem yapılmış olmalı
        assert result['num_trades'] > 0, \
            "Tüm tahminler doğru olduğunda en az bir işlem yapılmalı"
        
        # Sürekli artan piyasada kar edilmeli
        assert result['total_return'] > 0, \
            "Sürekli artan piyasada ve doğru tahminlerle kar edilmeli"
        
        # Final sermaye başlangıçtan fazla olmalı
        assert result['final_capital'] > initial_capital, \
            "Kar edildiğinde final sermaye başlangıçtan fazla olmalı"
    
    def test_all_predictions_wrong(self):
        """
        Tüm tahminler yanlış testi.
        
        Tüm tahminlerin yanlış olduğu bir senaryo.
        Doğruluk oranının %0 olması beklenir.
        """
        # Sürekli artan fiyatlar
        actual_prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
        
        # Tüm tahminler yanlış: her zaman aşağı tahmin et
        # Her tahmin mevcut fiyattan düşük (ama bir sonraki fiyat yükseliyor)
        predictions = np.array([99.0, 109.0, 119.0, 129.0, 139.0])
        
        initial_capital = 10000.0
        
        # Geriye dönük test çalıştır
        result = simple_backtest(predictions, actual_prices, initial_capital)
        
        # Doğruluk oranı %0 olmalı
        assert result['accuracy'] == 0.0, \
            f"Tüm tahminler yanlış olduğunda accuracy %0 olmalı, alınan: {result['accuracy']}"
        
        # Yanlış tahminlerle zarar edilmeli veya hiç işlem yapılmamalı
        # (strateji gereği aşağı tahmin edildiğinde pozisyon açılmaz)
        assert result['final_capital'] <= initial_capital, \
            "Tüm tahminler yanlış olduğunda kar edilmemeli"
    
    def test_empty_arrays(self):
        """
        Boş array'ler testi.
        
        Boş tahmin ve fiyat array'leri ile çağrıldığında
        varsayılan değerler döndürülmeli.
        """
        predictions = np.array([])
        actual_prices = np.array([])
        initial_capital = 10000.0
        
        result = simple_backtest(predictions, actual_prices, initial_capital)
        
        # Varsayılan değerleri kontrol et
        assert result['total_return'] == 0.0
        assert result['accuracy'] == 0.0
        assert result['num_trades'] == 0
        assert result['final_capital'] == initial_capital
    
    def test_mismatched_array_lengths(self):
        """
        Uyumsuz array uzunlukları testi.
        
        Tahmin ve fiyat array'lerinin uzunlukları uyumsuz olduğunda
        ValueError fırlatılmalı.
        """
        # Yanlış uzunluklar: predictions, actual_prices'dan 1 eksik olmalı
        actual_prices = np.array([100.0, 110.0, 120.0])
        predictions = np.array([105.0, 115.0, 125.0])  # 3 tahmin, olması gereken 2
        
        with pytest.raises(ValueError, match="must be one less than"):
            simple_backtest(predictions, actual_prices, 10000.0)
    
    def test_single_prediction(self):
        """
        Tek tahmin testi.
        
        Sadece bir tahmin ve iki fiyat ile çalışabilmeli.
        """
        actual_prices = np.array([100.0, 110.0])
        predictions = np.array([105.0])
        initial_capital = 10000.0
        
        result = simple_backtest(predictions, actual_prices, initial_capital)
        
        # Sonuç yapısı geçerli olmalı
        assert 'total_return' in result
        assert 'accuracy' in result
        assert 'num_trades' in result
        assert 'final_capital' in result
        
        # Doğruluk %0 veya %100 olmalı (tek tahmin)
        assert result['accuracy'] in [0.0, 100.0]
    
    def test_constant_prices(self):
        """
        Sabit fiyatlar testi.
        
        Fiyatlar hiç değişmediğinde, işlem yapılmamalı
        veya çok az işlem yapılmalı.
        """
        # Tüm fiyatlar aynı
        actual_prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        predictions = np.array([100.0, 100.0, 100.0, 100.0])
        initial_capital = 10000.0
        
        result = simple_backtest(predictions, actual_prices, initial_capital)
        
        # Sabit fiyatlarda kar/zarar olmamalı (veya çok az)
        assert abs(result['total_return']) < 1.0, \
            "Sabit fiyatlarda önemli kar/zarar olmamalı"
    
    def test_volatile_prices(self):
        """
        Volatil fiyatlar testi.
        
        Çok volatil fiyatlarla çalışabilmeli.
        """
        # Çok volatil fiyatlar: yukarı-aşağı
        actual_prices = np.array([100.0, 150.0, 80.0, 120.0, 90.0, 130.0])
        predictions = np.array([140.0, 90.0, 110.0, 100.0, 120.0])
        initial_capital = 10000.0
        
        result = simple_backtest(predictions, actual_prices, initial_capital)
        
        # Sonuç yapısı geçerli olmalı
        assert 'total_return' in result
        assert 'accuracy' in result
        assert 'num_trades' in result
        assert 'final_capital' in result
        
        # Değerler geçerli aralıklarda olmalı
        assert 0.0 <= result['accuracy'] <= 100.0
        assert result['num_trades'] >= 0
        assert result['final_capital'] > 0


class TestCalculateMetrics:
    """calculate_metrics fonksiyonu için birim testler."""
    
    def test_perfect_predictions(self):
        """
        Mükemmel tahminler testi.
        
        Tahminler gerçek değerlere eşit olduğunda,
        MSE, RMSE, MAE sıfır, R2 bir olmalı.
        """
        y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        y_pred = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        
        result = calculate_metrics(y_true, y_pred)
        
        # Mükemmel tahminlerde hatalar sıfır olmalı
        assert result['mse'] == 0.0
        assert result['rmse'] == 0.0
        assert result['mae'] == 0.0
        
        # R2 skoru 1.0 olmalı
        assert result['r2'] == 1.0
    
    def test_constant_predictions(self):
        """
        Sabit tahminler testi.
        
        Tüm tahminler aynı değerde olduğunda,
        metrikler hesaplanabilmeli.
        """
        y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        y_pred = np.array([115.0, 115.0, 115.0, 115.0, 115.0])  # Sabit tahmin
        
        result = calculate_metrics(y_true, y_pred)
        
        # Metrikler hesaplanmış olmalı
        assert result['mse'] > 0
        assert result['rmse'] > 0
        assert result['mae'] > 0
        
        # R2 düşük olmalı (sabit tahmin genellikle kötü performans gösterir)
        assert result['r2'] < 0.5
    
    def test_empty_arrays(self):
        """
        Boş array'ler testi.
        
        Boş array'lerle çağrıldığında varsayılan değerler döndürülmeli.
        """
        y_true = np.array([])
        y_pred = np.array([])
        
        result = calculate_metrics(y_true, y_pred)
        
        # Varsayılan değerleri kontrol et
        assert result['mse'] == 0.0
        assert result['rmse'] == 0.0
        assert result['mae'] == 0.0
        assert result['r2'] == 0.0
    
    def test_mismatched_lengths(self):
        """
        Uyumsuz uzunluklar testi.
        
        Farklı uzunluktaki array'lerle ValueError fırlatılmalı.
        """
        y_true = np.array([100.0, 110.0, 120.0])
        y_pred = np.array([105.0, 115.0])
        
        with pytest.raises(ValueError, match="must equal"):
            calculate_metrics(y_true, y_pred)
    
    def test_single_value(self):
        """
        Tek değer testi.
        
        Tek elemanlı array'lerle çalışabilmeli.
        """
        y_true = np.array([100.0])
        y_pred = np.array([105.0])
        
        result = calculate_metrics(y_true, y_pred)
        
        # Metrikler hesaplanmış olmalı
        assert result['mse'] > 0
        assert result['rmse'] > 0
        assert result['mae'] > 0
    
    def test_rmse_is_sqrt_of_mse(self):
        """
        RMSE = sqrt(MSE) testi.
        
        RMSE, MSE'nin karekökü olmalı.
        """
        y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        y_pred = np.array([102.0, 108.0, 125.0, 128.0, 145.0])
        
        result = calculate_metrics(y_true, y_pred)
        
        # RMSE, MSE'nin karekökü olmalı
        expected_rmse = np.sqrt(result['mse'])
        assert abs(result['rmse'] - expected_rmse) < 1e-6
    
    def test_metrics_non_negative(self):
        """
        Metrikler negatif olmayan testi.
        
        MSE, RMSE, MAE her zaman negatif olmamalı.
        """
        y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        y_pred = np.array([95.0, 115.0, 118.0, 135.0, 142.0])
        
        result = calculate_metrics(y_true, y_pred)
        
        # Tüm metrikler negatif olmamalı
        assert result['mse'] >= 0
        assert result['rmse'] >= 0
        assert result['mae'] >= 0
    
    def test_large_errors(self):
        """
        Büyük hatalar testi.
        
        Tahminler çok yanlış olduğunda, metrikler büyük değerler almalı.
        """
        y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        y_pred = np.array([200.0, 210.0, 220.0, 230.0, 240.0])  # Çok yüksek tahminler
        
        result = calculate_metrics(y_true, y_pred)
        
        # Hatalar büyük olmalı
        assert result['mse'] > 1000
        assert result['rmse'] > 30
        assert result['mae'] > 30
        
        # R2 negatif olmalı (ortalamadan daha kötü)
        assert result['r2'] < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
