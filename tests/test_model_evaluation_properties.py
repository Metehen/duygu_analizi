"""
Özellik Tabanlı Testler: Model Değerlendirme Modülü

Bu dosya, model_evaluation.py modülündeki fonksiyonlar için özellik tabanlı testleri içerir.
Hypothesis kütüphanesi kullanılarak evrensel doğruluk özellikleri test edilir.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
import shutil
import json
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime

# src dizinini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_evaluation import (
    ModelTracker,
    calculate_comprehensive_metrics,
    calculate_error_distribution,
    save_model_report,
    load_model_reports,
    select_best_model
)


# Property 13: Model Eğitim Geçmişi Kaydı
# **Validates: Requirements 7.1**

@given(
    st.integers(min_value=5, max_value=50),  # num_epochs
    st.lists(st.floats(min_value=0.001, max_value=10.0), min_size=5, max_size=50)  # loss values
)
@settings(max_examples=10, deadline=None)
def test_property_13_training_history_recording(num_epochs, loss_values):
    """
    Feature: turkish-sentiment-stock-prediction, Property 13:
    Herhangi bir model eğitim geçmişi için, ModelTracker.save_training_history() 
    çağrıldığında, eğitim metrikleri CSV formatında kaydedilmelidir. Kaydedilen 
    dosya okunabilir olmalı ve orijinal metriklerle eşleşmelidir.
    
    **Validates: Requirements 7.1**
    """
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # ModelTracker oluştur
        tracker = ModelTracker(model_name='test_model', save_dir=temp_dir + '/')
        
        # Sahte eğitim geçmişi oluştur
        num_epochs = min(num_epochs, len(loss_values))
        history_dict = {
            'loss': loss_values[:num_epochs],
            'val_loss': [v * 1.1 for v in loss_values[:num_epochs]],
            'mae': [v * 0.5 for v in loss_values[:num_epochs]],
            'val_mae': [v * 0.6 for v in loss_values[:num_epochs]]
        }
        
        # Mock history object
        class MockHistory:
            def __init__(self, history):
                self.history = history
        
        mock_history = MockHistory(history_dict)
        
        # Eğitim geçmişini kaydet
        tracker.save_training_history(mock_history)
        
        # CSV dosyasının oluşturulduğunu kontrol et
        csv_path = os.path.join(temp_dir, 'test_model_history.csv')
        assert os.path.exists(csv_path), "Eğitim geçmişi CSV dosyası oluşturulmadı"
        
        # CSV'yi oku
        loaded_df = pd.read_csv(csv_path)
        
        # Sütunların varlığını kontrol et
        assert 'loss' in loaded_df.columns, "loss sütunu eksik"
        assert 'val_loss' in loaded_df.columns, "val_loss sütunu eksik"
        assert 'mae' in loaded_df.columns, "mae sütunu eksik"
        assert 'val_mae' in loaded_df.columns, "val_mae sütunu eksik"
        
        # Satır sayısının eşleştiğini kontrol et
        assert len(loaded_df) == num_epochs, \
            f"Epoch sayısı eşleşmiyor: beklenen {num_epochs}, alınan {len(loaded_df)}"
        
        # Değerlerin eşleştiğini kontrol et (küçük tolerans)
        np.testing.assert_allclose(
            loaded_df['loss'].values,
            history_dict['loss'],
            rtol=1e-5,
            err_msg="Loss değerleri eşleşmiyor"
        )
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


@given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
@settings(max_examples=10, deadline=None)
def test_property_13_model_name_in_filename(model_name):
    """
    Property 13 ek test: Model adı dosya isimlerinde korunmalıdır.
    
    **Validates: Requirements 7.1**
    """
    # Geçersiz karakterleri temizle
    model_name = model_name.replace('/', '_').replace('\\', '_')
    if not model_name:
        model_name = 'test'
    
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # ModelTracker oluştur
        tracker = ModelTracker(model_name=model_name, save_dir=temp_dir + '/')
        
        # Sahte eğitim geçmişi
        history_dict = {'loss': [1.0, 0.9, 0.8], 'val_loss': [1.1, 1.0, 0.9]}
        
        class MockHistory:
            def __init__(self, history):
                self.history = history
        
        tracker.save_training_history(MockHistory(history_dict))
        
        # Dosya isminin model adını içerdiğini kontrol et
        csv_path = os.path.join(temp_dir, f'{model_name}_history.csv')
        assert os.path.exists(csv_path), \
            f"Beklenen dosya bulunamadı: {csv_path}"
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


# Property 14: Kapsamlı Metrik Hesaplama
# **Validates: Requirements 7.2**

@given(
    st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=10, max_size=100),
    st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=10, max_size=100)
)
@settings(max_examples=15, deadline=None)
def test_property_14_comprehensive_metrics_structure(y_true_list, y_pred_list):
    """
    Feature: turkish-sentiment-stock-prediction, Property 14:
    Herhangi bir geçerli y_true ve y_pred array'i için, 
    calculate_comprehensive_metrics() fonksiyonu çağrıldığında, dönen dictionary 
    tüm gerekli metrikleri içermelidir: mse, rmse, mae, mape, r2, directional_accuracy. 
    Tüm değerler numeric ve finite olmalıdır.
    
    **Validates: Requirements 7.2**
    """
    # Aynı uzunlukta array'ler oluştur
    min_len = min(len(y_true_list), len(y_pred_list))
    assume(min_len >= 10)
    
    y_true = np.array(y_true_list[:min_len])
    y_pred = np.array(y_pred_list[:min_len])
    
    # Metrikleri hesapla
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    
    # Dictionary olduğunu kontrol et
    assert isinstance(metrics, dict), \
        f"Metrikler dictionary olmalı, alınan: {type(metrics)}"
    
    # Gerekli anahtarların varlığını kontrol et
    required_keys = ['mse', 'rmse', 'mae', 'mape', 'r2', 'directional_accuracy']
    for key in required_keys:
        assert key in metrics, f"Gerekli metrik eksik: {key}"
    
    # Tüm değerlerin numeric olduğunu kontrol et
    for key, value in metrics.items():
        assert isinstance(value, (int, float)), \
            f"{key} numeric olmalı, alınan: {type(value)}"
        assert np.isfinite(value), \
            f"{key} finite olmalı, alınan: {value}"
    
    # RMSE'nin MSE'nin karekökü olduğunu kontrol et
    np.testing.assert_allclose(
        metrics['rmse'],
        np.sqrt(metrics['mse']),
        rtol=1e-5,
        err_msg="RMSE, MSE'nin karekökü olmalı"
    )
    
    # Metriklerin makul aralıklarda olduğunu kontrol et
    assert metrics['mse'] >= 0, "MSE negatif olamaz"
    assert metrics['rmse'] >= 0, "RMSE negatif olamaz"
    assert metrics['mae'] >= 0, "MAE negatif olamaz"
    assert metrics['mape'] >= 0, "MAPE negatif olamaz"
    assert 0 <= metrics['directional_accuracy'] <= 100, \
        f"Directional accuracy 0-100 arasında olmalı, alınan: {metrics['directional_accuracy']}"


@given(st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=10, max_size=50))
@settings(max_examples=10, deadline=None)
def test_property_14_perfect_predictions(y_true_list):
    """
    Property 14 ek test: Mükemmel tahminler için metrikler beklenen değerlerde olmalıdır.
    
    **Validates: Requirements 7.2**
    """
    y_true = np.array(y_true_list)
    y_pred = y_true.copy()  # Mükemmel tahmin
    
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    
    # Mükemmel tahmin için beklenen değerler
    assert metrics['mse'] < 1e-10, "Mükemmel tahmin için MSE ~0 olmalı"
    assert metrics['rmse'] < 1e-5, "Mükemmel tahmin için RMSE ~0 olmalı"
    assert metrics['mae'] < 1e-5, "Mükemmel tahmin için MAE ~0 olmalı"
    assert metrics['mape'] < 1e-3, "Mükemmel tahmin için MAPE ~0 olmalı"
    assert metrics['r2'] > 0.99, "Mükemmel tahmin için R² ~1 olmalı"


@given(
    st.lists(st.floats(min_value=1.0, max_value=100.0), min_size=10, max_size=50),
    st.floats(min_value=0.1, max_value=10.0)
)
@settings(max_examples=10, deadline=None)
def test_property_14_constant_error(y_true_list, constant_error):
    """
    Property 14 ek test: Sabit hata için MAE, hata miktarına eşit olmalıdır.
    
    **Validates: Requirements 7.2**
    """
    y_true = np.array(y_true_list)
    y_pred = y_true + constant_error  # Sabit hata ekle
    
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    
    # MAE sabit hataya eşit olmalı
    np.testing.assert_allclose(
        metrics['mae'],
        abs(constant_error),
        rtol=1e-5,
        err_msg=f"Sabit hata için MAE {abs(constant_error)} olmalı"
    )


# Property 15: Hata Dağılımı Yapısı
# **Validates: Requirements 7.2**

@given(
    st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=20, max_size=100),
    st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=20, max_size=100)
)
@settings(max_examples=15, deadline=None)
def test_property_15_error_distribution_structure(y_true_list, y_pred_list):
    """
    Feature: turkish-sentiment-stock-prediction, Property 15:
    Herhangi bir geçerli y_true ve y_pred array'i için, 
    calculate_error_distribution() fonksiyonu çağrıldığında, dönen dictionary 
    errors, mean_error, std_error ve percentiles anahtarlarını içermelidir. 
    Percentiles 25, 50, 75 değerlerini içermelidir ve artan sırada olmalıdır.
    
    **Validates: Requirements 7.2**
    """
    # Aynı uzunlukta array'ler oluştur
    min_len = min(len(y_true_list), len(y_pred_list))
    assume(min_len >= 20)
    
    y_true = np.array(y_true_list[:min_len])
    y_pred = np.array(y_pred_list[:min_len])
    
    # Hata dağılımını hesapla
    error_dist = calculate_error_distribution(y_true, y_pred)
    
    # Dictionary olduğunu kontrol et
    assert isinstance(error_dist, dict), \
        f"Hata dağılımı dictionary olmalı, alınan: {type(error_dist)}"
    
    # Gerekli anahtarların varlığını kontrol et
    required_keys = ['errors', 'mean_error', 'std_error', 'percentiles']
    for key in required_keys:
        assert key in error_dist, f"Gerekli anahtar eksik: {key}"
    
    # errors array'inin doğru uzunlukta olduğunu kontrol et
    assert len(error_dist['errors']) == min_len, \
        f"Hata array'i uzunluğu eşleşmiyor: beklenen {min_len}, alınan {len(error_dist['errors'])}"
    
    # mean_error ve std_error'ın numeric olduğunu kontrol et
    assert isinstance(error_dist['mean_error'], (int, float)), \
        f"mean_error numeric olmalı, alınan: {type(error_dist['mean_error'])}"
    assert isinstance(error_dist['std_error'], (int, float)), \
        f"std_error numeric olmalı, alınan: {type(error_dist['std_error'])}"
    
    # std_error'ın negatif olmadığını kontrol et
    assert error_dist['std_error'] >= 0, "std_error negatif olamaz"
    
    # percentiles'ın dictionary olduğunu kontrol et
    assert isinstance(error_dist['percentiles'], dict), \
        "percentiles dictionary olmalı"
    
    # Gerekli percentile'ların varlığını kontrol et
    required_percentiles = ['25', '50', '75']
    for p in required_percentiles:
        assert p in error_dist['percentiles'], f"Percentile {p} eksik"
    
    # Percentile'ların artan sırada olduğunu kontrol et
    p25 = error_dist['percentiles']['25']
    p50 = error_dist['percentiles']['50']
    p75 = error_dist['percentiles']['75']
    
    assert p25 <= p50, f"25. percentile (={p25}) 50. percentile'dan (={p50}) büyük olamaz"
    assert p50 <= p75, f"50. percentile (={p50}) 75. percentile'dan (={p75}) büyük olamaz"
    
    # Hataların doğru hesaplandığını kontrol et
    expected_errors = y_pred - y_true
    np.testing.assert_allclose(
        error_dist['errors'],
        expected_errors,
        rtol=1e-5,
        err_msg="Hatalar doğru hesaplanmadı"
    )


@given(st.lists(st.floats(min_value=1.0, max_value=100.0), min_size=20, max_size=50))
@settings(max_examples=10, deadline=None)
def test_property_15_zero_error_distribution(y_true_list):
    """
    Property 15 ek test: Sıfır hata için dağılım istatistikleri sıfır olmalıdır.
    
    **Validates: Requirements 7.2**
    """
    y_true = np.array(y_true_list)
    y_pred = y_true.copy()  # Mükemmel tahmin
    
    error_dist = calculate_error_distribution(y_true, y_pred)
    
    # Tüm hatalar sıfır olmalı
    assert np.allclose(error_dist['errors'], 0, atol=1e-10), \
        "Mükemmel tahmin için hatalar sıfır olmalı"
    assert abs(error_dist['mean_error']) < 1e-10, \
        "Mükemmel tahmin için ortalama hata sıfır olmalı"
    assert error_dist['std_error'] < 1e-10, \
        "Mükemmel tahmin için std hata sıfır olmalı"


# Property 16: Model Raporu Serileştirme Round-Trip
# **Validates: Requirements 7.4**

@given(
    st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    st.dictionaries(
        keys=st.sampled_from(['mse', 'rmse', 'mae', 'mape', 'r2', 'directional_accuracy']),
        values=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=6
    )
)
@settings(max_examples=15, deadline=None)
def test_property_16_model_report_serialization_roundtrip(model_name, metrics):
    """
    Feature: turkish-sentiment-stock-prediction, Property 16:
    Herhangi bir model adı ve metrik dictionary'si için, save_model_report() 
    ile kaydedilen rapor, load_model_reports() ile yüklendiğinde, orijinal 
    metriklerle eşleşmelidir. Timestamp ve model_name alanları korunmalıdır.
    
    **Validates: Requirements 7.4**
    """
    # Geçersiz karakterleri temizle
    model_name = model_name.replace('/', '_').replace('\\', '_')
    if not model_name:
        model_name = 'test'
    
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Rapor kaydet
        report_path = os.path.join(temp_dir, f'{model_name}_report.json')
        save_model_report(model_name, metrics, report_path)
        
        # Dosyanın oluşturulduğunu kontrol et
        assert os.path.exists(report_path), "Rapor dosyası oluşturulmadı"
        
        # Raporları yükle
        reports = load_model_reports(temp_dir)
        
        # Model adının dictionary'de olduğunu kontrol et
        assert model_name in reports, f"Model {model_name} yüklenen raporlarda bulunamadı"
        
        loaded_report = reports[model_name]
        
        # Gerekli alanların varlığını kontrol et
        assert 'model_name' in loaded_report, "model_name alanı eksik"
        assert 'timestamp' in loaded_report, "timestamp alanı eksik"
        assert 'metrics' in loaded_report, "metrics alanı eksik"
        
        # Model adının korunduğunu kontrol et
        assert loaded_report['model_name'] == model_name, \
            f"Model adı eşleşmiyor: beklenen {model_name}, alınan {loaded_report['model_name']}"
        
        # Timestamp'in geçerli ISO format olduğunu kontrol et
        try:
            datetime.fromisoformat(loaded_report['timestamp'])
        except ValueError:
            pytest.fail(f"Geçersiz timestamp formatı: {loaded_report['timestamp']}")
        
        # Metriklerin eşleştiğini kontrol et
        loaded_metrics = loaded_report['metrics']
        for key, value in metrics.items():
            assert key in loaded_metrics, f"Metrik {key} yüklenen raporda eksik"
            np.testing.assert_allclose(
                loaded_metrics[key],
                value,
                rtol=1e-5,
                err_msg=f"Metrik {key} eşleşmiyor"
            )
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


@given(
    st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
            st.dictionaries(
                keys=st.sampled_from(['mse', 'rmse', 'mae']),
                values=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=2,
                max_size=3
            )
        ),
        min_size=2,
        max_size=5,
        unique_by=lambda x: x[0]
    )
)
@settings(max_examples=10, deadline=None)
def test_property_16_multiple_reports(model_metrics_list):
    """
    Property 16 ek test: Birden fazla model raporu kaydedilip yüklenebilmelidir.
    
    **Validates: Requirements 7.4**
    """
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Tüm raporları kaydet
        for model_name, metrics in model_metrics_list:
            model_name = model_name.replace('/', '_').replace('\\', '_')
            if not model_name:
                model_name = f'test_{hash(str(metrics))}'
            
            report_path = os.path.join(temp_dir, f'{model_name}_report.json')
            save_model_report(model_name, metrics, report_path)
        
        # Tüm raporları yükle
        reports = load_model_reports(temp_dir)
        
        # Yüklenen rapor sayısının doğru olduğunu kontrol et
        assert len(reports) == len(model_metrics_list), \
            f"Rapor sayısı eşleşmiyor: beklenen {len(model_metrics_list)}, alınan {len(reports)}"
        
    finally:
        # Temizle
        shutil.rmtree(temp_dir)


# Property 17: En İyi Model Seçimi Tutarlılığı
# **Validates: Requirements 7.4**

@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        values=st.dictionaries(
            keys=st.sampled_from(['rmse', 'mae', 'r2']),
            values=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=3
        ),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=15, deadline=None)
def test_property_17_best_model_selection_consistency(metrics_dict):
    """
    Feature: turkish-sentiment-stock-prediction, Property 17:
    Herhangi bir model metrikleri dictionary'si için, select_best_model() 
    fonksiyonu çağrıldığında, seçilen model primary_metric'e göre en iyi 
    değere sahip olmalıdır. Minimize=True için en düşük, False için en yüksek 
    değer seçilmelidir. Aynı girdi için fonksiyon deterministik olmalıdır.
    
    **Validates: Requirements 7.4**
    """
    assume(len(metrics_dict) >= 2)
    
    # Tüm modellerin primary_metric'e sahip olduğundan emin ol
    primary_metric = 'rmse'
    for model_name, metrics in metrics_dict.items():
        if primary_metric not in metrics:
            metrics[primary_metric] = np.random.uniform(0.1, 100.0)
    
    # En iyi modeli seç (minimize=True)
    best_model_min = select_best_model(metrics_dict, primary_metric=primary_metric, minimize=True)
    
    # Seçilen modelin dictionary'de olduğunu kontrol et
    assert best_model_min in metrics_dict, \
        f"Seçilen model {best_model_min} metrics_dict'te bulunamadı"
    
    # Seçilen modelin en düşük değere sahip olduğunu kontrol et
    best_value = metrics_dict[best_model_min][primary_metric]
    for model_name, metrics in metrics_dict.items():
        assert best_value <= metrics[primary_metric], \
            f"Seçilen model en düşük değere sahip değil: {best_model_min}={best_value}, {model_name}={metrics[primary_metric]}"
    
    # En iyi modeli seç (minimize=False)
    best_model_max = select_best_model(metrics_dict, primary_metric='r2', minimize=False)
    
    # r2 için en yüksek değeri kontrol et
    if 'r2' in metrics_dict[best_model_max]:
        best_r2 = metrics_dict[best_model_max]['r2']
        for model_name, metrics in metrics_dict.items():
            if 'r2' in metrics:
                assert best_r2 >= metrics['r2'], \
                    f"Seçilen model en yüksek r2'ye sahip değil"


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        values=st.dictionaries(
            keys=st.sampled_from(['rmse', 'mae']),
            values=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=2
        ),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=10, deadline=None)
def test_property_17_deterministic_selection(metrics_dict):
    """
    Property 17 ek test: Aynı girdi için fonksiyon deterministik olmalıdır.
    
    **Validates: Requirements 7.4**
    """
    assume(len(metrics_dict) >= 2)
    
    # Tüm modellerin rmse'ye sahip olduğundan emin ol
    for model_name, metrics in metrics_dict.items():
        if 'rmse' not in metrics:
            metrics['rmse'] = np.random.uniform(0.1, 100.0)
    
    # Birden fazla kez seç
    selections = [
        select_best_model(metrics_dict, primary_metric='rmse', minimize=True)
        for _ in range(3)
    ]
    
    # Tüm seçimlerin aynı olduğunu kontrol et
    assert all(s == selections[0] for s in selections), \
        f"Fonksiyon deterministik değil: {selections}"


@given(
    st.integers(min_value=2, max_value=10),
    st.floats(min_value=0.1, max_value=10.0)
)
@settings(max_examples=10, deadline=None)
def test_property_17_unique_best_model(num_models, base_value):
    """
    Property 17 ek test: Benzersiz en iyi değer için doğru model seçilmelidir.
    
    **Validates: Requirements 7.4**
    """
    # Metrikler oluştur (bir model açıkça en iyi)
    metrics_dict = {}
    for i in range(num_models):
        model_name = f'model_{i}'
        if i == 0:
            # İlk model en iyi
            metrics_dict[model_name] = {'rmse': base_value}
        else:
            # Diğer modeller daha kötü
            metrics_dict[model_name] = {'rmse': base_value + i}
    
    # En iyi modeli seç
    best_model = select_best_model(metrics_dict, primary_metric='rmse', minimize=True)
    
    # İlk modelin seçildiğini kontrol et
    assert best_model == 'model_0', \
        f"En düşük RMSE'ye sahip model seçilmedi: beklenen model_0, alınan {best_model}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
