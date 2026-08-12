"""
Birim Testler: Model Değerlendirme Modülü

Bu dosya, model_evaluation.py modülündeki fonksiyonlar için birim testleri içerir.
Belirli örnekler ve kenar durumları test edilir.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
import shutil
import json
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalışması için
import matplotlib.pyplot as plt

# src dizinini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_evaluation import (
    ModelTracker,
    calculate_comprehensive_metrics,
    calculate_error_distribution,
    plot_training_history,
    plot_predictions_vs_actual,
    plot_error_distribution,
    plot_metrics_comparison,
    save_model_report,
    load_model_reports,
    select_best_model
)


class TestModelTracker:
    """ModelTracker sınıfı için birim testler."""
    
    def test_init_creates_directory(self):
        """ModelTracker başlatıldığında dizin oluşturulmalıdır."""
        temp_dir = tempfile.mkdtemp()
        try:
            save_dir = os.path.join(temp_dir, 'new_dir/')
            tracker = ModelTracker(model_name='test_model', save_dir=save_dir)
            
            assert os.path.exists(save_dir), "Dizin oluşturulmadı"
            assert tracker.model_name == 'test_model'
            assert tracker.save_dir == save_dir
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_callbacks_returns_list(self):
        """get_callbacks() bir liste döndürmelidir."""
        temp_dir = tempfile.mkdtemp()
        try:
            tracker = ModelTracker(model_name='test_model', save_dir=temp_dir + '/')
            
            # TensorFlow yüklü değilse test atla
            try:
                callbacks = tracker.get_callbacks()
                assert isinstance(callbacks, list), "Callbacks liste olmalı"
                assert len(callbacks) == 3, "3 callback olmalı"
            except ImportError:
                pytest.skip("TensorFlow not installed")
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_callbacks_without_tensorflow(self):
        """TensorFlow yoksa ImportError fırlatmalıdır."""
        temp_dir = tempfile.mkdtemp()
        try:
            tracker = ModelTracker(model_name='test_model', save_dir=temp_dir + '/')
            
            # TensorFlow'u geçici olarak None yap
            import src.model_evaluation as me
            original_tf = me.tf
            me.tf = None
            
            try:
                with pytest.raises(ImportError, match="TensorFlow is required"):
                    tracker.get_callbacks()
            finally:
                me.tf = original_tf
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_training_history_creates_csv(self):
        """save_training_history() CSV dosyası oluşturmalıdır."""
        temp_dir = tempfile.mkdtemp()
        try:
            tracker = ModelTracker(model_name='test_model', save_dir=temp_dir + '/')
            
            # Mock history object
            class MockHistory:
                def __init__(self):
                    self.history = {
                        'loss': [1.0, 0.9, 0.8],
                        'val_loss': [1.1, 1.0, 0.9],
                        'mae': [0.5, 0.4, 0.3],
                        'val_mae': [0.6, 0.5, 0.4]
                    }
            
            mock_history = MockHistory()
            tracker.save_training_history(mock_history)
            
            # CSV dosyasının oluşturulduğunu kontrol et
            csv_path = os.path.join(temp_dir, 'test_model_history.csv')
            assert os.path.exists(csv_path), "CSV dosyası oluşturulmadı"
            
            # CSV'yi oku ve içeriği kontrol et
            df = pd.read_csv(csv_path)
            assert 'loss' in df.columns
            assert 'val_loss' in df.columns
            assert len(df) == 3
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_training_history_stores_reference(self):
        """save_training_history() history referansını saklamalıdır."""
        temp_dir = tempfile.mkdtemp()
        try:
            tracker = ModelTracker(model_name='test_model', save_dir=temp_dir + '/')
            
            class MockHistory:
                def __init__(self):
                    self.history = {'loss': [1.0, 0.9]}
            
            mock_history = MockHistory()
            tracker.save_training_history(mock_history)
            
            assert tracker.history is mock_history, "History referansı saklanmadı"
        finally:
            shutil.rmtree(temp_dir)


class TestCalculateComprehensiveMetrics:
    """calculate_comprehensive_metrics() fonksiyonu için birim testler."""
    
    def test_basic_metrics_calculation(self):
        """Temel metrikler doğru hesaplanmalıdır."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        assert 'directional_accuracy' in metrics
        
        # MSE ve RMSE ilişkisi
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']))
    
    def test_perfect_predictions(self):
        """Mükemmel tahminler için metrikler sıfır/bir olmalıdır."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        
        assert metrics['mse'] < 1e-10, "MSE sıfıra yakın olmalı"
        assert metrics['rmse'] < 1e-5, "RMSE sıfıra yakın olmalı"
        assert metrics['mae'] < 1e-5, "MAE sıfıra yakın olmalı"
        assert metrics['mape'] < 1e-3, "MAPE sıfıra yakın olmalı"
        assert metrics['r2'] > 0.99, "R² bire yakın olmalı"
    
    def test_with_zero_values(self):
        """Sıfır değerler içeren veriler için MAPE hesaplanmalıdır."""
        y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.1, 1.1, 2.1, 3.1, 4.1])
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        
        # MAPE hesaplanabilmeli (sıfır değerler atlanır)
        assert 'mape' in metrics
        assert np.isfinite(metrics['mape'])
    
    def test_all_zeros(self):
        """Tüm değerler sıfır ise MAPE sıfır olmalıdır."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 0.0, 0.0])
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        
        assert metrics['mape'] == 0.0, "Tüm sıfırlar için MAPE sıfır olmalı"
    
    def test_single_value(self):
        """Tek değer için directional_accuracy sıfır olmalıdır."""
        y_true = np.array([5.0])
        y_pred = np.array([5.1])
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        
        assert metrics['directional_accuracy'] == 0.0, \
            "Tek değer için directional_accuracy sıfır olmalı"
    
    def test_directional_accuracy_calculation(self):
        """Yönsel doğruluk doğru hesaplanmalıdır."""
        # Tüm yönler doğru
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        assert metrics['directional_accuracy'] == 100.0, \
            "Tüm yönler doğru ise 100% olmalı"
        
        # Tüm yönler yanlış - gerçek artarken tahmin azalıyor
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Tam ters yön
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        assert metrics['directional_accuracy'] == 0.0, \
            "Tüm yönler yanlış ise 0% olmalı"
    
    def test_negative_values(self):
        """Negatif değerler için metrikler hesaplanabilmelidir."""
        y_true = np.array([-5.0, -3.0, -1.0, 1.0, 3.0])
        y_pred = np.array([-4.8, -2.9, -1.1, 1.2, 2.8])
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        
        assert all(np.isfinite(v) for v in metrics.values()), \
            "Tüm metrikler finite olmalı"


class TestCalculateErrorDistribution:
    """calculate_error_distribution() fonksiyonu için birim testler."""
    
    def test_basic_error_distribution(self):
        """Temel hata dağılımı doğru hesaplanmalıdır."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.3, 4.8])
        
        error_dist = calculate_error_distribution(y_true, y_pred)
        
        assert 'errors' in error_dist
        assert 'mean_error' in error_dist
        assert 'std_error' in error_dist
        assert 'percentiles' in error_dist
        
        # Hataların doğru hesaplandığını kontrol et
        expected_errors = y_pred - y_true
        np.testing.assert_allclose(error_dist['errors'], expected_errors)
    
    def test_zero_errors(self):
        """Sıfır hata için istatistikler sıfır olmalıdır."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        error_dist = calculate_error_distribution(y_true, y_pred)
        
        assert np.allclose(error_dist['errors'], 0.0)
        assert abs(error_dist['mean_error']) < 1e-10
        assert error_dist['std_error'] < 1e-10
    
    def test_percentiles_ordering(self):
        """Percentile'lar artan sırada olmalıdır."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5, 8.5, 8.5, 10.5])
        
        error_dist = calculate_error_distribution(y_true, y_pred)
        
        p25 = error_dist['percentiles']['25']
        p50 = error_dist['percentiles']['50']
        p75 = error_dist['percentiles']['75']
        
        assert p25 <= p50, "25. percentile 50.'den küçük veya eşit olmalı"
        assert p50 <= p75, "50. percentile 75.'den küçük veya eşit olmalı"
    
    def test_constant_error(self):
        """Sabit hata için std sıfır olmalıdır."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true + 0.5  # Sabit hata
        
        error_dist = calculate_error_distribution(y_true, y_pred)
        
        assert abs(error_dist['mean_error'] - 0.5) < 1e-10
        assert error_dist['std_error'] < 1e-10


class TestPlottingFunctions:
    """Görselleştirme fonksiyonları için birim testler."""
    
    def test_plot_training_history_basic(self):
        """plot_training_history() figure döndürmelidir."""
        history_df = pd.DataFrame({
            'loss': [1.0, 0.9, 0.8, 0.7],
            'val_loss': [1.1, 1.0, 0.9, 0.8],
            'mae': [0.5, 0.4, 0.3, 0.2],
            'val_mae': [0.6, 0.5, 0.4, 0.3]
        })
        
        fig = plot_training_history(history_df)
        
        assert isinstance(fig, plt.Figure), "Figure döndürülmeli"
        assert len(fig.axes) == 2, "2 subplot olmalı"
        plt.close(fig)
    
    def test_plot_training_history_without_mae(self):
        """MAE olmadan da çalışmalıdır."""
        history_df = pd.DataFrame({
            'loss': [1.0, 0.9, 0.8],
            'val_loss': [1.1, 1.0, 0.9]
        })
        
        fig = plot_training_history(history_df)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_predictions_vs_actual_basic(self):
        """plot_predictions_vs_actual() figure döndürmelidir."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        fig = plot_predictions_vs_actual(y_true, y_pred)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2, "2 subplot olmalı (zaman serisi + scatter)"
        plt.close(fig)
    
    def test_plot_predictions_vs_actual_with_dates(self):
        """Tarihlerle birlikte çalışmalıdır."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        dates = pd.date_range('2024-01-01', periods=5)
        
        fig = plot_predictions_vs_actual(y_true, y_pred, dates)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_error_distribution_basic(self):
        """plot_error_distribution() figure döndürmelidir."""
        errors = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.3, 0.4])
        
        fig = plot_error_distribution(errors)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1, "1 subplot olmalı"
        plt.close(fig)
    
    def test_plot_metrics_comparison_basic(self):
        """plot_metrics_comparison() figure döndürmelidir."""
        metrics_dict = {
            'model_1': {
                'rmse': 0.5,
                'mae': 0.3,
                'mape': 5.0,
                'r2': 0.9,
                'directional_accuracy': 85.0
            },
            'model_2': {
                'rmse': 0.6,
                'mae': 0.4,
                'mape': 6.0,
                'r2': 0.85,
                'directional_accuracy': 80.0
            }
        }
        
        fig = plot_metrics_comparison(metrics_dict)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6, "6 subplot olmalı"
        plt.close(fig)
    
    def test_plot_metrics_comparison_single_model(self):
        """Tek model için de çalışmalıdır."""
        metrics_dict = {
            'model_1': {
                'rmse': 0.5,
                'mae': 0.3,
                'mape': 5.0,
                'r2': 0.9,
                'directional_accuracy': 85.0
            }
        }
        
        fig = plot_metrics_comparison(metrics_dict)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestModelReportFunctions:
    """Model rapor fonksiyonları için birim testler."""
    
    def test_save_model_report_creates_file(self):
        """save_model_report() JSON dosyası oluşturmalıdır."""
        temp_dir = tempfile.mkdtemp()
        try:
            model_name = 'test_model'
            metrics = {
                'mse': 0.25,
                'rmse': 0.5,
                'mae': 0.3,
                'r2': 0.9
            }
            save_path = os.path.join(temp_dir, 'report.json')
            
            save_model_report(model_name, metrics, save_path)
            
            assert os.path.exists(save_path), "Rapor dosyası oluşturulmadı"
            
            # Dosyayı oku ve içeriği kontrol et
            with open(save_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            assert report['model_name'] == model_name
            assert 'timestamp' in report
            assert report['metrics'] == metrics
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_model_report_with_turkish_characters(self):
        """Türkçe karakterler doğru kaydedilmelidir."""
        temp_dir = tempfile.mkdtemp()
        try:
            model_name = 'türkçe_model_ışık'
            metrics = {'rmse': 0.5}
            save_path = os.path.join(temp_dir, 'report.json')
            
            save_model_report(model_name, metrics, save_path)
            
            with open(save_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            assert report['model_name'] == model_name
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_model_reports_empty_directory(self):
        """Boş dizin için boş dictionary döndürmelidir."""
        temp_dir = tempfile.mkdtemp()
        try:
            reports = load_model_reports(temp_dir)
            
            assert isinstance(reports, dict)
            assert len(reports) == 0
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_model_reports_nonexistent_directory(self):
        """Var olmayan dizin için boş dictionary döndürmelidir."""
        reports = load_model_reports('/nonexistent/directory/path')
        
        assert isinstance(reports, dict)
        assert len(reports) == 0
    
    def test_load_model_reports_multiple_files(self):
        """Birden fazla rapor dosyası yüklenebilmelidir."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Birden fazla rapor kaydet
            for i in range(3):
                model_name = f'model_{i}'
                metrics = {'rmse': 0.5 + i * 0.1}
                save_path = os.path.join(temp_dir, f'report_{i}.json')
                save_model_report(model_name, metrics, save_path)
            
            # Raporları yükle
            reports = load_model_reports(temp_dir)
            
            assert len(reports) == 3
            assert 'model_0' in reports
            assert 'model_1' in reports
            assert 'model_2' in reports
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_model_reports_ignores_invalid_json(self):
        """Geçersiz JSON dosyaları atlanmalıdır."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Geçerli rapor
            save_model_report('valid_model', {'rmse': 0.5}, 
                            os.path.join(temp_dir, 'valid.json'))
            
            # Geçersiz JSON
            with open(os.path.join(temp_dir, 'invalid.json'), 'w') as f:
                f.write('{ invalid json }')
            
            # JSON ama model_name eksik
            with open(os.path.join(temp_dir, 'no_name.json'), 'w') as f:
                json.dump({'metrics': {'rmse': 0.5}}, f)
            
            reports = load_model_reports(temp_dir)
            
            # Sadece geçerli rapor yüklenmeli
            assert len(reports) == 1
            assert 'valid_model' in reports
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_model_reports_ignores_non_json_files(self):
        """JSON olmayan dosyalar atlanmalıdır."""
        temp_dir = tempfile.mkdtemp()
        try:
            # JSON rapor
            save_model_report('model_1', {'rmse': 0.5}, 
                            os.path.join(temp_dir, 'report.json'))
            
            # Metin dosyası
            with open(os.path.join(temp_dir, 'readme.txt'), 'w') as f:
                f.write('This is not a JSON file')
            
            reports = load_model_reports(temp_dir)
            
            assert len(reports) == 1
            assert 'model_1' in reports
        finally:
            shutil.rmtree(temp_dir)


class TestSelectBestModel:
    """select_best_model() fonksiyonu için birim testler."""
    
    def test_select_best_model_minimize(self):
        """minimize=True için en düşük değer seçilmelidir."""
        metrics_dict = {
            'model_1': {'rmse': 0.5, 'mae': 0.3},
            'model_2': {'rmse': 0.3, 'mae': 0.2},  # En iyi
            'model_3': {'rmse': 0.7, 'mae': 0.4}
        }
        
        best_model = select_best_model(metrics_dict, primary_metric='rmse', minimize=True)
        
        assert best_model == 'model_2', "En düşük RMSE'ye sahip model seçilmeli"
    
    def test_select_best_model_maximize(self):
        """minimize=False için en yüksek değer seçilmelidir."""
        metrics_dict = {
            'model_1': {'r2': 0.85, 'rmse': 0.5},
            'model_2': {'r2': 0.92, 'rmse': 0.3},  # En iyi
            'model_3': {'r2': 0.80, 'rmse': 0.7}
        }
        
        best_model = select_best_model(metrics_dict, primary_metric='r2', minimize=False)
        
        assert best_model == 'model_2', "En yüksek R²'ye sahip model seçilmeli"
    
    def test_select_best_model_tie(self):
        """Eşit değerler için ilk model seçilmelidir."""
        metrics_dict = {
            'model_1': {'rmse': 0.5},
            'model_2': {'rmse': 0.5},
            'model_3': {'rmse': 0.5}
        }
        
        best_model = select_best_model(metrics_dict, primary_metric='rmse', minimize=True)
        
        # İlk model seçilmeli (dictionary sırası)
        assert best_model in metrics_dict
    
    def test_select_best_model_missing_metric(self):
        """Eksik metrik için inf değeri kullanılmalıdır."""
        metrics_dict = {
            'model_1': {'rmse': 0.5, 'mae': 0.3},
            'model_2': {'mae': 0.2},  # rmse eksik
            'model_3': {'rmse': 0.7, 'mae': 0.4}
        }
        
        best_model = select_best_model(metrics_dict, primary_metric='rmse', minimize=True)
        
        # model_2 rmse'si olmadığı için seçilmemeli
        assert best_model in ['model_1', 'model_3']
    
    def test_select_best_model_single_model(self):
        """Tek model için o model seçilmelidir."""
        metrics_dict = {
            'model_1': {'rmse': 0.5}
        }
        
        best_model = select_best_model(metrics_dict, primary_metric='rmse', minimize=True)
        
        assert best_model == 'model_1'
    
    def test_select_best_model_different_metrics(self):
        """Farklı metrikler için farklı modeller seçilebilmelidir."""
        metrics_dict = {
            'model_1': {'rmse': 0.3, 'mae': 0.5},  # En iyi RMSE
            'model_2': {'rmse': 0.5, 'mae': 0.2}   # En iyi MAE
        }
        
        best_rmse = select_best_model(metrics_dict, primary_metric='rmse', minimize=True)
        best_mae = select_best_model(metrics_dict, primary_metric='mae', minimize=True)
        
        assert best_rmse == 'model_1'
        assert best_mae == 'model_2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
