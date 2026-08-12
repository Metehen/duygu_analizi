"""
Birim Testleri: XGBoost Model Modülü
XGBoost model eğitimi, tahmin, özellik önem ve kaydetme/yükleme fonksiyonları için testler.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
import xgboost as xgb

from src.model_xgboost import (
    prepare_classification_data,
    train_xgboost_model,
    get_feature_importance,
    predict_direction,
    save_model,
    load_model
)


class TestPrepareClassificationData:
    """prepare_classification_data fonksiyonu için testler."""
    
    def test_prepare_data_basic(self):
        """Basit veri hazırlama testi."""
        # Örnek DataFrame oluştur
        df = pd.DataFrame({
            'close': [100, 105, 103, 108, 110],
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [10, 20, 30, 40, 50]
        })
        
        X, y = prepare_classification_data(df)
        
        # X ve y boyutlarını kontrol et
        assert len(X) == 4  # Son satır çıkarılır
        assert len(y) == 4
        assert 'close' not in X.columns  # Hedef sütun çıkarılmalı
        assert 'feature_1' in X.columns
        assert 'feature_2' in X.columns
    
    def test_prepare_data_direction_labels(self):
        """Yön etiketlerinin doğru oluşturulduğunu kontrol et."""
        df = pd.DataFrame({
            'close': [100, 105, 103, 108, 110],  # yukarı, aşağı, yukarı, yukarı
            'feature_1': [1, 2, 3, 4, 5]
        })
        
        X, y = prepare_classification_data(df)
        
        # Beklenen etiketler: [1, 0, 1, 1]
        expected = pd.Series([1, 0, 1, 1], dtype='int32')
        pd.testing.assert_series_equal(y.reset_index(drop=True), expected, check_names=False)
    
    def test_prepare_data_empty_dataframe(self):
        """Boş DataFrame ile hata."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame boş olamaz"):
            prepare_classification_data(df)
    
    def test_prepare_data_missing_target_column(self):
        """Hedef sütun yoksa hata."""
        df = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [10, 20, 30]
        })
        
        with pytest.raises(ValueError, match="Hedef sütun.*bulunamadı"):
            prepare_classification_data(df)
    
    def test_prepare_data_insufficient_rows(self):
        """Yetersiz satır sayısı."""
        df = pd.DataFrame({
            'close': [100]
        })
        
        with pytest.raises(ValueError, match="En az 2 satır veri gerekli"):
            prepare_classification_data(df)
    
    def test_prepare_data_case_insensitive(self):
        """Hedef sütun case-insensitive olmalı."""
        df = pd.DataFrame({
            'Close': [100, 105, 103],  # Büyük harf
            'feature_1': [1, 2, 3]
        })
        
        X, y = prepare_classification_data(df, target_column='Close')
        
        assert len(X) == 2
        assert len(y) == 2


class TestTrainXGBoostModel:
    """train_xgboost_model fonksiyonu için testler."""
    
    def test_train_model_basic(self):
        """Basit model eğitimi."""
        # Veri oluştur
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50),
            'feature_2': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        # Model eğit
        model = train_xgboost_model(X_train, y_train)
        
        assert model is not None
        assert isinstance(model, xgb.XGBClassifier)
        assert hasattr(model, 'predict')
    
    def test_train_model_with_validation(self):
        """Validasyon verisi ile eğitim."""
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50),
            'feature_2': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        X_val = pd.DataFrame({
            'feature_1': np.random.rand(20),
            'feature_2': np.random.rand(20)
        })
        y_val = pd.Series(np.random.randint(0, 2, 20))
        
        model = train_xgboost_model(X_train, y_train, X_val, y_val)
        
        assert model is not None
    
    def test_train_model_custom_params(self):
        """Özel parametrelerle eğitim."""
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50),
            'feature_2': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        custom_params = {
            'max_depth': 3,
            'n_estimators': 50,
            'learning_rate': 0.05
        }
        
        model = train_xgboost_model(X_train, y_train, params=custom_params)
        
        assert model.max_depth == 3
        assert model.n_estimators == 50
        assert model.learning_rate == 0.05
    
    def test_train_model_empty_data(self):
        """Boş veri ile hata."""
        X_train = pd.DataFrame()
        y_train = pd.Series()
        
        with pytest.raises(ValueError, match="Eğitim verileri boş olamaz"):
            train_xgboost_model(X_train, y_train)
    
    def test_train_model_mismatched_sizes(self):
        """X ve y boyutları uyumsuz."""
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 40))  # Farklı boyut
        
        with pytest.raises(ValueError, match="boyutları uyumsuz"):
            train_xgboost_model(X_train, y_train)


class TestGetFeatureImportance:
    """get_feature_importance fonksiyonu için testler."""
    
    def test_feature_importance_basic(self):
        """Basit özellik önem testi."""
        # Model eğit
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50),
            'feature_2': np.random.rand(50),
            'feature_3': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        model = train_xgboost_model(X_train, y_train)
        
        # Özellik önemlerini al
        importance_df = get_feature_importance(model, feature_names=X_train.columns.tolist())
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == 3
    
    def test_feature_importance_sorted(self):
        """Özellik önemleri azalan sırada olmalı."""
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50),
            'feature_2': np.random.rand(50),
            'feature_3': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        model = train_xgboost_model(X_train, y_train)
        importance_df = get_feature_importance(model, feature_names=X_train.columns.tolist())
        
        # Sıralamanın doğru olduğunu kontrol et
        importances = importance_df['importance'].values
        assert np.all(importances[:-1] >= importances[1:])
    
    def test_feature_importance_untrained_model(self):
        """Eğitilmemiş model ile hata."""
        model = xgb.XGBClassifier()
        
        with pytest.raises(ValueError, match="Model henüz eğitilmemiş"):
            get_feature_importance(model)
    
    def test_feature_importance_without_names(self):
        """Özellik isimleri olmadan."""
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50),
            'feature_2': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        model = train_xgboost_model(X_train, y_train)
        importance_df = get_feature_importance(model)
        
        # Varsayılan isimler kullanılmalı
        assert len(importance_df) == 2
        assert importance_df['feature'].str.startswith('feature_').all()


class TestPredictDirection:
    """predict_direction fonksiyonu için testler."""
    
    def test_predict_direction_basic(self):
        """Basit tahmin testi."""
        # Model eğit
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50),
            'feature_2': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        model = train_xgboost_model(X_train, y_train)
        
        # Tahmin yap
        X_test = pd.DataFrame({
            'feature_1': np.random.rand(10),
            'feature_2': np.random.rand(10)
        })
        
        predictions, probabilities = predict_direction(model, X_test)
        
        assert len(predictions) == 10
        assert len(probabilities) == 10
        assert set(predictions).issubset({0, 1})
        assert np.all((probabilities >= 0) & (probabilities <= 1))
    
    def test_predict_direction_output_types(self):
        """Çıktı tiplerini kontrol et."""
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50),
            'feature_2': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        model = train_xgboost_model(X_train, y_train)
        
        X_test = pd.DataFrame({
            'feature_1': np.random.rand(10),
            'feature_2': np.random.rand(10)
        })
        
        predictions, probabilities = predict_direction(model, X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert isinstance(probabilities, np.ndarray)
    
    def test_predict_direction_empty_input(self):
        """Boş girdi ile hata."""
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        model = train_xgboost_model(X_train, y_train)
        
        X_test = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Girdi verileri boş olamaz"):
            predict_direction(model, X_test)
    
    def test_predict_direction_untrained_model(self):
        """Eğitilmemiş model ile hata."""
        model = xgb.XGBClassifier()
        
        X_test = pd.DataFrame({
            'feature_1': np.random.rand(10)
        })
        
        # XGBoost kendi hatasını fırlatır
        with pytest.raises(Exception):  # NotFittedError veya ValueError
            predict_direction(model, X_test)
    
    def test_predict_direction_deterministic(self):
        """Aynı girdi için aynı tahminler."""
        X_train = pd.DataFrame({
            'feature_1': np.random.rand(50),
            'feature_2': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        model = train_xgboost_model(X_train, y_train)
        
        X_test = pd.DataFrame({
            'feature_1': np.random.rand(10),
            'feature_2': np.random.rand(10)
        })
        
        predictions_1, probabilities_1 = predict_direction(model, X_test)
        predictions_2, probabilities_2 = predict_direction(model, X_test)
        
        np.testing.assert_array_equal(predictions_1, predictions_2)
        np.testing.assert_allclose(probabilities_1, probabilities_2, rtol=1e-5)


class TestSaveLoadModel:
    """save_model ve load_model fonksiyonları için testler."""
    
    def test_save_and_load_model(self):
        """Model kaydetme ve yükleme."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Model eğit
            X_train = pd.DataFrame({
                'feature_1': np.random.rand(50),
                'feature_2': np.random.rand(50)
            })
            y_train = pd.Series(np.random.randint(0, 2, 50))
            
            model = train_xgboost_model(X_train, y_train)
            
            # Kaydet
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            save_model(model, filepath)
            
            # Dosya oluşturuldu mu?
            assert os.path.exists(filepath)
            
            # Yükle
            loaded_model = load_model(filepath)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, xgb.XGBClassifier)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_and_load_predictions_match(self):
        """Kaydedilen ve yüklenen model aynı tahminleri üretmeli."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Model eğit
            X_train = pd.DataFrame({
                'feature_1': np.random.rand(50),
                'feature_2': np.random.rand(50)
            })
            y_train = pd.Series(np.random.randint(0, 2, 50))
            
            model = train_xgboost_model(X_train, y_train)
            
            # Test verisi
            X_test = pd.DataFrame({
                'feature_1': np.random.rand(10),
                'feature_2': np.random.rand(10)
            })
            
            # Orijinal tahminler
            predictions_original, probabilities_original = predict_direction(model, X_test)
            
            # Kaydet ve yükle
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            save_model(model, filepath)
            loaded_model = load_model(filepath)
            
            # Yüklenen modelden tahminler
            predictions_loaded, probabilities_loaded = predict_direction(loaded_model, X_test)
            
            # Tahminler eşit olmalı
            np.testing.assert_array_equal(predictions_original, predictions_loaded)
            np.testing.assert_allclose(probabilities_original, probabilities_loaded, rtol=1e-5)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_nonexistent_model(self):
        """Var olmayan model dosyası."""
        with pytest.raises(FileNotFoundError):
            load_model('nonexistent_model.pkl')
    
    def test_save_creates_directory(self):
        """Kaydetme dizini yoksa oluşturulmalı."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            X_train = pd.DataFrame({
                'feature_1': np.random.rand(50)
            })
            y_train = pd.Series(np.random.randint(0, 2, 50))
            
            model = train_xgboost_model(X_train, y_train)
            
            # Var olmayan alt dizin
            filepath = os.path.join(temp_dir, 'subdir', 'model.pkl')
            save_model(model, filepath)
            
            assert os.path.exists(filepath)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_untrained_model(self):
        """Eğitilmemiş model kaydetme - pickle her modeli kaydeder."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            model = xgb.XGBClassifier()
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            
            # Pickle eğitilmemiş modeli de kaydeder, bu beklenen davranış
            save_model(model, filepath)
            assert os.path.exists(filepath)
                
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
