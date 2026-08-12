"""
End-to-end entegrasyon testleri.

Bu test dosyası, tüm modüllerin birlikte çalıştığını doğrular.
Tam analiz akışını test eder: haber toplama -> duygu analizi -> 
özellik hazırlama -> model eğitimi -> tahmin -> geriye dönük test.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta

from src.news_fetcher import fetch_rss_news, save_news, load_news
from src.sentiment import SentimentAnalyzer
from src.features import (
    fetch_stock_data,
    calculate_technical_indicators,
    merge_sentiment_with_prices,
    prepare_training_data
)
from src.model_lstm import build_lstm_model, train_lstm_model, predict_lstm
from src.model_xgboost import (
    prepare_classification_data,
    train_xgboost_model,
    get_feature_importance,
    predict_direction
)
from src.backtest import simple_backtest, calculate_metrics


class TestEndToEndIntegration:
    """End-to-end entegrasyon test sınıfı."""
    
    def test_full_analysis_pipeline_with_mock_data(self):
        """
        Tam analiz akışını mock verilerle test eder.
        
        Bu test, gerçek RSS ve API çağrıları yapmadan tüm modüllerin
        birlikte çalıştığını doğrular.
        """
        # 1. Mock haber verisi oluştur
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        news_data = pd.DataFrame({
            'date': dates,
            'title': [f'Ekonomi Haberi {i}' for i in range(30)],
            'content': [
                'Pozitif ekonomik gelişme bekleniyor. Büyüme devam ediyor.' if i % 2 == 0
                else 'Negatif piyasa koşulları. Düşüş yaşanıyor.'
                for i in range(30)
            ],
            'source': ['Test Kaynak'] * 30
        })
        
        # 2. Duygu analizi yap
        analyzer = SentimentAnalyzer(method='lexicon')
        sentiment_scores = []
        for content in news_data['content']:
            score = analyzer.analyze(content)
            sentiment_scores.append(score)
        
        sentiment_df = pd.DataFrame(sentiment_scores)
        sentiment_df['date'] = news_data['date']
        
        # Duygu analizi sonuçlarını doğrula
        assert 'positive' in sentiment_df.columns
        assert 'negative' in sentiment_df.columns
        assert 'neutral' in sentiment_df.columns
        assert 'compound' in sentiment_df.columns
        assert len(sentiment_df) == 30
        
        # 3. Mock hisse fiyat verisi oluştur
        price_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 30),
            'high': np.random.uniform(110, 120, 30),
            'low': np.random.uniform(90, 100, 30),
            'close': np.random.uniform(100, 110, 30),
            'volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)
        
        # 4. Teknik göstergeler hesapla
        price_with_indicators = calculate_technical_indicators(price_data)
        
        # Teknik göstergelerin eklendiğini doğrula
        assert 'ema_10' in price_with_indicators.columns
        assert 'rsi_14' in price_with_indicators.columns
        
        # 5. Duygu skorlarını fiyat verileriyle birleştir
        sentiment_df_indexed = sentiment_df.set_index('date')
        merged_df = merge_sentiment_with_prices(
            price_with_indicators,
            sentiment_df_indexed
        )
        
        # Birleştirmenin başarılı olduğunu doğrula
        assert 'sentiment_positive' in merged_df.columns
        assert 'sentiment_compound' in merged_df.columns
        assert len(merged_df) > 0
        
        # 6. Eğitim verisi hazırla (küçük lookback)
        lookback = 5
        if len(merged_df) > lookback + 5:
            X, y, scaler = prepare_training_data(merged_df, lookback=lookback)
            
            # Veri boyutlarını doğrula
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == lookback
            assert X.shape[0] > 0
            
            # 7. LSTM modeli oluştur ve eğit (küçük model)
            input_shape = (X.shape[1], X.shape[2])
            lstm_model = build_lstm_model(
                input_shape=input_shape,
                lstm_units=[10, 10],  # Küçük model
                dropout=0.2
            )
            
            # Model yapısını doğrula
            assert lstm_model is not None
            assert len(lstm_model.layers) > 0
            
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            if len(X_train) > 0 and len(X_test) > 0:
                # Hızlı eğitim (sadece birkaç epoch)
                history = train_lstm_model(
                    lstm_model,
                    X_train, y_train,
                    X_test, y_test,
                    epochs=2,  # Çok kısa eğitim
                    batch_size=8
                )
                
                # Eğitim geçmişini doğrula
                assert history is not None
                assert 'loss' in history.history
                
                # 8. LSTM tahmin yap (scaler olmadan, normalize edilmiş değerler)
                predictions_lstm = predict_lstm(lstm_model, X_test, scaler=None)
                
                # Tahminleri doğrula (normalize edilmiş değerler 0-1 aralığında)
                assert len(predictions_lstm) == len(X_test)
                assert all(pred >= -1 for pred in predictions_lstm)  # Makul aralık
                
                # 9. XGBoost için veri hazırla
                X_class, y_class = prepare_classification_data(merged_df)
                
                # Datetime sütunlarını çıkar (XGBoost datetime kabul etmez)
                datetime_cols = X_class.select_dtypes(include=['datetime64']).columns
                if len(datetime_cols) > 0:
                    X_class = X_class.drop(columns=datetime_cols)
                
                if len(X_class) > 10:
                    # Train-test split
                    split_idx_xgb = int(len(X_class) * 0.8)
                    X_train_xgb = X_class.iloc[:split_idx_xgb]
                    y_train_xgb = y_class.iloc[:split_idx_xgb]
                    X_test_xgb = X_class.iloc[split_idx_xgb:]
                    y_test_xgb = y_class.iloc[split_idx_xgb:]
                    
                    # 10. XGBoost modeli eğit
                    xgb_model = train_xgboost_model(
                        X_train_xgb,
                        y_train_xgb,
                        params={'n_estimators': 10, 'max_depth': 3}  # Küçük model
                    )
                    
                    # Model doğrula
                    assert xgb_model is not None
                    
                    # 11. Özellik önem analizi
                    feature_importance = get_feature_importance(xgb_model)
                    assert len(feature_importance) > 0
                    assert 'feature' in feature_importance.columns
                    assert 'importance' in feature_importance.columns
                    
                    # 12. Yön tahmini yap
                    predictions_xgb, probabilities = predict_direction(
                        xgb_model,
                        X_test_xgb
                    )
                    
                    # Tahminleri doğrula
                    assert len(predictions_xgb) == len(X_test_xgb)
                    assert all(pred in [0, 1] for pred in predictions_xgb)
                    assert len(probabilities) == len(X_test_xgb)
                    
                    # 13. Geriye dönük test
                    # Backtest için actual_prices bir eleman daha uzun olmalı
                    # (tahminler bir sonraki günün fiyatını tahmin eder)
                    if len(y_test) > len(predictions_lstm):
                        actual_prices_for_backtest = y_test[:len(predictions_lstm)+1]
                        backtest_results = simple_backtest(
                            predictions_lstm,
                            actual_prices_for_backtest
                        )
                        
                        # Geriye dönük test sonuçlarını doğrula
                        assert 'total_return' in backtest_results
                        assert 'accuracy' in backtest_results
                        assert 'num_trades' in backtest_results
                        assert 'final_capital' in backtest_results
                    
                    # 14. Performans metrikleri
                    # Metrikler için aynı uzunlukta olmalı
                    metrics = calculate_metrics(
                        y_test[:len(predictions_lstm)],
                        predictions_lstm
                    )
                    
                    # Metrikleri doğrula
                    assert 'mse' in metrics
                    assert 'rmse' in metrics
                    assert 'mae' in metrics
                    assert 'r2' in metrics
                    assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_data_persistence_roundtrip(self):
        """
        Veri kaydetme ve yükleme işlemlerinin doğru çalıştığını test eder.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock haber verisi oluştur
            news_data = pd.DataFrame({
                'date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
                'title': ['Haber 1', 'Haber 2', 'Haber 3', 'Haber 4', 'Haber 5'],
                'content': ['İçerik 1', 'İçerik 2', 'İçerik 3', 'İçerik 4', 'İçerik 5'],
                'source': ['Kaynak'] * 5
            })
            
            # CSV formatında kaydet ve yükle
            csv_path = os.path.join(tmpdir, 'test_news.csv')
            save_news(news_data, csv_path, format='csv')
            loaded_csv = load_news(csv_path, format='csv')
            
            assert len(loaded_csv) == len(news_data)
            assert list(loaded_csv.columns) == list(news_data.columns)
            
            # JSON formatında kaydet ve yükle
            json_path = os.path.join(tmpdir, 'test_news.json')
            save_news(news_data, json_path, format='json')
            loaded_json = load_news(json_path, format='json')
            
            assert len(loaded_json) == len(news_data)
            assert list(loaded_json.columns) == list(news_data.columns)
    
    def test_model_persistence_roundtrip(self):
        """
        Model kaydetme ve yükleme işlemlerinin doğru çalıştığını test eder.
        """
        import tensorflow as tf
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Basit LSTM modeli oluştur (en az 2 LSTM katmanı gerekli)
            model = build_lstm_model(
                input_shape=(10, 5),
                lstm_units=[10, 10],  # İki katman gerekli
                dropout=0.2
            )
            
            # Model kaydet
            model_path = os.path.join(tmpdir, 'test_model.keras')
            model.save(model_path)
            
            # Model yükle
            loaded_model = tf.keras.models.load_model(model_path)
            
            # Aynı tahminleri ürettiğini doğrula
            test_input = np.random.random((1, 10, 5))
            original_pred = model.predict(test_input, verbose=0)
            loaded_pred = loaded_model.predict(test_input, verbose=0)
            
            np.testing.assert_allclose(original_pred, loaded_pred, rtol=1e-5)
    
    def test_sentiment_analysis_integration(self):
        """
        Duygu analizi modülünün farklı metinlerle çalıştığını test eder.
        """
        analyzer = SentimentAnalyzer(method='lexicon')
        
        # Farklı duygu içeren metinler
        texts = [
            'Harika bir gelişme! Ekonomi büyüyor ve herkes mutlu.',
            'Kötü haberler geliyor. Piyasalar düşüşte.',
            'Normal bir gün. Özel bir şey yok.',
        ]
        
        results = analyzer.analyze_batch(texts)
        
        # Sonuçları doğrula
        assert len(results) == len(texts)
        assert 'positive' in results.columns
        assert 'negative' in results.columns
        assert 'neutral' in results.columns
        assert 'compound' in results.columns
        
        # İlk metin pozitif olmalı
        assert results.iloc[0]['positive'] > results.iloc[0]['negative']
        
        # İkinci metin negatif olmalı
        assert results.iloc[1]['negative'] > results.iloc[1]['positive']
    
    def test_feature_engineering_pipeline(self):
        """
        Özellik mühendisliği pipeline'ının doğru çalıştığını test eder.
        """
        # Mock fiyat verisi
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        price_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Teknik göstergeler ekle
        with_indicators = calculate_technical_indicators(price_data)
        
        # Tüm göstergelerin eklendiğini doğrula
        expected_indicators = ['ema_10', 'ema_20', 'ema_50', 'rsi_14', 
                              'macd', 'macd_signal', 'volume_ma_20']
        for indicator in expected_indicators:
            assert indicator in with_indicators.columns
        
        # Mock duygu verisi
        sentiment_data = pd.DataFrame({
            'date': dates,
            'positive': np.random.uniform(0, 1, 100),
            'negative': np.random.uniform(0, 1, 100),
            'neutral': np.random.uniform(0, 1, 100),
            'compound': np.random.uniform(-1, 1, 100)
        })
        sentiment_data.set_index('date', inplace=True)
        
        # Birleştir
        merged = merge_sentiment_with_prices(with_indicators, sentiment_data)
        
        # Birleştirmenin başarılı olduğunu doğrula
        assert 'sentiment_positive' in merged.columns
        assert 'sentiment_negative' in merged.columns
        assert 'sentiment_neutral' in merged.columns
        assert 'sentiment_compound' in merged.columns
        assert len(merged) > 0
        
        # Eğitim verisi hazırla
        X, y, scaler = prepare_training_data(merged, lookback=10)
        
        # Boyutları doğrula
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 10
        assert X.shape[0] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
