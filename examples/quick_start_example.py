"""
Hızlı Başlangıç Örneği

Bu script, sistemin temel kullanımını gösterir.
Haber toplama, duygu analizi, model eğitimi ve tahmin adımlarını içerir.
"""

import sys
import os

# Proje kök dizinini Python path'ine ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Modülleri import et
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


def main():
    """Ana fonksiyon - Tam analiz akışı"""
    
    print("=" * 80)
    print("Türkçe Ekonomi Haberleri Duygu Analizi & Hisse Tahmin Sistemi")
    print("Hızlı Başlangıç Örneği")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 1. HABER TOPLAMA
    # ========================================================================
    print("1. Haber Toplama")
    print("-" * 80)
    
    # RSS kaynaklarından haber topla
    rss_urls = [
        'https://www.bloomberght.com/rss',
    ]
    
    print(f"RSS kaynaklarından haber toplanıyor: {len(rss_urls)} kaynak")
    news_df = fetch_rss_news(rss_urls, max_articles=50)
    print(f"✓ Toplam {len(news_df)} haber toplandı")
    print()
    
    # Haberleri kaydet
    save_news(news_df, 'data/news_example.csv', format='csv')
    print("✓ Haberler kaydedildi: data/news_example.csv")
    print()
    
    # ========================================================================
    # 2. DUYGU ANALİZİ
    # ========================================================================
    print("2. Duygu Analizi")
    print("-" * 80)
    
    # Duygu analizi yap (lexicon yöntemi - hızlı)
    print("Duygu analizi yapılıyor (lexicon yöntemi)...")
    analyzer = SentimentAnalyzer(method='lexicon')
    
    # Batch analiz
    sentiment_scores = []
    for content in news_df['content']:
        score = analyzer.analyze(content)
        sentiment_scores.append(score)
    
    sentiment_df = pd.DataFrame(sentiment_scores)
    sentiment_df['date'] = news_df['date'].values
    
    print(f"✓ {len(sentiment_df)} haber için duygu analizi tamamlandı")
    print(f"  Ortalama compound skoru: {sentiment_df['compound'].mean():.3f}")
    print(f"  Pozitif haber oranı: {(sentiment_df['compound'] > 0).sum() / len(sentiment_df) * 100:.1f}%")
    print()
    
    # ========================================================================
    # 3. HİSSE VERİSİ ÇEKME
    # ========================================================================
    print("3. Hisse Verisi Çekme")
    print("-" * 80)
    
    # Hisse sembolü ve tarih aralığı
    symbol = 'THYAO.IS'  # Türk Hava Yolları
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # Son 6 ay
    
    print(f"Hisse verisi çekiliyor: {symbol}")
    print(f"Tarih aralığı: {start_date.date()} - {end_date.date()}")
    
    stock_df = fetch_stock_data(
        symbol,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    print(f"✓ {len(stock_df)} günlük hisse verisi çekildi")
    print(f"  Fiyat aralığı: {stock_df['close'].min():.2f} - {stock_df['close'].max():.2f} TL")
    print()
    
    # ========================================================================
    # 4. TEKNİK GÖSTERGELER
    # ========================================================================
    print("4. Teknik Göstergeler")
    print("-" * 80)
    
    print("Teknik göstergeler hesaplanıyor...")
    stock_df = calculate_technical_indicators(stock_df)
    
    print("✓ Teknik göstergeler eklendi:")
    print("  - EMA (10, 20, 50)")
    print("  - RSI (14)")
    print("  - MACD")
    print("  - Volume MA (20)")
    print()
    
    # ========================================================================
    # 5. VERİ BİRLEŞTİRME
    # ========================================================================
    print("5. Veri Birleştirme")
    print("-" * 80)
    
    print("Duygu skorları ve hisse verileri birleştiriliyor...")
    sentiment_df_indexed = sentiment_df.set_index('date')
    merged_df = merge_sentiment_with_prices(stock_df, sentiment_df_indexed)
    
    print(f"✓ Birleştirilmiş veri: {len(merged_df)} satır, {len(merged_df.columns)} sütun")
    print()
    
    # ========================================================================
    # 6. EĞİTİM VERİSİ HAZIRLA
    # ========================================================================
    print("6. Eğitim Verisi Hazırlama")
    print("-" * 80)
    
    lookback = 30  # 30 günlük pencere
    print(f"Eğitim verisi hazırlanıyor (lookback={lookback})...")
    
    X, y, scaler = prepare_training_data(merged_df, lookback=lookback)
    
    print(f"✓ Eğitim verisi hazır:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print()
    
    # Train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  Eğitim seti: {len(X_train)} örnek")
    print(f"  Test seti: {len(X_test)} örnek")
    print()
    
    # ========================================================================
    # 7. LSTM MODELİ
    # ========================================================================
    print("7. LSTM Model Eğitimi")
    print("-" * 80)
    
    print("LSTM modeli oluşturuluyor...")
    lstm_model = build_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        lstm_units=[25, 25],  # Küçük model (hızlı eğitim)
        dropout=0.2
    )
    
    print("✓ Model oluşturuldu")
    print(f"  Toplam parametre: {lstm_model.count_params():,}")
    print()
    
    print("Model eğitiliyor (5 epoch - demo için kısa)...")
    history = train_lstm_model(
        lstm_model,
        X_train, y_train,
        X_test, y_test,
        epochs=5,  # Demo için kısa
        batch_size=16
    )
    
    print("✓ Model eğitimi tamamlandı")
    print(f"  Final loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final val_loss: {history.history['val_loss'][-1]:.4f}")
    print()
    
    # Tahmin yap
    print("LSTM tahminleri yapılıyor...")
    predictions_lstm = predict_lstm(lstm_model, X_test, scaler=None)
    
    print(f"✓ {len(predictions_lstm)} tahmin yapıldı")
    print()
    
    # ========================================================================
    # 8. XGBOOST MODELİ
    # ========================================================================
    print("8. XGBoost Model Eğitimi")
    print("-" * 80)
    
    print("XGBoost için veri hazırlanıyor...")
    X_class, y_class = prepare_classification_data(merged_df)
    
    # Datetime sütunlarını çıkar
    datetime_cols = X_class.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        X_class = X_class.drop(columns=datetime_cols)
    
    # Train-test split
    split_idx_xgb = int(len(X_class) * 0.8)
    X_train_xgb = X_class.iloc[:split_idx_xgb]
    y_train_xgb = y_class.iloc[:split_idx_xgb]
    X_test_xgb = X_class.iloc[split_idx_xgb:]
    y_test_xgb = y_class.iloc[split_idx_xgb:]
    
    print(f"  Eğitim seti: {len(X_train_xgb)} örnek")
    print(f"  Test seti: {len(X_test_xgb)} örnek")
    print()
    
    print("XGBoost modeli eğitiliyor...")
    xgb_model = train_xgboost_model(
        X_train_xgb,
        y_train_xgb,
        params={'n_estimators': 50, 'max_depth': 4}  # Küçük model
    )
    
    print("✓ Model eğitimi tamamlandı")
    print()
    
    # Özellik önem analizi
    print("Özellik önem analizi:")
    importance_df = get_feature_importance(xgb_model)
    print(importance_df.head(5).to_string(index=False))
    print()
    
    # Tahmin yap
    predictions_xgb, probabilities = predict_direction(xgb_model, X_test_xgb)
    
    accuracy_xgb = (predictions_xgb == y_test_xgb.values).sum() / len(y_test_xgb) * 100
    print(f"✓ XGBoost Test Doğruluğu: {accuracy_xgb:.2f}%")
    print()
    
    # ========================================================================
    # 9. PERFORMANS DEĞERLENDİRME
    # ========================================================================
    print("9. Performans Değerlendirme")
    print("-" * 80)
    
    # LSTM metrikleri
    print("LSTM Performans Metrikleri:")
    metrics_lstm = calculate_metrics(y_test[:len(predictions_lstm)], predictions_lstm)
    print(f"  MSE:  {metrics_lstm['mse']:.4f}")
    print(f"  RMSE: {metrics_lstm['rmse']:.4f}")
    print(f"  MAE:  {metrics_lstm['mae']:.4f}")
    print(f"  R2:   {metrics_lstm['r2']:.4f}")
    print()
    
    # XGBoost metrikleri
    print("XGBoost Performans Metrikleri:")
    print(f"  Doğruluk: {accuracy_xgb:.2f}%")
    print(f"  Yukarı tahmin: {predictions_xgb.sum()} / {len(predictions_xgb)}")
    print(f"  Aşağı tahmin: {len(predictions_xgb) - predictions_xgb.sum()} / {len(predictions_xgb)}")
    print()
    
    # ========================================================================
    # 10. ÖZET
    # ========================================================================
    print("=" * 80)
    print("ÖZET")
    print("=" * 80)
    print(f"✓ {len(news_df)} haber toplandı ve analiz edildi")
    print(f"✓ {len(stock_df)} günlük hisse verisi işlendi")
    print(f"✓ LSTM modeli eğitildi (RMSE: {metrics_lstm['rmse']:.4f})")
    print(f"✓ XGBoost modeli eğitildi (Doğruluk: {accuracy_xgb:.2f}%)")
    print()
    print("Sistem başarıyla çalıştı!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nİşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n\nHata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
