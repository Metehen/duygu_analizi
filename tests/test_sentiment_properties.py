"""
Duygu analizi modülü için özellik tabanlı testler (Property-Based Tests)

Bu testler, tasarım dokümanındaki doğruluk özelliklerini doğrular.
"""

import pytest
import pandas as pd
import os
import sys
from hypothesis import given, strategies as st, settings

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment import SentimentAnalyzer


# Property 3: Duygu Analizi Çıktı Yapısı
@given(st.text(min_size=10, max_size=1000))
@settings(max_examples=100, deadline=10000)
def test_property_sentiment_output_structure_single(text):
    """
    **Validates: Requirements 2.2, 2.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 3:
    Herhangi bir metin için, duygu analizi gerekli sütunları
    ve geçerli değer aralıklarını içermelidir.
    
    Bu test, tek metin için analyze() fonksiyonunu test eder.
    """
    # Sözlük tabanlı yöntem kullan (BERT modeli yükleme sorunlarını önlemek için)
    analyzer = SentimentAnalyzer(method='lexicon')
    result = analyzer.analyze(text)
    
    # Gerekli anahtarları kontrol et
    required_keys = ['positive', 'negative', 'neutral', 'compound']
    assert all(key in result for key in required_keys), \
        f"Sonuç dictionary'si gerekli anahtarları içermiyor: {required_keys}"
    
    # Değer aralıklarını kontrol et
    assert 0.0 <= result['positive'] <= 1.0, \
        f"positive değeri 0.0-1.0 aralığında olmalı, alınan: {result['positive']}"
    assert 0.0 <= result['negative'] <= 1.0, \
        f"negative değeri 0.0-1.0 aralığında olmalı, alınan: {result['negative']}"
    assert 0.0 <= result['neutral'] <= 1.0, \
        f"neutral değeri 0.0-1.0 aralığında olmalı, alınan: {result['neutral']}"
    assert -1.0 <= result['compound'] <= 1.0, \
        f"compound değeri -1.0-1.0 aralığında olmalı, alınan: {result['compound']}"
    
    # Tüm değerlerin numeric olduğunu kontrol et
    assert isinstance(result['positive'], (int, float)), \
        f"positive değeri numeric olmalı, alınan tip: {type(result['positive'])}"
    assert isinstance(result['negative'], (int, float)), \
        f"negative değeri numeric olmalı, alınan tip: {type(result['negative'])}"
    assert isinstance(result['neutral'], (int, float)), \
        f"neutral değeri numeric olmalı, alınan tip: {type(result['neutral'])}"
    assert isinstance(result['compound'], (int, float)), \
        f"compound değeri numeric olmalı, alınan tip: {type(result['compound'])}"


@given(st.lists(st.text(min_size=10, max_size=500), min_size=1, max_size=20))
@settings(max_examples=100, deadline=10000)
def test_property_sentiment_output_structure_batch(texts):
    """
    **Validates: Requirements 2.2, 2.3**
    
    Feature: turkish-sentiment-stock-prediction, Property 3:
    Herhangi bir metin listesi için, duygu analizi gerekli sütunları
    ve geçerli değer aralıklarını içermelidir.
    
    Bu test, metin listesi için analyze_batch() fonksiyonunu test eder.
    """
    # Sözlük tabanlı yöntem kullan (BERT modeli yükleme sorunlarını önlemek için)
    analyzer = SentimentAnalyzer(method='lexicon')
    result_df = analyzer.analyze_batch(texts)
    
    # DataFrame olduğunu kontrol et
    assert isinstance(result_df, pd.DataFrame), \
        f"Sonuç bir DataFrame olmalı, alınan tip: {type(result_df)}"
    
    # Gerekli sütunları kontrol et
    required_columns = ['positive', 'negative', 'neutral', 'compound']
    assert all(col in result_df.columns for col in required_columns), \
        f"DataFrame gerekli sütunları içermiyor: {required_columns}"
    
    # Satır sayısının girdi ile eşleştiğini kontrol et
    assert len(result_df) == len(texts), \
        f"DataFrame satır sayısı girdi ile eşleşmiyor: beklenen={len(texts)}, alınan={len(result_df)}"
    
    # Her satır için değer aralıklarını kontrol et
    for idx, row in result_df.iterrows():
        assert 0.0 <= row['positive'] <= 1.0, \
            f"Satır {idx}: positive değeri 0.0-1.0 aralığında olmalı, alınan: {row['positive']}"
        assert 0.0 <= row['negative'] <= 1.0, \
            f"Satır {idx}: negative değeri 0.0-1.0 aralığında olmalı, alınan: {row['negative']}"
        assert 0.0 <= row['neutral'] <= 1.0, \
            f"Satır {idx}: neutral değeri 0.0-1.0 aralığında olmalı, alınan: {row['neutral']}"
        assert -1.0 <= row['compound'] <= 1.0, \
            f"Satır {idx}: compound değeri -1.0-1.0 aralığında olmalı, alınan: {row['compound']}"
    
    # Tüm sütunların numeric olduğunu kontrol et
    assert pd.api.types.is_numeric_dtype(result_df['positive']), \
        "positive sütunu numeric olmalı"
    assert pd.api.types.is_numeric_dtype(result_df['negative']), \
        "negative sütunu numeric olmalı"
    assert pd.api.types.is_numeric_dtype(result_df['neutral']), \
        "neutral sütunu numeric olmalı"
    assert pd.api.types.is_numeric_dtype(result_df['compound']), \
        "compound sütunu numeric olmalı"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
