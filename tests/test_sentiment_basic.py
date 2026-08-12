"""
Temel duygu analizi modülü testleri
"""

import pytest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment import SentimentAnalyzer


def test_analyze_empty_text():
    """Boş metin testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    # Boş string
    result = analyzer.analyze('')
    assert isinstance(result, dict)
    assert 'positive' in result
    assert 'negative' in result
    assert 'neutral' in result
    assert 'compound' in result
    assert result['neutral'] == 1.0
    assert result['positive'] == 0.0
    assert result['negative'] == 0.0
    assert result['compound'] == 0.0
    
    # Sadece boşluk karakterleri
    result = analyzer.analyze('   ')
    assert result['neutral'] == 1.0
    assert result['compound'] == 0.0
    
    # None değeri (string olmayan)
    result = analyzer.analyze(None)
    assert result['neutral'] == 1.0
    assert result['compound'] == 0.0


def test_analyze_very_long_text():
    """Çok uzun metin testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    # 10000 kelimelik uzun metin oluştur
    long_text = ' '.join(['iyi' if i % 2 == 0 else 'kötü' for i in range(10000)])
    
    result = analyzer.analyze(long_text)
    
    # Sonuç yapısını kontrol et
    assert isinstance(result, dict)
    assert 'positive' in result
    assert 'negative' in result
    assert 'neutral' in result
    assert 'compound' in result
    
    # Değer aralıklarını kontrol et
    assert 0.0 <= result['positive'] <= 1.0
    assert 0.0 <= result['negative'] <= 1.0
    assert 0.0 <= result['neutral'] <= 1.0
    assert -1.0 <= result['compound'] <= 1.0
    
    # Uzun metinde pozitif ve negatif kelimeler dengeli olmalı
    assert result['positive'] > 0.0
    assert result['negative'] > 0.0


def test_analyze_special_characters():
    """Özel karakterler testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    # Özel karakterler içeren metin
    special_texts = [
        'İyi! Harika... Mükemmel!!!',
        'Kötü??? Berbat... Başarısız!!!',
        '@#$%^&*()_+-=[]{}|;:,.<>?',
        'İyi 😊 harika 🎉 mükemmel 👍',
        'Türkçe karakterler: ğüşıöçĞÜŞİÖÇ',
        'Sayılar: 123 456 789',
        'Email: test@example.com',
        'URL: https://www.example.com',
        'Karışık: İyi! 123 @test kötü... #hashtag'
    ]
    
    for text in special_texts:
        result = analyzer.analyze(text)
        
        # Sonuç yapısını kontrol et
        assert isinstance(result, dict)
        assert 'positive' in result
        assert 'negative' in result
        assert 'neutral' in result
        assert 'compound' in result
        
        # Değer aralıklarını kontrol et
        assert 0.0 <= result['positive'] <= 1.0
        assert 0.0 <= result['negative'] <= 1.0
        assert 0.0 <= result['neutral'] <= 1.0
        assert -1.0 <= result['compound'] <= 1.0


def test_model_loading_error_fallback():
    """Model yükleme hatası testi - sözlük tabanlı yönteme geri dönüş."""
    # BERT modeli yüklenirken hata oluştuğunda lexicon'a geri dönmeli
    with patch('sentiment.SentimentAnalyzer._load_bert_model') as mock_load:
        mock_load.side_effect = Exception("Model yükleme hatası")
        
        # BERT ile başlatmaya çalış
        analyzer = SentimentAnalyzer(method='bert')
        
        # Lexicon'a geri dönmüş olmalı
        assert analyzer.method == 'lexicon'
        
        # Analiz çalışmalı
        result = analyzer.analyze('İyi bir haber')
        assert isinstance(result, dict)
        assert 'positive' in result


def test_model_loading_import_error():
    """transformers kütüphanesi yoksa ImportError fırlatmalı."""
    with patch('sentiment.SentimentAnalyzer._load_bert_model') as mock_load:
        mock_load.side_effect = ImportError("transformers kütüphanesi bulunamadı")
        
        # BERT ile başlatmaya çalış
        analyzer = SentimentAnalyzer(method='bert')
        
        # Lexicon'a geri dönmüş olmalı
        assert analyzer.method == 'lexicon'


def test_analyze_batch_empty_list():
    """Boş liste ile batch analizi testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    result = analyzer.analyze_batch([])
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    # Boş DataFrame'de sütunlar olmayabilir veya olabilir
    if len(result.columns) > 0:
        assert list(result.columns) == ['positive', 'negative', 'neutral', 'compound']


def test_analyze_batch_single_text():
    """Tek metin ile batch analizi testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    result = analyzer.analyze_batch(['iyi bir haber'])
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert list(result.columns) == ['positive', 'negative', 'neutral', 'compound']
    assert result.iloc[0]['positive'] > 0.0


def test_analyze_batch_multiple_texts():
    """Birden fazla metin ile batch analizi testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    texts = [
        'iyi bir haber',
        'kötü bir durum',
        'nötr bir açıklama'
    ]
    
    result = analyzer.analyze_batch(texts)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(texts)
    assert list(result.columns) == ['positive', 'negative', 'neutral', 'compound']
    
    # İlk metin pozitif olmalı
    assert result.iloc[0]['positive'] > result.iloc[0]['negative']
    
    # İkinci metin negatif olmalı
    assert result.iloc[1]['negative'] > result.iloc[1]['positive']


def test_analyze_exception_handling():
    """Analiz sırasında exception durumunda nötr skor dönmeli."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    # _analyze_lexicon metodunu mock'la ve exception fırlat
    with patch.object(analyzer, '_analyze_lexicon') as mock_analyze:
        mock_analyze.side_effect = Exception("Beklenmeyen hata")
        
        result = analyzer.analyze('Test metni')
        
        # Hata durumunda nötr skor dönmeli
        assert result['neutral'] == 1.0
        assert result['positive'] == 0.0
        assert result['negative'] == 0.0
        assert result['compound'] == 0.0


def test_lexicon_positive_text():
    """Pozitif kelimeler içeren metin testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    positive_text = 'İyi harika mükemmel başarılı kazanç artış yükseliş büyüme'
    result = analyzer.analyze(positive_text)
    
    # Pozitif skor yüksek olmalı
    assert result['positive'] > 0.0
    assert result['compound'] > 0.0


def test_lexicon_negative_text():
    """Negatif kelimeler içeren metin testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    negative_text = 'Kötü berbat başarısız kayıp düşüş azalma gerileme olumsuz'
    result = analyzer.analyze(negative_text)
    
    # Negatif skor yüksek olmalı
    assert result['negative'] > 0.0
    assert result['compound'] < 0.0


def test_lexicon_neutral_text():
    """Nötr metin testi (duygu kelimesi yok)."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    neutral_text = 'Şirket bugün toplantı yaptı rapor sunuldu veriler incelendi'
    result = analyzer.analyze(neutral_text)
    
    # Nötr skor yüksek olmalı
    assert result['neutral'] == 1.0
    assert result['positive'] == 0.0
    assert result['negative'] == 0.0


def test_lexicon_mixed_sentiment():
    """Karışık duygu içeren metin testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    mixed_text = 'iyi haberler var ama kötü sonuçlar da mevcut'
    result = analyzer.analyze(mixed_text)
    
    # Hem pozitif hem negatif skor olmalı
    assert result['positive'] > 0.0
    assert result['negative'] > 0.0


def test_analyze_non_string_input():
    """String olmayan girdi testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    # Integer
    result = analyzer.analyze(123)
    assert result['neutral'] == 1.0
    
    # List
    result = analyzer.analyze(['test'])
    assert result['neutral'] == 1.0
    
    # Dict
    result = analyzer.analyze({'key': 'value'})
    assert result['neutral'] == 1.0


def test_bert_method_initialization():
    """BERT metodu ile başlatma testi (transformers yoksa lexicon'a düşer)."""
    try:
        analyzer = SentimentAnalyzer(method='bert')
        # BERT yüklendiyse veya lexicon'a düştüyse, method set edilmiş olmalı
        assert analyzer.method in ['bert', 'lexicon']
    except ImportError:
        # transformers yoksa bu beklenen bir durum
        pytest.skip("transformers kütüphanesi yüklü değil")


def test_lexicon_method_initialization():
    """Lexicon metodu ile başlatma testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    assert analyzer.method == 'lexicon'
    assert hasattr(analyzer, 'positive_words')
    assert hasattr(analyzer, 'negative_words')
    assert len(analyzer.positive_words) > 0
    assert len(analyzer.negative_words) > 0


def test_case_insensitivity():
    """Büyük/küçük harf duyarsızlığı testi."""
    analyzer = SentimentAnalyzer(method='lexicon')
    
    # Aynı metin farklı case'lerde
    text_lower = 'iyi harika mükemmel'
    text_upper = 'İYİ HARİKA MÜKEMMEL'
    text_mixed = 'İyi Harika Mükemmel'
    
    result_lower = analyzer.analyze(text_lower)
    result_upper = analyzer.analyze(text_upper)
    result_mixed = analyzer.analyze(text_mixed)
    
    # Sonuçlar benzer olmalı (tam eşit olmayabilir ama pozitif olmalı)
    assert result_lower['positive'] > 0.0
    assert result_upper['positive'] > 0.0
    assert result_mixed['positive'] > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
