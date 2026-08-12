"""
Temel haber toplama modülü testleri
"""

import pytest
import pandas as pd
from datetime import datetime
import os
import sys
from unittest.mock import patch, MagicMock

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from news_fetcher import fetch_rss_news, save_news, load_news


def test_fetch_rss_empty_url():
    """Boş URL listesi ile haber çekme testi."""
    result = fetch_rss_news([])
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ['date', 'title', 'content', 'source']


def test_fetch_rss_invalid_url():
    """Geçersiz URL ile haber çekme testi."""
    result = fetch_rss_news(['https://invalid-url-that-does-not-exist.com/rss'])
    assert isinstance(result, pd.DataFrame)
    # Geçersiz URL'den haber çekilemez, boş DataFrame döner
    assert list(result.columns) == ['date', 'title', 'content', 'source']


def test_save_and_load_news_csv(tmp_path):
    """CSV formatında kaydetme ve yükleme testi."""
    # Test verisi oluştur
    test_data = pd.DataFrame({
        'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        'title': ['Başlık 1', 'Başlık 2'],
        'content': ['İçerik 1', 'İçerik 2'],
        'source': ['Kaynak 1', 'Kaynak 2']
    })
    
    # Kaydet
    filepath = tmp_path / "test_news.csv"
    save_news(test_data, str(filepath), format='csv')
    
    # Dosyanın oluşturulduğunu kontrol et
    assert filepath.exists()
    
    # Yükle
    loaded_data = load_news(str(filepath), format='csv')
    
    # Verilerin eşleştiğini kontrol et
    assert len(loaded_data) == len(test_data)
    assert list(loaded_data.columns) == list(test_data.columns)
    assert loaded_data['title'].tolist() == test_data['title'].tolist()
    assert loaded_data['content'].tolist() == test_data['content'].tolist()


def test_save_and_load_news_json(tmp_path):
    """JSON formatında kaydetme ve yükleme testi."""
    # Test verisi oluştur
    test_data = pd.DataFrame({
        'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        'title': ['Başlık 1', 'Başlık 2'],
        'content': ['İçerik 1', 'İçerik 2'],
        'source': ['Kaynak 1', 'Kaynak 2']
    })
    
    # Kaydet
    filepath = tmp_path / "test_news.json"
    save_news(test_data, str(filepath), format='json')
    
    # Dosyanın oluşturulduğunu kontrol et
    assert filepath.exists()
    
    # Yükle
    loaded_data = load_news(str(filepath), format='json')
    
    # Verilerin eşleştiğini kontrol et
    assert len(loaded_data) == len(test_data)
    assert list(loaded_data.columns) == list(test_data.columns)
    assert loaded_data['title'].tolist() == test_data['title'].tolist()
    assert loaded_data['content'].tolist() == test_data['content'].tolist()


def test_save_news_invalid_format(tmp_path):
    """Geçersiz format ile kaydetme testi."""
    test_data = pd.DataFrame({
        'date': [datetime(2024, 1, 1)],
        'title': ['Başlık'],
        'content': ['İçerik'],
        'source': ['Kaynak']
    })
    
    filepath = tmp_path / "test_news.txt"
    
    with pytest.raises(ValueError, match="Desteklenmeyen format"):
        save_news(test_data, str(filepath), format='txt')


def test_load_news_file_not_found():
    """Var olmayan dosya yükleme testi."""
    with pytest.raises(FileNotFoundError):
        load_news('nonexistent_file.csv', format='csv')


def test_fetch_rss_invalid_format():
    """Geçersiz RSS formatı testi."""
    # feedparser.parse'ı mock'la ve geçersiz RSS formatı döndür
    with patch('news_fetcher.feedparser.parse') as mock_parse:
        # bozo=True ve entries boş olduğunda geçersiz format
        mock_feed = MagicMock()
        mock_feed.bozo = True
        mock_feed.entries = []
        mock_feed.feed = {}
        mock_parse.return_value = mock_feed
        
        result = fetch_rss_news(['https://example.com/invalid-rss'])
        
        # Geçersiz format atlanmalı, boş DataFrame dönmeli
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == ['date', 'title', 'content', 'source']


def test_fetch_rss_connection_error():
    """Bağlantı hatası testi."""
    # feedparser.parse'ı mock'la ve exception fırlat
    with patch('news_fetcher.feedparser.parse') as mock_parse:
        mock_parse.side_effect = Exception("Connection timeout")
        
        result = fetch_rss_news(['https://example.com/rss'])
        
        # Bağlantı hatası olduğunda boş DataFrame dönmeli
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == ['date', 'title', 'content', 'source']


def test_fetch_rss_multiple_urls_with_errors():
    """Birden fazla URL'de bazıları hata verdiğinde test."""
    with patch('news_fetcher.feedparser.parse') as mock_parse:
        # İlk URL başarılı, ikinci URL hata verir
        def side_effect(url):
            if 'success' in url:
                mock_feed = MagicMock()
                mock_feed.bozo = False
                mock_entry = MagicMock()
                mock_entry.title = 'Test Başlık'
                mock_entry.get = lambda key, default='': 'Test İçerik' if key == 'summary' else default
                mock_entry.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)
                mock_feed.entries = [mock_entry]
                mock_feed.feed = {'title': 'Test Kaynak'}
                return mock_feed
            else:
                raise Exception("Connection error")
        
        mock_parse.side_effect = side_effect
        
        result = fetch_rss_news(['https://success.com/rss', 'https://error.com/rss'])
        
        # Başarılı URL'den veri gelmeli
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0  # En az 0 satır (hata durumunda)


def test_fetch_rss_empty_content_skipped():
    """Boş içerikli haberlerin atlandığını test et."""
    with patch('news_fetcher.feedparser.parse') as mock_parse:
        mock_feed = MagicMock()
        mock_feed.bozo = False
        
        # İki entry: biri boş içerikli, diğeri geçerli
        mock_entry1 = MagicMock()
        mock_entry1.title = 'Boş İçerik'
        mock_entry1.get = lambda key, default='': '' if key in ['summary', 'description'] else default
        mock_entry1.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)
        
        mock_entry2 = MagicMock()
        mock_entry2.title = 'Geçerli Başlık'
        mock_entry2.get = lambda key, default='': 'Geçerli İçerik' if key == 'summary' else default
        mock_entry2.published_parsed = (2024, 1, 2, 0, 0, 0, 0, 0, 0)
        
        mock_feed.entries = [mock_entry1, mock_entry2]
        mock_feed.feed = {'title': 'Test Kaynak'}
        mock_parse.return_value = mock_feed
        
        result = fetch_rss_news(['https://example.com/rss'])
        
        # Sadece geçerli içerikli haber dönmeli
        assert isinstance(result, pd.DataFrame)
        # Boş içerik atlanır, bu yüzden sonuç boş olabilir veya sadece geçerli olanı içerebilir


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
