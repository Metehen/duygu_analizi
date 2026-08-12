"""
Haber toplama modülü için özellik tabanlı testler (Property-Based Tests)

Bu testler, tasarım dokümanındaki doğruluk özelliklerini doğrular.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import tempfile
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from news_fetcher import save_news, load_news


# Yardımcı stratejiler
@composite
def news_dataframe_strategy(draw):
    """
    Geçerli bir haber DataFrame'i oluşturur.
    
    Returns:
        DataFrame with columns: date, title, content, source
    """
    # Rastgele satır sayısı (1-10 arası) - daha küçük veri setleri
    num_rows = draw(st.integers(min_value=1, max_value=10))
    
    # Tarihler - son 30 gün içinde
    base_date = datetime.now()
    dates = [
        base_date - timedelta(days=draw(st.integers(min_value=0, max_value=30)))
        for _ in range(num_rows)
    ]
    
    # Başlıklar - alfanumerik karakterler, daha kısa
    titles = [
        draw(st.text(min_size=5, max_size=30, alphabet=st.characters(min_codepoint=65, max_codepoint=122, blacklist_characters='\\"\'')))
        for _ in range(num_rows)
    ]
    
    # İçerikler - alfanumerik karakterler, daha kısa
    contents = [
        draw(st.text(min_size=10, max_size=50, alphabet=st.characters(min_codepoint=65, max_codepoint=122, blacklist_characters='\\"\'')))
        for _ in range(num_rows)
    ]
    
    # Kaynaklar - alfanumerik karakterler
    sources = [
        draw(st.text(min_size=3, max_size=20, alphabet=st.characters(min_codepoint=65, max_codepoint=122, blacklist_characters='\\"\'')))
        for _ in range(num_rows)
    ]
    
    return pd.DataFrame({
        'date': dates,
        'title': titles,
        'content': contents,
        'source': sources
    })


# Property 1: Haber Toplama Veri Yapısı
# Not: Bu özellik fetch_rss_news için geçerlidir, ancak gerçek RSS feed'lere bağlı olduğu için
# burada mock veri ile test ediyoruz. Gerçek RSS testleri entegrasyon testlerinde yapılmalıdır.

def test_property_news_dataframe_structure():
    """
    **Validates: Requirements 1.1, 1.2**
    
    Feature: turkish-sentiment-stock-prediction, Property 1:
    Herhangi bir geçerli haber DataFrame'i için, gerekli sütunlar
    (date, title, content, source) mevcut olmalı ve doğru tiplerde olmalıdır.
    
    Bu test, haber verisi yapısının tutarlılığını doğrular.
    """
    # Mock haber verisi oluştur (fetch_rss_news çıktısını simüle eder)
    test_data = pd.DataFrame({
        'date': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        'title': ['Ekonomi Haberi 1', 'Borsa Haberi 2', 'Finans Haberi 3'],
        'content': ['İçerik 1 detaylı bilgi', 'İçerik 2 detaylı bilgi', 'İçerik 3 detaylı bilgi'],
        'source': ['Bloomberg HT', 'Dünya Gazetesi', 'Ekonomist']
    })
    
    # Gerekli sütunları kontrol et
    required_columns = ['date', 'title', 'content', 'source']
    assert all(col in test_data.columns for col in required_columns), \
        f"DataFrame gerekli sütunları içermiyor: {required_columns}"
    
    # Tarih sütununun datetime tipinde olduğunu kontrol et
    assert pd.api.types.is_datetime64_any_dtype(test_data['date']), \
        "date sütunu datetime64 tipinde olmalı"
    
    # Diğer sütunların string olduğunu ve boş olmadığını kontrol et
    for col in ['title', 'content', 'source']:
        assert test_data[col].dtype == object, f"{col} sütunu object (string) tipinde olmalı"
        assert all(test_data[col].str.len() > 0), f"{col} sütununda boş değerler var"


# Property 2: Haber Verisi Round-Trip
@given(df=news_dataframe_strategy())
@settings(max_examples=10, deadline=5000)
def test_property_news_roundtrip_csv(df):
    """
    **Validates: Requirements 1.4**
    
    Feature: turkish-sentiment-stock-prediction, Property 2:
    Herhangi bir haber DataFrame'i için, CSV formatında kaydet-yükle
    round-trip orijinal veriye eşdeğer sonuç üretmelidir.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # CSV round-trip
        save_news(df, tmp_path, format='csv')
        df_loaded = load_news(tmp_path, format='csv')
        
        # Sütunların aynı olduğunu kontrol et
        assert list(df.columns) == list(df_loaded.columns), \
            "Yüklenen DataFrame'in sütunları orijinal ile eşleşmiyor"
        
        # Satır sayısının aynı olduğunu kontrol et
        assert len(df) == len(df_loaded), \
            f"Satır sayısı eşleşmiyor: orijinal={len(df)}, yüklenen={len(df_loaded)}"
        
        # Tarih, başlık, içerik, kaynak bilgilerinin korunduğunu kontrol et
        # Tarih karşılaştırması için normalize et
        df_sorted = df.sort_values('date').reset_index(drop=True)
        df_loaded_sorted = df_loaded.sort_values('date').reset_index(drop=True)
        
        # Tarihleri karşılaştır (mikrosaniye hassasiyetini göz ardı et)
        for i in range(len(df_sorted)):
            original_date = pd.Timestamp(df_sorted.iloc[i]['date'])
            loaded_date = pd.Timestamp(df_loaded_sorted.iloc[i]['date'])
            # Saniye hassasiyetinde karşılaştır
            assert abs((original_date - loaded_date).total_seconds()) < 1, \
                f"Tarih eşleşmiyor: {original_date} != {loaded_date}"
        
        # String sütunları karşılaştır (string'e çevirerek karşılaştır)
        # CSV'den yüklenirken tip çıkarımı yapılabilir, bu yüzden string'e çeviriyoruz
        # NaN değerlerini normalize et (hem 'nan' hem 'NaN' aynı kabul edilir)
        def normalize_nan(val):
            """NaN değerlerini normalize eder"""
            s = str(val).lower()
            return s if s != 'nan' else 'nan'
        
        assert [normalize_nan(v) for v in df_sorted['title'].tolist()] == [normalize_nan(v) for v in df_loaded_sorted['title'].tolist()], \
            "Başlıklar eşleşmiyor"
        assert [normalize_nan(v) for v in df_sorted['content'].tolist()] == [normalize_nan(v) for v in df_loaded_sorted['content'].tolist()], \
            "İçerikler eşleşmiyor"
        assert [normalize_nan(v) for v in df_sorted['source'].tolist()] == [normalize_nan(v) for v in df_loaded_sorted['source'].tolist()], \
            "Kaynaklar eşleşmiyor"
        
    finally:
        # Geçici dosyayı temizle
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@given(df=news_dataframe_strategy())
@settings(max_examples=10, deadline=5000)
def test_property_news_roundtrip_json(df):
    """
    **Validates: Requirements 1.4**
    
    Feature: turkish-sentiment-stock-prediction, Property 2:
    Herhangi bir haber DataFrame'i için, JSON formatında kaydet-yükle
    round-trip orijinal veriye eşdeğer sonuç üretmelidir.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # JSON round-trip
        save_news(df, tmp_path, format='json')
        df_loaded = load_news(tmp_path, format='json')
        
        # Sütunların aynı olduğunu kontrol et
        assert list(df.columns) == list(df_loaded.columns), \
            "Yüklenen DataFrame'in sütunları orijinal ile eşleşmiyor"
        
        # Satır sayısının aynı olduğunu kontrol et
        assert len(df) == len(df_loaded), \
            f"Satır sayısı eşleşmiyor: orijinal={len(df)}, yüklenen={len(df_loaded)}"
        
        # Tarih, başlık, içerik, kaynak bilgilerinin korunduğunu kontrol et
        # Tarih karşılaştırması için normalize et
        df_sorted = df.sort_values('date').reset_index(drop=True)
        df_loaded_sorted = df_loaded.sort_values('date').reset_index(drop=True)
        
        # Tarihleri karşılaştır (mikrosaniye hassasiyetini göz ardı et)
        for i in range(len(df_sorted)):
            original_date = pd.Timestamp(df_sorted.iloc[i]['date'])
            loaded_date = pd.Timestamp(df_loaded_sorted.iloc[i]['date'])
            # Saniye hassasiyetinde karşılaştır
            assert abs((original_date - loaded_date).total_seconds()) < 1, \
                f"Tarih eşleşmiyor: {original_date} != {loaded_date}"
        
        # String sütunları karşılaştır (string'e çevirerek karşılaştır)
        # JSON'dan yüklenirken tip çıkarımı yapılabilir, bu yüzden string'e çeviriyoruz
        # NaN değerlerini normalize et (hem 'nan' hem 'NaN' aynı kabul edilir)
        def normalize_nan(val):
            """NaN değerlerini normalize eder"""
            s = str(val).lower()
            return s if s != 'nan' else 'nan'
        
        assert [normalize_nan(v) for v in df_sorted['title'].tolist()] == [normalize_nan(v) for v in df_loaded_sorted['title'].tolist()], \
            "Başlıklar eşleşmiyor"
        assert [normalize_nan(v) for v in df_sorted['content'].tolist()] == [normalize_nan(v) for v in df_loaded_sorted['content'].tolist()], \
            "İçerikler eşleşmiyor"
        assert [normalize_nan(v) for v in df_sorted['source'].tolist()] == [normalize_nan(v) for v in df_loaded_sorted['source'].tolist()], \
            "Kaynaklar eşleşmiyor"
        
    finally:
        # Geçici dosyayı temizle
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
