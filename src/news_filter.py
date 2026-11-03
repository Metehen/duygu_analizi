# Copyright (C) 2026  Metehen
# Licensed under GNU AGPL v3. See LICENSE file for details.
"""
Haber Filtreleme Modülü
Haberleri hisse senedi ile ilişkisine göre filtreler.
"""

import pandas as pd
from typing import List, Dict
from src.logger import get_logger

logger = get_logger(__name__)

# Hisse senedi için anahtar kelime haritası
STOCK_KEYWORDS = {
    'THYAO': [
        'türk hava yolları', 'thy', 'turkish airlines', 'havayolu', 'uçak', 'havacılık',
        'türk hava', 'hava yolları', 'thy a.o', 'thy a.ş', 'thy hisse',
        'uçuş', 'pilot', 'hostes', 'yolcu', 'kargo', 'boeing', 'airbus',
        'istanbul havalimanı', 'sabiha gökçen', 'atatürk havalimanı',
        'bilet', 'rezervasyon', 'miles', 'miles&smiles'
    ],
    'GARAN': ['garanti', 'garanti bankası', 'bbva', 'garanti bbva', 'garanti bank'],
    'ISCTR': ['iş bankası', 'işbank', 'türkiye iş bankası', 'iş bank'],
    'AKBNK': ['akbank', 'sabancı', 'akbank t.a.ş'],
    'TUPRS': ['tüpraş', 'türkiye petrol', 'rafineri', 'petrol', 'tüpraş a.ş'],
    'ASELS': ['aselsan', 'savunma', 'askeri', 'elektronik', 'aselsan a.ş'],
    'EREGL': ['ereğli', 'erdemir', 'demir çelik', 'ereğli demir'],
    'KCHOL': ['koç holding', 'koç', 'holding', 'koç ailesi'],
    'SAHOL': ['sabancı holding', 'sabancı', 'sabancı ailesi'],
    'PETKM': ['petkim', 'petrokimya', 'petkim a.ş'],
    'SISE': ['şişe cam', 'şişecam', 'cam', 'şişe cam a.ş'],
    'TCELL': ['turkcell', 'türk telekom', 'telekomünikasyon', 'gsm', 'mobil'],
    'TTKOM': ['türk telekom', 'telekom', 'tt', 'türk telekom a.ş'],
    'VAKBN': ['vakıfbank', 'vakıf bankası', 'vakıf bank'],
    'YKBNK': ['yapı kredi', 'yapı ve kredi', 'yapı kredi bankası'],
    'BIMAS': ['bim', 'bim market', 'bim mağazaları', 'market'],
    'MGROS': ['migros', 'migros ticaret', 'süpermarket'],
    'TOASO': ['tofaş', 'fiat', 'otomotiv', 'tofaş a.ş'],
    'FROTO': ['ford otosan', 'ford', 'otomotiv', 'ticari araç'],
}

# Sektör bazlı genel kelimeler
SECTOR_KEYWORDS = {
    'banking': [
        'banka', 'kredi', 'mevduat', 'faiz', 'finansal', 'bankacılık',
        'tüketici kredisi', 'konut kredisi', 'ticari kredi', 'kart', 'pos',
        'dijital bankacılık', 'internet bankacılığı', 'mobil bankacılık'
    ],
    'airline': [
        'havayolu', 'uçak', 'havacılık', 'yolcu', 'kargo', 'uçuş',
        'havalimanı', 'pilot', 'hostes', 'bilet', 'rezervasyon',
        'boeing', 'airbus', 'sefer', 'rota', 'destinasyon',
        'iata', 'icao', 'sivil havacılık'
    ],
    'energy': [
        'enerji', 'petrol', 'doğalgaz', 'elektrik', 'rafineri',
        'petrokimya', 'akaryakıt', 'benzin', 'motorin', 'lpg',
        'ham petrol', 'brent', 'opec', 'enerji piyasası'
    ],
    'technology': [
        'teknoloji', 'yazılım', 'dijital', 'telekom', 'telekomünikasyon',
        'internet', 'mobil', 'gsm', '5g', '4.5g', 'fiber',
        'veri', 'bulut', 'yapay zeka', 'iot', 'akıllı'
    ],
    'manufacturing': [
        'üretim', 'sanayi', 'fabrika', 'imalat', 'montaj',
        'otomotiv', 'araç', 'ihracat', 'ithalat', 'kapasite'
    ],
    'retail': [
        'perakende', 'market', 'mağaza', 'satış', 'tüketici',
        'alışveriş', 'zincir', 'süpermarket', 'hipermarket', 'e-ticaret'
    ],
    'defense': [
        'savunma', 'askeri', 'silah', 'füze', 'radar', 'elektronik',
        'milli savunma', 'ihale', 'proje', 'teknoloji geliştirme'
    ],
}


def get_stock_info(symbol: str) -> Dict[str, any]:
    """
    Hisse senedi bilgilerini döndürür.
    
    Args:
        symbol: Hisse sembolü (örn: 'THYAO.IS')
    
    Returns:
        Dictionary: {'code': 'THYAO', 'keywords': [...], 'sector': '...'}
    """
    # .IS uzantısını kaldır
    code = symbol.replace('.IS', '').upper()
    
    # Anahtar kelimeleri al
    keywords = STOCK_KEYWORDS.get(code, [])
    
    # Sektör belirle (basit mantık)
    sector = None
    if code in ['GARAN', 'ISCTR', 'AKBNK', 'VAKBN', 'YKBNK']:
        sector = 'banking'
    elif code in ['THYAO']:
        sector = 'airline'
    elif code in ['TUPRS', 'PETKM']:
        sector = 'energy'
    elif code in ['TCELL', 'TTKOM']:
        sector = 'technology'
    elif code in ['TOASO', 'FROTO']:
        sector = 'manufacturing'
    elif code in ['BIMAS', 'MGROS']:
        sector = 'retail'
    elif code in ['ASELS']:
        sector = 'defense'
    
    return {
        'code': code,
        'keywords': keywords,
        'sector': sector
    }


def calculate_relevance_score(text: str, keywords: List[str]) -> float:
    """
    Metnin anahtar kelimelerle ilgili olma skorunu hesaplar.
    
    Args:
        text: Analiz edilecek metin
        keywords: Anahtar kelime listesi
    
    Returns:
        Relevance score (0.0 - 1.0)
    """
    if not text or not keywords:
        return 0.0
    
    text_lower = text.lower()
    matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
    
    # Normalize et (en fazla 3 eşleşme = 1.0 skor)
    score = min(matches / 3.0, 1.0)
    
    return score


def filter_news_by_stock(
    news_df: pd.DataFrame,
    symbol: str,
    min_relevance: float = 0.3,
    include_sector_news: bool = True
) -> pd.DataFrame:
    """
    Haberleri hisse senedi ile ilişkisine göre filtreler.

    Args:
        news_df: Haber DataFrame'i (date, title, content, source)
        symbol: Hisse sembolü (örn: 'THYAO.IS' veya 'THYAO')
        min_relevance: Minimum ilişki skoru (0.0 - 1.0)
        include_sector_news: Sektör haberlerini de dahil et

    Returns:
        Filtrelenmiş haber DataFrame'i, 'relevance_score' sütunu eklenmiş
    """
    if news_df.empty:
        return news_df

    logger.info(f"Haberler filtreleniyor: {symbol}")
    logger.info(f"Toplam haber sayısı: {len(news_df)}")

    stock_info = get_stock_info(symbol)
    code = stock_info['code']

    # Bilinmeyen hisse: sembol adını ve şirket adını keyword olarak kullan
    if not stock_info['keywords']:
        auto_keywords = [code.lower()]
        # Örn: SASA -> 'sasa', REEDR -> 'reedr', 'reeder'
        if len(code) >= 4:
            auto_keywords.append(code[:4].lower())
        stock_info['keywords'] = auto_keywords
        logger.info(f"{code} için otomatik keyword üretildi: {auto_keywords}")

    relevance_scores = []

    for _, row in news_df.iterrows():
        # Başlık ağırlıklı: başlıkta geçen eşleşme daha değerli
        title_text = str(row.get('title', ''))
        content_text = str(row.get('content', ''))
        full_text = title_text + ' ' + content_text

        title_lower = title_text.lower()
        full_lower = full_text.lower()

        # Başlıkta eşleşme varsa direkt yüksek skor ver
        title_matches = sum(1 for kw in stock_info['keywords'] if kw in title_lower)
        content_matches = sum(1 for kw in stock_info['keywords'] if kw in full_lower)

        if title_matches > 0:
            # Başlıkta geçiyorsa kesinlikle ilgili
            stock_score = min(0.5 + title_matches * 0.25, 1.0)
        elif content_matches > 0:
            stock_score = min(content_matches / max(len(stock_info['keywords']), 3), 1.0)
        else:
            stock_score = 0.0

        # Sektör skoru (sadece hisse özel eşleşme yoksa devreye girer)
        sector_score = 0.0
        if include_sector_news and stock_info['sector'] and stock_score == 0.0:
            sector_keywords = SECTOR_KEYWORDS.get(stock_info['sector'], [])
            raw = calculate_relevance_score(full_lower, sector_keywords)
            sector_score = raw * 0.4  # Sektör haberleri daha düşük ağırlık

        total_score = max(stock_score, sector_score)
        relevance_scores.append(total_score)

    news_df = news_df.copy()
    news_df['relevance_score'] = relevance_scores

    filtered_df = news_df[news_df['relevance_score'] >= min_relevance].copy()
    filtered_df = filtered_df.sort_values('relevance_score', ascending=False)

    logger.info(f"Filtreleme sonrası: {len(filtered_df)} haber kaldı")
    if not filtered_df.empty:
        logger.info(f"Ortalama ilişki skoru: {filtered_df['relevance_score'].mean():.2f}")

    if len(filtered_df) < 5:
        logger.warning(f"Çok az ilgili haber bulundu ({len(filtered_df)})")

    return filtered_df


def add_custom_keywords(symbol: str, keywords: List[str]) -> None:
    """
    Bir hisse senedi için özel anahtar kelimeler ekler.
    
    Args:
        symbol: Hisse kodu (örn: 'THYAO')
        keywords: Eklenecek anahtar kelimeler
    """
    code = symbol.replace('.IS', '').upper()
    
    if code not in STOCK_KEYWORDS:
        STOCK_KEYWORDS[code] = []
    
    STOCK_KEYWORDS[code].extend(keywords)
    logger.info(f"{code} için {len(keywords)} anahtar kelime eklendi")


def get_relevance_summary(news_df: pd.DataFrame) -> Dict:
    """
    Filtrelenmiş haberlerin ilişki özetini döndürür.
    
    Args:
        news_df: Filtrelenmiş haber DataFrame'i (relevance_score sütunu olmalı)
    
    Returns:
        Dictionary: İstatistikler
    """
    if 'relevance_score' not in news_df.columns:
        return {}
    
    return {
        'total_news': len(news_df),
        'avg_relevance': news_df['relevance_score'].mean(),
        'high_relevance': len(news_df[news_df['relevance_score'] >= 0.7]),
        'medium_relevance': len(news_df[(news_df['relevance_score'] >= 0.3) & (news_df['relevance_score'] < 0.7)]),
        'low_relevance': len(news_df[news_df['relevance_score'] < 0.3]),
    }
