"""
Haber Toplama Modülü

Bu modül RSS beslemelerinden Türkçe ekonomi haberlerini toplar,
yapılandırır ve CSV/JSON formatında kaydeder.
"""

import logging
from typing import List, Optional
from datetime import datetime
import pandas as pd
import feedparser
from dateutil import parser as date_parser

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_rss_news(
    rss_urls: List[str], 
    max_articles: int = 100,
    filter_keywords: Optional[List[str]] = None,
    per_feed_limit: int = 50
) -> pd.DataFrame:
    """
    RSS beslemelerinden haberleri çeker.
    
    Args:
        rss_urls: RSS besleme URL'leri listesi
        max_articles: Maksimum toplam haber sayısı
        filter_keywords: Filtreleme için anahtar kelimeler (opsiyonel)
        per_feed_limit: Her RSS kaynağından çekilecek maksimum haber sayısı
    
    Returns:
        DataFrame with columns: date, title, content, source
    """
    all_articles = []
    
    for url in rss_urls:
        try:
            logger.info(f"RSS beslemesi çekiliyor: {url}")
            feed = feedparser.parse(url)
            
            # RSS formatı geçerli mi kontrol et
            if feed.bozo and not feed.entries:
                logger.warning(f"RSS formatı geçersiz: {url}")
                continue
            
            feed_articles = []
            
            # Her haber için bilgileri çıkar
            for entry in feed.entries[:per_feed_limit]:  # Her kaynaktan daha fazla haber çek
                try:
                    # Tarih bilgisini çıkar
                    article_date = _extract_date(entry)
                    
                    # Başlık
                    title = entry.get('title', '').strip()
                    if not title:
                        continue
                    
                    # İçerik (summary veya description)
                    content = entry.get('summary', entry.get('description', '')).strip()
                    if not content:
                        logger.debug(f"Boş içerik atlandı: {title}")
                        continue
                    
                    # Anahtar kelime filtresi varsa uygula
                    if filter_keywords:
                        text = f"{title} {content}".lower()
                        # En az bir anahtar kelime eşleşmeli
                        if not any(keyword.lower() in text for keyword in filter_keywords):
                            continue
                    
                    # Kaynak (feed başlığı veya URL'den)
                    source = feed.feed.get('title', _extract_source_from_url(url))

                    # Haber linki
                    link = entry.get('link', '')

                    # Google News entry'lerinde gerçek kaynak source alanında
                    entry_source = entry.get('source', {})
                    if entry_source.get('title'):
                        source = entry_source['title']

                    feed_articles.append({
                        'date': article_date,
                        'title': title,
                        'content': content,
                        'source': source,
                        'link': link,
                    })
                    
                except Exception as e:
                    logger.debug(f"Haber işlenirken hata: {e}")
                    continue
            
            all_articles.extend(feed_articles)
            logger.info(f"{url} adresinden {len(feed_articles)} haber çekildi")
            
        except Exception as e:
            logger.error(f"RSS beslemesine bağlanılamadı: {url} - Hata: {e}")
            continue
    
    # DataFrame oluştur
    if not all_articles:
        logger.warning("Hiç haber çekilemedi, boş DataFrame döndürülüyor")
        return pd.DataFrame(columns=['date', 'title', 'content', 'source', 'link'])
    
    df = pd.DataFrame(all_articles)
    
    # Tarihe göre sırala
    df = df.sort_values('date', ascending=False).reset_index(drop=True)
    
    # Maksimum makale sayısını uygula
    if len(df) > max_articles:
        df = df.head(max_articles)
    
    logger.info(f"Toplam {len(df)} haber başarıyla çekildi")
    
    return df


def _extract_date(entry: dict) -> datetime:
    """
    RSS entry'den tarih bilgisini çıkarır.
    
    Args:
        entry: feedparser entry objesi
    
    Returns:
        datetime objesi
    """
    # Önce published_parsed'ı dene
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6])
        except Exception:
            pass
    
    # Sonra published string'ini parse et
    if hasattr(entry, 'published') and entry.published:
        try:
            return date_parser.parse(entry.published)
        except Exception:
            pass
    
    # updated_parsed'ı dene
    if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
        try:
            return datetime(*entry.updated_parsed[:6])
        except Exception:
            pass
    
    # updated string'ini parse et
    if hasattr(entry, 'updated') and entry.updated:
        try:
            return date_parser.parse(entry.updated)
        except Exception:
            pass
    
    # Hiçbiri yoksa şu anki zamanı kullan
    logger.debug("Tarih bilgisi bulunamadı, şu anki zaman kullanılıyor")
    return datetime.now()


def _extract_source_from_url(url: str) -> str:
    """
    URL'den kaynak adını çıkarır.
    
    Args:
        url: RSS feed URL'si
    
    Returns:
        Kaynak adı
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        # www. önekini kaldır
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return 'Bilinmeyen Kaynak'


def save_news(df: pd.DataFrame, filepath: str, format: str = 'csv') -> None:
    """
    Haberleri dosyaya kaydeder.
    
    Args:
        df: Haber verileri içeren DataFrame
        filepath: Kayıt yolu
        format: 'csv' veya 'json'
    """
    try:
        if format == 'csv':
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Haberler CSV formatında kaydedildi: {filepath}")
        elif format == 'json':
            df.to_json(filepath, orient='records', date_format='iso', force_ascii=False, indent=2)
            logger.info(f"Haberler JSON formatında kaydedildi: {filepath}")
        else:
            raise ValueError(f"Desteklenmeyen format: {format}. 'csv' veya 'json' kullanın.")
    except Exception as e:
        logger.error(f"Haberler kaydedilirken hata oluştu: {e}")
        raise


def load_news(filepath: str, format: str = 'csv') -> pd.DataFrame:
    """
    Kaydedilmiş haberleri yükler.
    
    Args:
        filepath: Dosya yolu
        format: 'csv' veya 'json'
    
    Returns:
        Haber verileri içeren DataFrame
    """
    try:
        if format == 'csv':
            df = pd.read_csv(filepath, encoding='utf-8', parse_dates=['date'])
            logger.info(f"Haberler CSV formatından yüklendi: {filepath}")
        elif format == 'json':
            df = pd.read_json(filepath, orient='records')
            # Tarih sütununu datetime'a çevir
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Haberler JSON formatından yüklendi: {filepath}")
        else:
            raise ValueError(f"Desteklenmeyen format: {format}. 'csv' veya 'json' kullanın.")
        
        return df
    except FileNotFoundError:
        logger.error(f"Dosya bulunamadı: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Haberler yüklenirken hata oluştu: {e}")
        raise
