"""
SQLite veri kalıcılık modülü.

Haber ve fiyat verilerini önbelleğe alarak tekrar tekrar RSS/Yahoo Finance
isteklerini azaltır.

Tablolar:
    news  — Haber makaleleri (title UNIQUE ile dedup)
    prices— OHLCV fiyat verisi (symbol+date UNIQUE ile dedup)
    meta  — Cache meta bilgisi (son güncelleme zamanı)

Kullanım:
    from src.database import init_db, cache_news, get_cached_news

    init_db()  # İlk kurulum

    # Haber önbelleğe al
    cache_news(news_df, symbol="THYAO.IS")

    # Taze haber getir (6 saatten yeni)
    cached = get_cached_news(symbol="THYAO.IS", hours_old=6)
    if cached is not None:
        news_df = cached  # RSS çekmeden devam et

Veritabanı: data/stock_analysis.db
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────────────────────
DB_DIR  = "data"
DB_PATH = os.path.join(DB_DIR, "stock_analysis.db")


# ──────────────────────────────────────────────────────────────
# Bağlantı yönetimi
# ──────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """SQLite bağlantısı döndürür (dizini oluşturur)."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Veritabanı tablolarını oluşturur (zaten varsa atlar)."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS news (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL UNIQUE,
                content     TEXT,
                source      TEXT,
                date        TEXT,
                symbol      TEXT,
                cached_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prices (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                date        TEXT NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                cached_at   TEXT NOT NULL,
                UNIQUE(symbol, date)
            );

            CREATE TABLE IF NOT EXISTS meta (
                key     TEXT PRIMARY KEY,
                value   TEXT
            );

            CREATE TABLE IF NOT EXISTS analysis_results (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id              TEXT,
                timestamp               TEXT NOT NULL,
                symbol                  TEXT NOT NULL,
                start_date              TEXT,
                end_date                TEXT,
                epochs                  INTEGER,
                lookback                INTEGER,
                sentiment_method        TEXT,
                lag                     INTEGER,
                n_features              INTEGER,
                n_train                 INTEGER,
                n_test                  INTEGER,
                groq_enabled             INTEGER DEFAULT 0,
                rmse                    REAL,
                mae                     REAL,
                r2                      REAL,
                mape                    REAL,
                directional_accuracy    REAL,
                total_return            REAL,
                num_trades              INTEGER,
                final_capital           REAL,
                mean_cv                REAL,
                xgb_accuracy            REAL,
                llm_explanation         TEXT,
                model_name              TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_news_symbol        ON news(symbol);
            CREATE INDEX IF NOT EXISTS idx_news_date          ON news(date);
            CREATE INDEX IF NOT EXISTS idx_prices_symbol      ON prices(symbol);
            CREATE INDEX IF NOT EXISTS idx_prices_date         ON prices(date);
            CREATE INDEX IF NOT EXISTS idx_results_symbol      ON analysis_results(symbol);
            CREATE INDEX IF NOT EXISTS idx_results_timestamp   ON analysis_results(timestamp);
        """)
        conn.commit()
        logger.debug("Veritabanı tabloları hazır: %s", DB_PATH)
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────
# Haber önbelleği
# ──────────────────────────────────────────────────────────────

def cache_news(news_df: pd.DataFrame, symbol: Optional[str] = None) -> int:
    """
    Haber DataFrame'ini önbelleğe kaydeder.
    Aynı başlığa sahip haberler tekrar yazılmaz (INSERT OR IGNORE).

    Args:
        news_df: date, title, content, source sütunları içeren DataFrame
        symbol:  Hangi hisse için çekildi (None = genel)

    Returns:
        Yeni eklenen satır sayısı
    """
    if news_df is None or news_df.empty:
        return 0

    init_db()
    conn = _get_conn()
    now = datetime.now().isoformat()
    added = 0

    try:
        for _, row in news_df.iterrows():
            title = str(row.get('title', '')).strip()
            if not title:
                continue
            cur = conn.execute(
                """INSERT OR IGNORE INTO news
                   (title, content, source, date, symbol, cached_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    title,
                    str(row.get('content', ''))[:5000],
                    str(row.get('source', '')),
                    str(row.get('date', '')),
                    symbol,
                    now,
                )
            )
            added += cur.rowcount
        conn.commit()
        logger.info("Önbelleğe %d haber kaydedildi (symbol=%s)", added, symbol)
    finally:
        conn.close()

    return added


def get_cached_news(
    symbol: Optional[str] = None,
    hours_old: int = 6
) -> Optional[pd.DataFrame]:
    """
    Önbelleğeki taze haberleri getirir.

    Args:
        symbol:    Hisse sembolü (None = tüm haberler)
        hours_old: Maksimum yaş (saat). Bu süreden yeni haberler döner.

    Returns:
        DataFrame (date, title, content, source) veya None (önbellek boş/eski)
    """
    init_db()
    conn = _get_conn()
    try:
        cutoff = (datetime.now() - timedelta(hours=hours_old)).isoformat()

        query = "SELECT date, title, content, source FROM news WHERE cached_at > ?"
        params = [cutoff]

        if symbol is not None:
            query += " AND (symbol = ? OR symbol IS NULL)"
            params.append(symbol)

        query += " ORDER BY date DESC"
        rows = conn.execute(query, params).fetchall()

        if not rows:
            return None

        df = pd.DataFrame([dict(r) for r in rows])
        logger.info(
            "Önbellekten %d haber getirildi (symbol=%s, hours_old=%d)",
            len(df), symbol, hours_old
        )
        return df
    finally:
        conn.close()


def get_news_cache_info(symbol: Optional[str] = None) -> dict:
    """
    Önbellek durumunu döndürür (UI'da göstermek için).

    Returns:
        {
            'count':    toplam haber sayısı,
            'last_update': son güncelleme zamanı (ISO string veya None),
            'symbol':   sorgulanan sembol
        }
    """
    init_db()
    conn = _get_conn()
    try:
        query = "SELECT COUNT(*) as cnt, MAX(cached_at) as last FROM news"
        params = []
        if symbol is not None:
            query += " WHERE symbol = ? OR symbol IS NULL"
            params.append(symbol)

        row = conn.execute(query, params).fetchone()
        return {
            'count':       row['cnt'] if row else 0,
            'last_update': row['last'] if row else None,
            'symbol':      symbol,
        }
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────
# Fiyat önbelleği
# ──────────────────────────────────────────────────────────────

def cache_prices(price_df: pd.DataFrame, symbol: str) -> int:
    """
    Fiyat DataFrame'ini önbelleğe kaydeder.
    Aynı (symbol, date) kombinasyonu tekrar yazılmaz.

    Args:
        price_df: date, open, high, low, close, volume sütunları içeren DataFrame
        symbol:   Hisse sembolü

    Returns:
        Yeni eklenen satır sayısı
    """
    if price_df is None or price_df.empty:
        return 0

    init_db()
    conn = _get_conn()
    now = datetime.now().isoformat()
    added = 0

    try:
        for _, row in price_df.iterrows():
            date_str = str(row.get('date', ''))
            if not date_str:
                continue
            # Tarih formatını normalize et (YYYY-MM-DD)
            try:
                date_str = pd.to_datetime(date_str).strftime('%Y-%m-%d')
            except Exception:
                pass

            cur = conn.execute(
                """INSERT OR IGNORE INTO prices
                   (symbol, date, open, high, low, close, volume, cached_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol,
                    date_str,
                    float(row.get('open', 0)),
                    float(row.get('high', 0)),
                    float(row.get('low', 0)),
                    float(row.get('close', 0)),
                    float(row.get('volume', 0)),
                    now,
                )
            )
            added += cur.rowcount
        conn.commit()
        logger.info("Önbelleğe %d fiyat satırı kaydedildi (%s)", added, symbol)
    finally:
        conn.close()

    return added


def get_cached_prices(
    symbol: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    Önbelleğeki fiyat verisini getirir.

    Tarih aralığının tamamı önbellekte varsa döner, eksik günler varsa
    None döner (o durumda Yahoo Finance'ten çekmek gerekir).

    Args:
        symbol:     Hisse sembolü
        start_date: Başlangıç (YYYY-MM-DD)
        end_date:   Bitiş (YYYY-MM-DD)

    Returns:
        DataFrame (date, open, high, low, close, volume) veya None
    """
    init_db()
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT date, open, high, low, close, volume
               FROM prices
               WHERE symbol = ? AND date >= ? AND date <= ?
               ORDER BY date ASC""",
            (symbol, start_date, end_date)
        ).fetchall()

        if not rows or len(rows) < 10:
            return None

        df = pd.DataFrame([dict(r) for r in rows])
        df['date'] = pd.to_datetime(df['date'])
        logger.info(
            "Önbellekten %d fiyat satırı getirildi (%s, %s → %s)",
            len(df), symbol, start_date, end_date
        )
        return df
    finally:
        conn.close()


def get_price_cache_info(symbol: str) -> dict:
    """
    Fiyat önbellek durumunu döndürür (UI'da göstermek için).

    Returns:
        {
            'count':       toplam satır,
            'last_update':  son güncelleme (ISO string veya None),
            'symbol':       sembol
        }
    """
    init_db()
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt, MAX(cached_at) as last FROM prices WHERE symbol = ?",
            (symbol,)
        ).fetchone()
        return {
            'count':       row['cnt'] if row else 0,
            'last_update':  row['last'] if row else None,
            'symbol':       symbol,
        }
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────
# Analiz sonuçları
# ──────────────────────────────────────────────────────────────

def save_analysis_result(
    results: dict,
    params: dict,
    session_id: Optional[str] = None
) -> int:
    """
    Bir analiz çalışmasının sonuç özetini veritabanına kaydeder.
    Analiz tamamlandıktan sonra çağrılır; geçmiş analizleri
    karşılaştırmak için kullanılır.

    Args:
        results:     run_analysis() sonuç sözlüğü
                     (comprehensive_metrics, backtest_results, lstm_cv,
                      xgb_accuracy, llm_explanation, model_name, used_lag)
        params:     Çalıştırma parametreleri
                     (symbol, start_date, end_date, epochs, lookback,
                      sentiment_method, groq_enabled, n_train, n_test,
                      n_features)
        session_id:  Oturum ID'si (logger'dan)

    Returns:
        Eklenen satır ID'si
    """
    init_db()
    conn = _get_conn()

    comp     = results.get('comprehensive_metrics', {})
    bt       = results.get('backtest_results', {})
    lstm_cv  = results.get('lstm_cv')
    mean_cv  = float(lstm_cv.mean()) if lstm_cv is not None and len(lstm_cv) > 0 else None

    try:
        cur = conn.execute(
            """INSERT INTO analysis_results (
                session_id, timestamp, symbol, start_date, end_date,
                epochs, lookback, sentiment_method, lag, n_features,
                n_train, n_test, groq_enabled,
                rmse, mae, r2, mape, directional_accuracy,
                total_return, num_trades, final_capital,
                mean_cv, xgb_accuracy, llm_explanation, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                datetime.now().isoformat(),
                params.get('symbol', ''),
                params.get('start_date'),
                params.get('end_date'),
                params.get('epochs'),
                params.get('lookback'),
                params.get('sentiment_method'),
                results.get('used_lag', params.get('lag', 0)),
                params.get('n_features'),
                params.get('n_train'),
                params.get('n_test'),
                1 if params.get('groq_enabled') else 0,
                comp.get('rmse'),
                comp.get('mae'),
                comp.get('r2'),
                comp.get('mape'),
                comp.get('directional_accuracy'),
                bt.get('total_return'),
                bt.get('num_trades'),
                bt.get('final_capital'),
                mean_cv,
                results.get('xgb_accuracy'),
                results.get('llm_explanation'),
                results.get('model_name'),
            )
        )
        conn.commit()
        row_id = cur.lastrowid
        logger.info("Analiz sonucu kaydedildi (id=%d, symbol=%s)", row_id, params.get('symbol'))
        return row_id
    finally:
        conn.close()


def get_analysis_history(
    symbol: Optional[str] = None,
    limit: int = 50
) -> pd.DataFrame:
    """
    Geçmiş analiz sonuçlarını getirir.

    Args:
        symbol: Hisse sembolü (None = tüm hisseler)
        limit:  Maksimum satır sayısı

    Returns:
        DataFrame — her satır bir analiz çalışması
    """
    init_db()
    conn = _get_conn()
    try:
        query = "SELECT * FROM analysis_results"
        params_list = []

        if symbol is not None:
            query += " WHERE symbol = ?"
            params_list.append(symbol)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params_list.append(limit)

        rows = conn.execute(query, params_list).fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        logger.info("Geçmiş analiz geçmişi getirildi: %d satır (symbol=%s)", len(df), symbol)
        return df
    finally:
        conn.close()
