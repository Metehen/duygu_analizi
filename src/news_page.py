# Copyright (C) 2026  Metehen
# Licensed under GNU AGPL v3. See LICENSE file for details.
"""
Haber Okuma Sayfası
Hisse haberleri ve genel piyasa haberlerini duygu analizi ile gösterir.
"""

import streamlit as st
import pandas as pd
import urllib.parse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RSS_FEEDS, DEFAULT_SYMBOLS
from src.news_fetcher import fetch_rss_news
from src.news_filter import STOCK_KEYWORDS
from src.sentiment import SentimentAnalyzer

# Sektör etiketleri ve RSS eşleşmesi
SEKTOR_LABELS = {
    'borsa_odakli': '📊 Borsa & Finans',
    'turizm': '✈️ Turizm',
    'enerji': '⚡ Enerji',
    'teknoloji': '💻 Teknoloji',
    'analiz_yorum': '🔍 Analiz & Yorum',
    'genel_ekonomi': '🌍 Genel Ekonomi',
}

# Hisse kodu -> arama terimi eşleşmesi (Google News için)
STOCK_SEARCH_TERMS = {
    'THYAO': 'THYAO türk hava yolları',
    'GARAN': 'GARAN garanti bankası',
    'ISCTR': 'ISCTR iş bankası',
    'AKBNK': 'AKBNK akbank',
    'TUPRS': 'TUPRS tüpraş',
    'ASELS': 'ASELS aselsan',
    'EREGL': 'EREGL ereğli demir',
    'KCHOL': 'KCHOL koç holding',
    'SAHOL': 'SAHOL sabancı holding',
    'PETKM': 'PETKM petkim',
    'SISE': 'SISE şişecam',
    'TCELL': 'TCELL turkcell',
    'TTKOM': 'TTKOM türk telekom',
    'VAKBN': 'VAKBN vakıfbank',
    'YKBNK': 'YKBNK yapı kredi',
    'BIMAS': 'BIMAS bim market',
    'MGROS': 'MGROS migros',
    'TOASO': 'TOASO tofaş',
    'FROTO': 'FROTO ford otosan',
    'SASA': 'SASA polyester',
    'REEDR': 'REEDR reeder teknoloji',
}


def _build_gnews_url(query: str) -> str:
    """Google News RSS URL'si oluşturur."""
    encoded = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=tr&gl=TR&ceid=TR:tr"


def _get_stock_search_term(symbol: str) -> str:
    """Hisse kodu için arama terimini döndürür. Bilinmiyorsa kodu kullanır."""
    code = symbol.replace('.IS', '').upper()
    return STOCK_SEARCH_TERMS.get(code, code)


def _get_sentiment_color(score: float) -> str:
    if score >= 0.3:
        return "🟢"
    elif score <= -0.3:
        return "🔴"
    else:
        return "🟡"


def _render_news_card(row: pd.Series, show_relevance: bool = False):
    """Tek bir haberi kart olarak render eder."""
    score = row.get('compound', 0.0)
    emoji = _get_sentiment_color(score)
    score_color = "green" if score >= 0.3 else ("red" if score <= -0.3 else "orange")
    link = row.get('link', '')

    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{row['title']}**")
            st.caption(f"📰 {row['source']}  |  🕐 {str(row['date'])[:16]}")
            if link:
                st.markdown(f"[🔗 Habere Git]({link})", unsafe_allow_html=False)
        with col2:
            st.markdown(f"### {emoji}")
            st.markdown(
                f"<span style='color:{score_color}; font-size:18px; font-weight:bold'>{score:+.2f}</span>",
                unsafe_allow_html=True
            )
            st.caption("Duygu")
            if show_relevance and 'relevance_score' in row.index:
                st.caption(f"İlgi: {row['relevance_score']:.2f}")
        st.divider()


def _run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame'deki haberlere duygu analizi uygular."""
    analyzer = SentimentAnalyzer(method='lexicon')
    texts = (df['title'] + ' ' + df['content'].fillna('')).tolist()
    scores = analyzer.analyze_batch(texts)
    df = df.copy()
    df['compound'] = scores['compound'].values
    return df


def display_stock_news():
    """Hisse haberleri alt sayfası."""
    st.subheader("📊 Hisse Haberleri")

    known_symbols = [s.replace('.IS', '') for s in DEFAULT_SYMBOLS]

    col1, col2 = st.columns([2, 2])
    with col1:
        selected = st.selectbox("Bilinen hisselerden seç", ["-- Seç --"] + known_symbols)
    with col2:
        manual = st.text_input("veya hisse kodu yaz (örn: SASA, REEDR)", "").strip().upper()

    symbol = manual if manual else (selected if selected != "-- Seç --" else None)

    if not symbol:
        st.info("Bir hisse seçin veya hisse kodu girin.")
        return

    max_news = st.slider("Maksimum haber sayısı", 10, 100, 50, step=10)

    if st.button("🔍 Haberleri Getir", key="stock_news_btn"):
        search_term = _get_stock_search_term(symbol)
        gnews_url = _build_gnews_url(search_term)

        with st.spinner(f"{symbol} haberleri Google News'ten çekiliyor..."):
            df = fetch_rss_news([gnews_url], max_articles=max_news, per_feed_limit=max_news)

        if df.empty:
            st.warning("Haber çekilemedi. İnternet bağlantınızı kontrol edin.")
            return

        with st.spinner("Duygu analizi yapılıyor..."):
            df = _run_sentiment(df)

        avg = df['compound'].mean()
        pos = (df['compound'] >= 0.3).sum()
        neg = (df['compound'] <= -0.3).sum()
        neu = len(df) - pos - neg

        st.markdown(f"### {symbol} — {len(df)} haber bulundu")
        st.caption(f"Arama terimi: `{search_term}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ortalama Duygu", f"{avg:+.2f}")
        m2.metric("🟢 Pozitif", pos)
        m3.metric("🟡 Nötr", neu)
        m4.metric("🔴 Negatif", neg)

        st.markdown("---")
        for _, row in df.iterrows():
            _render_news_card(row)


def display_market_news():
    """Genel piyasa haberleri alt sayfası."""
    st.subheader("🌍 Genel Piyasa Haberleri")

    sektor_options = {v: k for k, v in SEKTOR_LABELS.items()}
    secilen_label = st.selectbox("Kategori seç", list(SEKTOR_LABELS.values()))
    secilen_key = sektor_options[secilen_label]

    max_news = st.slider("Maksimum haber sayısı", 10, 150, 40, step=10)

    if st.button("🔍 Haberleri Getir", key="market_news_btn"):
        # İlgili RSS URL'lerini al
        if secilen_key == 'borsa_odakli':
            urls = RSS_FEEDS['borsa_odakli']
        elif secilen_key in RSS_FEEDS.get('sektorel', {}):
            urls = RSS_FEEDS['sektorel'][secilen_key]
        elif secilen_key == 'analiz_yorum':
            urls = RSS_FEEDS['analiz_yorum']
        elif secilen_key == 'genel_ekonomi':
            urls = RSS_FEEDS['genel_ekonomi']
        else:
            urls = []

        if not urls:
            st.warning("Bu kategori için RSS kaynağı bulunamadı.")
            return

        with st.spinner(f"{secilen_label} haberleri çekiliyor..."):
            df = fetch_rss_news(urls, max_articles=max_news)

        if df.empty:
            st.warning("Haber çekilemedi. İnternet bağlantınızı kontrol edin.")
            return

        with st.spinner("Duygu analizi yapılıyor..."):
            df = _run_sentiment(df)

        avg = df['compound'].mean()
        pos = (df['compound'] >= 0.3).sum()
        neg = (df['compound'] <= -0.3).sum()
        neu = len(df) - pos - neg

        st.markdown(f"### {secilen_label} — {len(df)} haber")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ortalama Duygu", f"{avg:+.2f}")
        m2.metric("🟢 Pozitif", pos)
        m3.metric("🟡 Nötr", neu)
        m4.metric("🔴 Negatif", neg)

        st.markdown("---")
        for _, row in df.iterrows():
            _render_news_card(row)


def display_news_page():
    """Ana haber sayfası — iki alt bölüm arasında geçiş."""
    st.title("📰 Haber Okuma")
    st.markdown("Hisse bazlı veya genel piyasa haberlerini okuyun, duygu skorlarını inceleyin.")
    st.markdown("---")

    if 'news_section' not in st.session_state:
        st.session_state.news_section = None

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        if st.button("📊 Hisse Haberleri", use_container_width=True):
            st.session_state.news_section = 'stock'

    with col3:
        if st.button("🌍 Genel Piyasa Haberleri", use_container_width=True):
            st.session_state.news_section = 'market'

    if st.session_state.news_section:
        st.markdown("---")
        if st.button("← Geri", key="news_back"):
            st.session_state.news_section = None
            st.rerun()

        if st.session_state.news_section == 'stock':
            display_stock_news()
        elif st.session_state.news_section == 'market':
            display_market_news()
