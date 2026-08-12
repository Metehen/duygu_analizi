"""
Streamlit Web Arayüzü
Türkçe Ekonomi Haberleri Duygu Analizi & Hisse Tahmin Sistemi
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging
import os
import sys

# Proje kök dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config'i import et
from config import (
    RSS_FEEDS, RSS_FEEDS_LIST, DEFAULT_SYMBOLS, DEFAULT_START_DATE, DEFAULT_END_DATE,
    LSTM_CONFIG, XGBOOST_CONFIG, SENTIMENT_METHOD
)

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Ana Streamlit uygulaması.
    Kullanıcı arayüzünü oluşturur ve analiz sürecini yönetir.
    """
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="Türkçe Duygu Analizi & Hisse Tahmin",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Session state ile aktif sayfa takibi
    if 'active_page' not in st.session_state:
        st.session_state.active_page = None

    # Ana menü
    if st.session_state.active_page is None:
        _display_home()
        return

    # Geri butonu (sidebar)
    with st.sidebar:
        if st.button("🏠 Ana Menü"):
            st.session_state.active_page = None
            if 'news_section' in st.session_state:
                del st.session_state.news_section
            st.rerun()

    if st.session_state.active_page == 'tahmin':
        _display_tahmin_app()
    elif st.session_state.active_page == 'haberler':
        from src.news_page import display_news_page
        display_news_page()


def _display_home():
    """Ana menü ekranı."""
    st.title("📈 Türkçe Ekonomi Haberleri & Hisse Tahmin Sistemi")
    st.markdown("Aşağıdan devam etmek istediğiniz bölümü seçin.")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.markdown("### 📈 Hisse Tahmin Uygulaması")
        st.markdown(
            "Duygu analizi ve makine öğrenmesi modelleriyle "
            "hisse senedi fiyat tahmini yapın."
        )
        if st.button("Tahmin Uygulamasına Git →", use_container_width=True):
            st.session_state.active_page = 'tahmin'
            st.rerun()

    with col3:
        st.markdown("### 📰 Haber Okuma")
        st.markdown(
            "Hisse bazlı veya genel piyasa haberlerini okuyun, "
            "duygu skorlarını inceleyin."
        )
        if st.button("Haber Okumasına Git →", use_container_width=True):
            st.session_state.active_page = 'haberler'
            st.rerun()


def _display_tahmin_app():
    """Mevcut tahmin uygulaması — hiçbir şey değişmedi."""
    st.title("📈 Türkçe Ekonomi Haberleri Duygu Analizi & Hisse Tahmin Sistemi")
    st.markdown("""
    Bu uygulama, Türkçe ekonomi haberlerinden duygu analizi yaparak hisse senedi fiyat tahminleri üretir.

    **Özellikler:**
    - 📰 RSS beslemelerinden haber toplama
    - 🧠 Türkçe BERT veya sözlük tabanlı duygu analizi
    - 🤖 Gemini LLM ile gelişmiş haber yorumu (opsiyonel)
    - 📊 Teknik göstergeler (EMA, RSI, MACD)
    - 🔮 LSTM ve XGBoost modelleri
    - 📉 Geriye dönük test ve performans metrikleri
    """)

    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 Analiz", "📈 Model Performansı"])

    with tab1:
        display_analysis_tab()

    with tab2:
        display_model_performance()


def display_analysis_tab():
    """
    Ana analiz sekmesini görüntüler.
    """
    # Sidebar: Parametreler
    st.sidebar.header("⚙️ Analiz Parametreleri")
    
    # Hisse sembolü seçimi (hem seçilebilir hem yazılabilir)
    st.sidebar.subheader("Hisse Senedi")
    
    # Manuel giriş seçeneğini listeye ekle
    symbol_options = ["Manuel Gir..."] + DEFAULT_SYMBOLS
    
    symbol_selection = st.sidebar.selectbox(
        "Hisse Sembolü Seçin veya Manuel Girin",
        options=symbol_options,
        index=1,  # İlk gerçek hisse senedini varsayılan yap
        help="Listeden seçin veya 'Manuel Gir...' seçeneğini kullanarak kendi sembolünüzü yazın"
    )
    
    # Eğer "Manuel Gir..." seçildiyse, text input göster
    if symbol_selection == "Manuel Gir...":
        symbol = st.sidebar.text_input(
            "Hisse Sembolü",
            value="",
            placeholder="Örn: THYAO.IS",
            help="Borsa Istanbul sembolü (.IS ile bitmelidir)"
        )
        if not symbol:
            st.sidebar.warning("⚠️ Lütfen bir hisse sembolü girin")
    else:
        symbol = symbol_selection
    
    # Tarih aralığı
    st.sidebar.subheader("Tarih Aralığı")
    
    # Varsayılan tarihler
    default_start = datetime.strptime(DEFAULT_START_DATE, '%Y-%m-%d')
    default_end = datetime.strptime(DEFAULT_END_DATE, '%Y-%m-%d')
    
    # Bugünün tarihi
    today = datetime.now().date()
    
    start_date = st.sidebar.date_input(
        "Başlangıç Tarihi",
        value=default_start,
        min_value=datetime(2020, 1, 1).date(),
        max_value=today,
        help="Analiz başlangıç tarihi"
    )
    
    # Bitiş tarihi için min_value'yu dinamik olarak ayarla
    # Eğer start_date default_end'den sonraysa, start_date'i kullan
    end_min_value = max(start_date, default_start.date())
    end_default_value = max(default_end.date(), end_min_value)
    
    end_date = st.sidebar.date_input(
        "Bitiş Tarihi",
        value=end_default_value,
        min_value=end_min_value,
        max_value=today,
        help="Analiz bitiş tarihi"
    )
    
    # Model parametreleri (gelişmiş ayarlar)
    with st.sidebar.expander("🔧 Gelişmiş Ayarlar"):
        st.subheader("Haber Filtreleme")
        enable_news_filter = st.checkbox(
            "Hisse Özel Haber Filtresi",
            value=True,
            help="Sadece seçilen hisse ile ilgili haberleri kullan"
        )
        
        # Varsayılan değerler
        min_relevance = 0.2
        
        if enable_news_filter:
            min_relevance = st.slider(
                "Minimum İlişki Skoru",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Düşük değer = daha fazla haber, Yüksek değer = daha az ama daha ilgili haber"
            )
        
        st.markdown("---")
        
        # RSS Kaynak Filtreleme
        st.subheader("📰 RSS Kaynak Seçimi")
        st.caption("Hangi kaynaklardan haber çekilsin?")
        
        # Borsa Odaklı
        use_borsa = st.checkbox(
            "📊 Borsa Odaklı Kaynaklar",
            value=True,
            help=f"Bloomberg HT, Para Analiz, Foreks vb. ({len(RSS_FEEDS['borsa_odakli'])} kaynak)"
        )
        
        # Sektörel Kaynaklar
        use_sektorel = st.checkbox(
            "🏭 Sektörel Kaynaklar",
            value=True,
            help="Turizm, enerji, teknoloji gibi sektörel haberler"
        )
        
        # Sektörel alt seçenekler
        selected_sectors = []
        if use_sektorel:
            st.caption("Hangi sektörler dahil edilsin?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("✈️ Turizm", value=True, help=f"{len(RSS_FEEDS['sektorel']['turizm'])} kaynak"):
                    selected_sectors.append('turizm')
                if st.checkbox("⚡ Enerji", value=True, help=f"{len(RSS_FEEDS['sektorel']['enerji'])} kaynak"):
                    selected_sectors.append('enerji')
            with col2:
                if st.checkbox("💻 Teknoloji", value=True, help=f"{len(RSS_FEEDS['sektorel']['teknoloji'])} kaynak"):
                    selected_sectors.append('teknoloji')
        
        # Analiz ve Yorum
        use_analiz = st.checkbox(
            "📈 Analiz ve Yorum Kaynakları",
            value=True,
            help=f"Uzman analizleri ve piyasa yorumları ({len(RSS_FEEDS['analiz_yorum'])} kaynak)"
        )
        
        # Genel Ekonomi
        use_genel = st.checkbox(
            "📰 Genel Ekonomi Haberleri",
            value=True,
            help=f"Ana akım haber kaynakları ({len(RSS_FEEDS['genel_ekonomi'])} kaynak)"
        )
        
        # Seçilen kaynak sayısını göster
        selected_feed_count = 0
        if use_borsa:
            selected_feed_count += len(RSS_FEEDS['borsa_odakli'])
        if use_sektorel:
            for sector in selected_sectors:
                selected_feed_count += len(RSS_FEEDS['sektorel'][sector])
        if use_analiz:
            selected_feed_count += len(RSS_FEEDS['analiz_yorum'])
        if use_genel:
            selected_feed_count += len(RSS_FEEDS['genel_ekonomi'])
        
        if selected_feed_count > 0:
            st.success(f"✅ {selected_feed_count} kaynak seçildi")
        else:
            st.error("⚠️ En az bir kaynak seçmelisiniz!")
        
        st.markdown("---")
        
        st.subheader("LSTM Parametreleri")
        lstm_epochs = st.number_input(
            "Epoch Sayısı",
            min_value=10,
            max_value=200,
            value=LSTM_CONFIG['epochs'],
            step=10,
            help="LSTM eğitim epoch sayısı"
        )
        
        lstm_lookback = st.number_input(
            "Lookback Penceresi",
            min_value=10,
            max_value=120,
            value=LSTM_CONFIG['lookback'],
            step=10,
            help="Geçmiş gün sayısı"
        )
        
        st.subheader("Duygu Analizi")
        sentiment_method = st.selectbox(
            "Yöntem",
            options=['bert', 'lexicon'],
            index=0 if SENTIMENT_METHOD == 'bert' else 1,
            help="BERT: Daha doğru ama yavaş, Lexicon: Hızlı ama basit"
        )

        st.markdown("---")
        st.subheader("🤖 Groq LLM Desteği")
        use_gemini = st.checkbox(
            "Groq ile Gelişmiş Haber Analizi",
            value=False,
            help="En ilgili haberleri Groq LLM ile derinlemesine analiz eder ve LSTM tahminini açıklar. API key: https://console.groq.com"
        )
        gemini_api_key = ""
        gemini_top_n = 20
        bert_weight = 0.5
        if use_gemini:
            gemini_api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="https://console.groq.com adresinden ücretsiz alabilirsiniz."
            )
            gemini_top_n = st.slider(
                "LLM'e Gönderilecek Haber Sayısı",
                min_value=5,
                max_value=30,
                value=20,
                step=5,
                help="En yüksek ilişki skorlu N haber Groq'a gönderilir."
            )
            bert_weight = st.slider(
                "BERT Ağırlığı (Groq = 1 - BERT)",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                help="0.5 = BERT ve Groq eşit ağırlıkta"
            )
            if not gemini_api_key:
                st.warning("⚠️ API key girilmedi, Groq devre dışı kalacak.")
    
    # Analiz Et düğmesi
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button(
        "🚀 Analiz Et",
        type="primary",
        use_container_width=True,
        help="Analizi başlatmak için tıklayın"
    )
    
    # Ana içerik alanı
    if analyze_button:
        # Tarih formatını kontrol et
        if start_date >= end_date:
            st.error("❌ Başlangıç tarihi bitiş tarihinden önce olmalıdır!")
            return
        
        # Sembol formatını kontrol et
        if not symbol.endswith('.IS'):
            st.warning(f"⚠️ Sembol '.IS' ile bitmiyor. Otomatik olarak ekleniyor: {symbol}.IS")
            symbol = f"{symbol}.IS"
        
        # Seçilen RSS kaynaklarını topla
        selected_feeds = []
        if use_borsa:
            selected_feeds.extend(RSS_FEEDS['borsa_odakli'])
        if use_sektorel and selected_sectors:
            for sector in selected_sectors:
                selected_feeds.extend(RSS_FEEDS['sektorel'][sector])
        if use_analiz:
            selected_feeds.extend(RSS_FEEDS['analiz_yorum'])
        if use_genel:
            selected_feeds.extend(RSS_FEEDS['genel_ekonomi'])
        
        # En az bir kaynak seçilmiş mi kontrol et
        if not selected_feeds:
            st.error("❌ En az bir RSS kaynağı seçmelisiniz!")
            return
        
        # Analizi çalıştır
        run_analysis(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            lstm_epochs=lstm_epochs,
            lstm_lookback=lstm_lookback,
            sentiment_method=sentiment_method,
            enable_news_filter=enable_news_filter,
            min_relevance=min_relevance,
            selected_rss_feeds=selected_feeds,
            use_gemini=use_gemini,
            gemini_api_key=gemini_api_key,
            gemini_top_n=gemini_top_n,
            bert_weight=bert_weight
        )
    else:
        # İlk açılış mesajı
        st.info("👈 Soldaki parametreleri ayarlayın ve 'Analiz Et' düğmesine tıklayın.")
        
        # Örnek bilgiler göster
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📰 RSS Kaynakları",
                value=len(RSS_FEEDS_LIST),
                help="Haber toplama için kullanılacak RSS kaynağı sayısı"
            )
        
        with col2:
            st.metric(
                label="🤖 Model Sayısı",
                value=3,
                help="LSTM (fiyat tahmini) + XGBoost (yön tahmini) + Groq LLM (haber analizi)"
            )
        
        with col3:
            st.metric(
                label="📊 Teknik Göstergeler",
                value=7,
                help="EMA (3), RSI, MACD (2), Volume MA"
            )


def run_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    lstm_epochs: int,
    lstm_lookback: int,
    sentiment_method: str,
    enable_news_filter: bool = True,
    min_relevance: float = 0.2,
    selected_rss_feeds: list = None,
    use_gemini: bool = False,
    gemini_api_key: str = "",
    gemini_top_n: int = 20,
    bert_weight: float = 0.5
):
    """
    Tam analiz sürecini çalıştırır.

    Args:
        symbol: Hisse sembolü
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date: Bitiş tarihi (YYYY-MM-DD)
        lstm_epochs: LSTM epoch sayısı
        lstm_lookback: Lookback penceresi
        sentiment_method: Duygu analizi yöntemi ('bert' veya 'lexicon')
        enable_news_filter: Haber filtreleme aktif mi
        min_relevance: Minimum ilişki skoru
        selected_rss_feeds: Kullanılacak RSS kaynakları listesi (None ise tümü)
        use_gemini: Gemini LLM desteği aktif mi
        gemini_api_key: Gemini API anahtarı
        gemini_top_n: Gemini'ye gönderilecek maksimum haber sayısı
        bert_weight: Hibrit skorlamada BERT ağırlığı
    """
    # Import modülleri
    from src.news_fetcher import fetch_rss_news, save_news
    from src.sentiment import SentimentAnalyzer
    from src.features import (
        fetch_stock_data, 
        calculate_technical_indicators,
        merge_sentiment_with_prices,
        prepare_training_data
    )
    from src.model_lstm import (
        build_lstm_model,
        train_lstm_model,
        predict_lstm,
        save_lstm_model
    )
    from src.model_xgboost import (
        prepare_classification_data,
        train_xgboost_model,
        get_feature_importance,
        predict_direction,
        save_model as save_xgb_model
    )
    from src.backtest import simple_backtest, calculate_metrics
    from sklearn.model_selection import train_test_split
    
    # Sonuçları saklamak için dictionary
    results = {}

    # Tarih aralığı kontrolü — LSTM için yeterli veri gerekiyor
    from datetime import date as date_type
    try:
        d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
        d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
        gun_sayisi = (d1 - d0).days
        min_gun = lstm_lookback + 30  # lookback + minimum eğitim verisi
        if gun_sayisi < min_gun:
            st.error(
                f"❌ Seçilen tarih aralığı çok kısa: **{gun_sayisi} gün**. "
                f"Lookback={lstm_lookback} ile çalışmak için en az **{min_gun} gün** "
                f"({min_gun // 30} ay) gerekli. "
                f"Lütfen başlangıç tarihini en az **{(d1 - __import__('datetime').timedelta(days=min_gun)).strftime('%Y-%m-%d')}** "
                f"olarak ayarlayın."
            )
            return
    except Exception:
        pass

    # Kullanılacak RSS kaynaklarını belirle
    if selected_rss_feeds is None or len(selected_rss_feeds) == 0:
        selected_rss_feeds = RSS_FEEDS_LIST
    
    # Hisse bilgilerini al (filtreleme için)
    filter_keywords = None
    if enable_news_filter:
        from src.news_filter import get_stock_info, SECTOR_KEYWORDS
        stock_info = get_stock_info(symbol)
        
        # Anahtar kelimeleri birleştir
        filter_keywords = stock_info['keywords'].copy() if stock_info['keywords'] else []
        
        # Sektör kelimelerini de ekle
        if stock_info['sector']:
            sector_keywords = SECTOR_KEYWORDS.get(stock_info['sector'], [])
            filter_keywords.extend(sector_keywords)
        
        if filter_keywords:
            st.info(f"🔍 {len(filter_keywords)} anahtar kelime ile filtreleme yapılacak")
            logger.info(f"Filtreleme anahtar kelimeleri: {filter_keywords[:10]}...")  # İlk 10'unu logla
    
    try:
        # 1. Haber Toplama
        st.header("📰 1. Haber Toplama")
        st.info(f"📡 {len(selected_rss_feeds)} RSS kaynağından haber çekiliyor...")
        
        with st.spinner("RSS beslemelerinden haberler çekiliyor..."):
            try:
                # Filtreleme ile haber çek
                news_df = fetch_rss_news(
                    selected_rss_feeds, 
                    max_articles=200,
                    filter_keywords=filter_keywords if enable_news_filter else None,
                    per_feed_limit=100  # Her kaynaktan daha fazla haber çek
                )
                
                if news_df.empty:
                    st.warning("⚠️ Hiç haber çekilemedi. Duygu analizi atlanacak.")
                    news_df = None
                else:
                    st.success(f"✅ {len(news_df)} ilgili haber başarıyla çekildi")
                    
                    # Ek filtreleme (daha hassas)
                    if enable_news_filter and not news_df.empty:
                        from src.news_filter import filter_news_by_stock, get_relevance_summary
                        
                        original_count = len(news_df)
                        news_df = filter_news_by_stock(
                            news_df, 
                            symbol, 
                            min_relevance=min_relevance
                        )
                        
                        if len(news_df) > 0:
                            st.info(f"🎯 Hassas filtreleme sonrası: {len(news_df)}/{original_count} haber")
                            
                            # İlişki özeti
                            summary = get_relevance_summary(news_df)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Yüksek İlişkili", summary.get('high_relevance', 0))
                            with col2:
                                st.metric("Orta İlişkili", summary.get('medium_relevance', 0))
                            with col3:
                                st.metric("Düşük İlişkili", summary.get('low_relevance', 0))
                        else:
                            st.warning(f"⚠️ Hassas filtreleme sonrası haber kalmadı, tüm çekilen haberler kullanılacak")
                            # Tüm çekilen haberleri kullan
                            news_df = fetch_rss_news(
                                selected_rss_feeds, 
                                max_articles=200,
                                filter_keywords=None,  # Filtresiz çek
                                per_feed_limit=100
                            )
                            if not news_df.empty:
                                news_df['relevance_score'] = 0.0  # Sıfır skor
                    
                    # Haber örnekleri göster
                    with st.expander("📄 Örnek Haberler"):
                        display_cols = ['date', 'title', 'source']
                        if 'relevance_score' in news_df.columns:
                            display_cols.append('relevance_score')
                        st.dataframe(news_df.head(5)[display_cols])
                    
                    results['news_df'] = news_df
            except Exception as e:
                st.error(f"❌ Haber toplama hatası: {e}")
                logger.error(f"Haber toplama hatası: {e}")
                news_df = None
        
        # 2. Duygu Analizi
        st.header("🧠 2. Duygu Analizi")
        if news_df is not None and not news_df.empty:
            with st.spinner(f"Duygu analizi yapılıyor ({sentiment_method} yöntemi)..."):
                try:
                    analyzer = SentimentAnalyzer(method=sentiment_method)
                    
                    # Batch analiz
                    sentiment_scores = analyzer.analyze_batch(news_df['content'].tolist())
                    
                    # Tarih bilgisini ekle
                    sentiment_df = pd.DataFrame(sentiment_scores)
                    sentiment_df['date'] = news_df['date'].values
                    
                    st.success(f"✅ {len(sentiment_df)} haber için duygu analizi tamamlandı")
                    
                    # Duygu istatistikleri
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ortalama Pozitif", f"{sentiment_df['positive'].mean():.3f}")
                    with col2:
                        st.metric("Ortalama Negatif", f"{sentiment_df['negative'].mean():.3f}")
                    with col3:
                        st.metric("Ortalama Compound", f"{sentiment_df['compound'].mean():.3f}")
                    
                    results['sentiment_df'] = sentiment_df
                except Exception as e:
                    st.error(f"❌ Duygu analizi hatası: {e}")
                    logger.error(f"Duygu analizi hatası: {e}")
                    sentiment_df = None
        else:
            st.info("ℹ️ Haber verisi olmadığı için duygu analizi atlanıyor")
            sentiment_df = None

        # 2.5 Groq LLM Analizi (opsiyonel)
        if use_gemini and gemini_api_key and news_df is not None and not news_df.empty:
            st.header("🤖 2.5 Groq LLM Analizi")
            llm_top_n_actual = min(gemini_top_n, len(news_df))
            st.info(f"📡 En ilgili {llm_top_n_actual} haber Groq'a gönderiliyor...")

            with st.spinner("Groq haberleri analiz ediyor..."):
                try:
                    from src.llm_analyzer import LLMAnalyzer
                    llm = LLMAnalyzer(api_key=gemini_api_key)

                    news_df_llm = llm.analyze_news_batch(
                        news_df,
                        symbol=symbol,
                        top_n=gemini_top_n,
                        delay=2.0
                    )

                    analyzed_count = int(news_df_llm['llm_analyzed'].sum())
                    st.success(f"✅ {analyzed_count} haber Groq ile analiz edildi")

                    # LLM sonuçlarını tabloda göster
                    with st.expander("🔍 Groq Haber Yorumları"):
                        llm_cols = ['title', 'llm_direction', 'llm_sentiment_score',
                                    'llm_impact', 'llm_time_horizon', 'llm_reasoning']
                        display_df = news_df_llm[news_df_llm['llm_analyzed']][llm_cols].head(10)
                        display_df = display_df.rename(columns={
                            'title': 'Başlık',
                            'llm_direction': 'Yön',
                            'llm_sentiment_score': 'Skor',
                            'llm_impact': 'Etki',
                            'llm_time_horizon': 'Vade',
                            'llm_reasoning': 'Yorum'
                        })
                        st.dataframe(display_df, use_container_width=True)

                    # İstatistikler
                    analyzed = news_df_llm[news_df_llm['llm_analyzed']]
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        pos = (analyzed['llm_direction'] == 'pozitif').sum()
                        st.metric("🟢 Pozitif Haber", pos)
                    with c2:
                        neg = (analyzed['llm_direction'] == 'negatif').sum()
                        st.metric("🔴 Negatif Haber", neg)
                    with c3:
                        avg_score = analyzed['llm_sentiment_score'].mean()
                        st.metric("Ortalama LLM Skoru", f"{avg_score:+.3f}")

                    # BERT/Lexicon ile hibrit birleştir
                    if sentiment_df is not None:
                        sentiment_df = LLMAnalyzer.merge_with_bert_scores(
                            sentiment_df=sentiment_df,
                            news_df_with_llm=news_df_llm,
                            bert_weight=bert_weight,
                            llm_weight=round(1.0 - bert_weight, 1)
                        )
                        st.success(
                            f"✅ Hibrit skor oluşturuldu "
                            f"(BERT: %{int(bert_weight*100)} + Groq: %{int((1-bert_weight)*100)})"
                        )

                    results['news_df_llm'] = news_df_llm
                    results['llm_analyzer'] = llm

                except Exception as e:
                    st.error(f"❌ Groq analiz hatası: {e}")
                    logger.error(f"Groq analiz hatası: {e}", exc_info=True)
        
        # 3. Özellik Hazırlama
        st.header("📊 3. Özellik Hazırlama")
        with st.spinner("Hisse fiyat verileri çekiliyor ve teknik göstergeler hesaplanıyor..."):
            try:
                # Fiyat verilerini çek
                price_df = fetch_stock_data(symbol, start_date, end_date)
                st.success(f"✅ {len(price_df)} günlük fiyat verisi çekildi")
                
                # Teknik göstergeleri hesapla
                price_df = calculate_technical_indicators(price_df)
                st.success("✅ Teknik göstergeler hesaplandı")
                
                # Duygu skorlarını birleştir (eğer varsa)
                if sentiment_df is not None:
                    features_df = merge_sentiment_with_prices(price_df, sentiment_df)
                    st.success("✅ Duygu skorları fiyat verileriyle birleştirildi")
                else:
                    features_df = price_df
                    # Boş duygu skorları ekle
                    for col in ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_compound']:
                        features_df[col] = 0.0
                    st.info("ℹ️ Duygu skorları olmadan devam ediliyor")
                
                # Özellik istatistikleri
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Toplam Satır", len(features_df))
                with col2:
                    st.metric("Özellik Sayısı", len(features_df.columns) - 1)  # date hariç
                with col3:
                    st.metric("Ortalama Fiyat", f"₺{features_df['close'].mean():.2f}")
                
                results['features_df'] = features_df
                
            except Exception as e:
                st.error(f"❌ Özellik hazırlama hatası: {e}")
                logger.error(f"Özellik hazırlama hatası: {e}")
                return
        
        # 4. Model Eğitimi
        st.header("🤖 4. Model Eğitimi")
        
        # 4.1 LSTM Modeli
        st.subheader("4.1 LSTM Modeli (Fiyat Tahmini)")
        with st.spinner("LSTM modeli eğitiliyor..."):
            try:
                # Model değerlendirme fonksiyonlarını import et
                from src.model_evaluation import (
                    calculate_comprehensive_metrics,
                    save_model_report
                )
                from datetime import datetime
                
                # Eğitim verisi hazırla
                X, y, scaler = prepare_training_data(
                    features_df,
                    target_column='close',
                    lookback=lstm_lookback
                )
                
                # Train-test split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                st.info(f"Eğitim: {len(X_train)} örnek, Test: {len(X_test)} örnek")
                
                # Model oluştur
                lstm_model = build_lstm_model(
                    input_shape=(X_train.shape[1], X_train.shape[2])
                )
                
                # Model adı oluştur (timestamp ile)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_name = f"lstm_{symbol.replace('.IS', '')}_{timestamp}"
                
                # Model kayıt dizinini oluştur
                os.makedirs('models/training_history/', exist_ok=True)
                
                # Model checkpoint yolu
                model_save_path = f'models/training_history/{model_name}_best.keras'
                
                # Modeli eğit (model_save_path ile)
                history = train_lstm_model(
                    lstm_model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    epochs=lstm_epochs,
                    batch_size=32,
                    model_save_path=model_save_path
                )
                
                # Eğitim geçmişini CSV olarak kaydet
                history_df = pd.DataFrame(history.history)
                history_csv_path = f'models/training_history/{model_name}_history.csv'
                history_df.to_csv(history_csv_path, index=False)
                
                st.success("✅ LSTM modeli eğitildi")
                
                # Tahmin yap — scaler ile orijinal fiyat ölçeğine dönüştür
                lstm_predictions = predict_lstm(lstm_model, X_test, scaler=scaler)
                
                # Basit metrikleri hesapla (backtest için)
                lstm_metrics = calculate_metrics(y_test, lstm_predictions)
                
                # Kapsamlı metrikleri hesapla (model raporu için)
                comprehensive_metrics = calculate_comprehensive_metrics(y_test, lstm_predictions)
                
                # Model raporunu kaydet
                report_path = f'models/training_history/{model_name}_report.json'
                save_model_report(model_name, comprehensive_metrics, report_path)
                
                st.success(f"✅ Model raporu kaydedildi: {report_path}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{comprehensive_metrics['rmse']:.2f}")
                with col2:
                    st.metric("MAE", f"{comprehensive_metrics['mae']:.2f}")
                with col3:
                    st.metric("R²", f"{comprehensive_metrics['r2']:.3f}")
                
                results['lstm_model'] = lstm_model
                results['lstm_predictions'] = lstm_predictions
                results['lstm_metrics'] = lstm_metrics
                results['comprehensive_metrics'] = comprehensive_metrics
                results['y_test'] = y_test
                results['scaler'] = scaler
                results['model_name'] = model_name

                # Groq ile LSTM tahmin açıklaması
                if use_gemini and gemini_api_key and 'llm_analyzer' in results:
                    try:
                        last_row = features_df.iloc[-1]
                        feature_snapshot = {
                            'rsi_14': round(float(last_row.get('rsi_14', 0)), 1),
                            'macd': round(float(last_row.get('macd', 0)), 4),
                            'ema_20': float(last_row.get('ema_20', 0)),
                            'ema_50': float(last_row.get('ema_50', 0)),
                            'sentiment_compound': round(float(last_row.get('sentiment_compound', 0)), 3),
                        }
                        if len(lstm_predictions) > 0 and len(y_test) > 0:
                            last_pred_real = float(lstm_predictions[-1])
                            last_actual_real = float(scaler.inverse_transform([[y_test[-1]]])[0][0])
                            change_pct = (last_pred_real - last_actual_real) / (last_actual_real + 1e-9) * 100
                        else:
                            change_pct = 0.0

                        with st.spinner("Groq tahmini açıklıyor..."):
                            explanation = results['llm_analyzer'].explain_prediction(
                                symbol=symbol.replace('.IS', ''),
                                lstm_change_pct=change_pct,
                                features=feature_snapshot
                            )
                        st.info(f"🤖 **Groq Tahmin Yorumu:** {explanation}")
                        results['llm_explanation'] = explanation
                    except Exception as e:
                        logger.warning(f"Groq açıklama hatası: {e}")
                
            except Exception as e:
                st.error(f"❌ LSTM eğitim hatası: {e}")
                logger.error(f"LSTM eğitim hatası: {e}", exc_info=True)
                import traceback
                st.error(f"Detay: {traceback.format_exc()}")
                return
        
        # 4.2 XGBoost Modeli
        st.subheader("4.2 XGBoost Modeli (Yön Tahmini)")
        with st.spinner("XGBoost modeli eğitiliyor..."):
            try:
                # Sınıflandırma verisi hazırla
                X_class, y_class = prepare_classification_data(features_df, target_column='close')
                
                # Tarih sütununu çıkar (eğer varsa)
                if 'date' in X_class.columns:
                    X_class = X_class.drop(columns=['date'])
                
                # Train-test split
                X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
                    X_class, y_class, test_size=0.2, random_state=42, shuffle=False
                )
                
                st.info(f"Eğitim: {len(X_train_xgb)} örnek, Test: {len(X_test_xgb)} örnek")
                
                # Model eğit
                xgb_model = train_xgboost_model(
                    X_train_xgb,
                    y_train_xgb,
                    X_test_xgb,
                    y_test_xgb
                )
                
                st.success("✅ XGBoost modeli eğitildi")
                
                # Tahmin yap
                xgb_predictions, xgb_probabilities = predict_direction(xgb_model, X_test_xgb)
                
                # Doğruluk hesapla
                accuracy = (xgb_predictions == y_test_xgb).mean() * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Doğruluk", f"{accuracy:.2f}%")
                with col2:
                    st.metric("Yukarı Tahmin", f"{xgb_predictions.sum()}/{len(xgb_predictions)}")
                
                # Özellik önemleri
                feature_importance = get_feature_importance(xgb_model, X_class.columns.tolist())
                
                results['xgb_model'] = xgb_model
                results['xgb_predictions'] = xgb_predictions
                results['xgb_probabilities'] = xgb_probabilities
                results['xgb_accuracy'] = accuracy
                results['feature_importance'] = feature_importance
                results['X_test_xgb'] = X_test_xgb
                results['y_test_xgb'] = y_test_xgb
                
            except Exception as e:
                st.error(f"❌ XGBoost eğitim hatası: {e}")
                logger.error(f"XGBoost eğitim hatası: {e}")
                return
        
        # 5. Geriye Dönük Test
        st.header("📉 5. Geriye Dönük Test")
        with st.spinner("Geriye dönük test yapılıyor..."):
            try:
                # y_test normalize, gerçek fiyata çevir
                actual_prices = scaler.inverse_transform(
                    y_test.reshape(-1, 1)
                ).flatten()

                test_dates = features_df['date'].iloc[-len(y_test):].values

                # actual_prices bir eleman daha uzun olmalı (backtest gereksinimi)
                actual_prices_extended = np.append(actual_prices, actual_prices[-1])
                
                backtest_results = simple_backtest(
                    lstm_predictions,
                    actual_prices_extended,
                    initial_capital=10000.0
                )
                
                st.success("✅ Geriye dönük test tamamlandı")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Toplam Getiri", f"{backtest_results['total_return']:.2f}%")
                with col2:
                    st.metric("Doğruluk", f"{backtest_results['accuracy']:.2f}%")
                with col3:
                    st.metric("İşlem Sayısı", backtest_results['num_trades'])
                with col4:
                    st.metric("Son Sermaye", f"₺{backtest_results['final_capital']:.2f}")
                
                results['backtest_results'] = backtest_results
                results['test_dates'] = test_dates
                results['actual_prices'] = actual_prices
                
            except Exception as e:
                st.error(f"❌ Geriye dönük test hatası: {e}")
                logger.error(f"Geriye dönük test hatası: {e}")
                # Devam et, kritik değil
        
        # 6. Sonuçları Görselleştir
        st.header("📊 6. Sonuçlar ve Görselleştirmeler")
        display_results(results)
        
        # Başarı mesajı
        st.success("🎉 Analiz başarıyla tamamlandı!")
        
    except Exception as e:
        st.error(f"❌ Beklenmeyen hata: {e}")
        logger.error(f"Beklenmeyen hata: {e}", exc_info=True)


def display_results(results: dict):
    """
    Analiz sonuçlarını görselleştirir.
    
    Args:
        results: Analiz sonuçları dictionary'si
    """
    # Fiyat ve Tahminler Grafiği
    if 'lstm_predictions' in results and 'actual_prices' in results:
        st.subheader("📈 Fiyat ve Tahminler")
        fig_price = plot_price_predictions(
            actual_prices=results['actual_prices'],
            predicted_prices=results['lstm_predictions'],
            dates=results.get('test_dates', None)
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # Groq açıklaması varsa göster
        if 'llm_explanation' in results:
            st.info(f"🤖 **Groq Yorumu:** {results['llm_explanation']}")
    
    # Duygu Skorları Grafiği
    if 'sentiment_df' in results and results['sentiment_df'] is not None:
        st.subheader("🧠 Duygu Skorları")
        fig_sentiment = plot_sentiment_scores(results['sentiment_df'])
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Performans Metrikleri
    st.subheader("📊 Model Performansı")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**LSTM Modeli (Fiyat Tahmini)**")
        if 'lstm_metrics' in results:
            metrics = results['lstm_metrics']
            st.write(f"- **RMSE:** {metrics['rmse']:.2f}")
            st.write(f"- **MAE:** {metrics['mae']:.2f}")
            st.write(f"- **R² Score:** {metrics['r2']:.3f}")
        
        if 'backtest_results' in results:
            bt = results['backtest_results']
            st.write(f"- **Toplam Getiri:** {bt['total_return']:.2f}%")
            st.write(f"- **İşlem Sayısı:** {bt['num_trades']}")
    
    with col2:
        st.markdown("**XGBoost Modeli (Yön Tahmini)**")
        if 'xgb_accuracy' in results:
            st.write(f"- **Doğruluk:** {results['xgb_accuracy']:.2f}%")
        
        if 'xgb_predictions' in results:
            up_count = results['xgb_predictions'].sum()
            total = len(results['xgb_predictions'])
            st.write(f"- **Yukarı Tahmin:** {up_count}/{total}")
            st.write(f"- **Aşağı Tahmin:** {total - up_count}/{total}")
    
    # Özellik Önemleri
    if 'feature_importance' in results:
        st.subheader("🔍 Özellik Önemleri (XGBoost)")
        fig_importance = plot_feature_importance(results['feature_importance'])
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Tablo olarak da göster
        with st.expander("📋 Detaylı Özellik Önemleri"):
            st.dataframe(results['feature_importance'], use_container_width=True)


def plot_price_predictions(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    dates: Optional[np.ndarray] = None
) -> 'plotly.graph_objects.Figure':
    """
    Gerçek ve tahmin edilen fiyatları görselleştirir.
    
    Args:
        actual_prices: Gerçek fiyatlar
        predicted_prices: Tahmin edilen fiyatlar
        dates: Tarih dizisi (opsiyonel)
    
    Returns:
        Plotly figure objesi
    """
    import plotly.graph_objects as go
    
    # X ekseni için index veya tarih kullan
    if dates is not None:
        x_axis = dates
        x_title = "Tarih"
    else:
        x_axis = np.arange(len(actual_prices))
        x_title = "Zaman Adımı"
    
    # Figure oluştur
    fig = go.Figure()
    
    # Gerçek fiyatlar
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=actual_prices,
        mode='lines',
        name='Gerçek Fiyat',
        line=dict(color='blue', width=2)
    ))
    
    # Tahmin edilen fiyatlar
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=predicted_prices,
        mode='lines',
        name='Tahmin Edilen Fiyat',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Layout ayarları
    fig.update_layout(
        title='Gerçek vs Tahmin Edilen Fiyatlar',
        xaxis_title=x_title,
        yaxis_title='Fiyat (₺)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_sentiment_scores(sentiment_df: pd.DataFrame) -> 'plotly.graph_objects.Figure':
    """
    Duygu skorlarını zaman serisinde görselleştirir.
    
    Args:
        sentiment_df: Duygu skorları DataFrame'i (date, positive, negative, neutral, compound)
    
    Returns:
        Plotly figure objesi
    """
    import plotly.graph_objects as go
    
    # Tarihe göre sırala
    sentiment_df = sentiment_df.sort_values('date')
    
    # Günlük ortalama hesapla (eğer aynı günde birden fazla haber varsa)
    daily_sentiment = sentiment_df.groupby('date').agg({
        'positive': 'mean',
        'negative': 'mean',
        'neutral': 'mean',
        'compound': 'mean'
    }).reset_index()
    
    # Figure oluştur
    fig = go.Figure()
    
    # Pozitif skorlar
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['positive'],
        mode='lines',
        name='Pozitif',
        line=dict(color='green', width=2)
    ))
    
    # Negatif skorlar
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['negative'],
        mode='lines',
        name='Negatif',
        line=dict(color='red', width=2)
    ))
    
    # Compound skorlar (ayrı bir eksen)
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['compound'],
        mode='lines',
        name='Compound',
        line=dict(color='purple', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    # Layout ayarları
    fig.update_layout(
        title='Günlük Duygu Skorları',
        xaxis_title='Tarih',
        yaxis_title='Pozitif/Negatif Skor',
        yaxis2=dict(
            title='Compound Skor',
            overlaying='y',
            side='right',
            range=[-1, 1]
        ),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> 'plotly.graph_objects.Figure':
    """
    Özellik önemlerini bar grafiği olarak görselleştirir.
    
    Args:
        importance_df: Özellik önemleri DataFrame'i (feature, importance sütunları)
        top_n: Gösterilecek en önemli özellik sayısı (varsayılan: 15)
    
    Returns:
        Plotly figure objesi
    """
    import plotly.graph_objects as go
    
    # En önemli N özelliği al
    top_features = importance_df.head(top_n)
    
    # Sıralamayı ters çevir (en önemli en üstte)
    top_features = top_features.iloc[::-1]
    
    # Figure oluştur
    fig = go.Figure()
    
    # Bar grafiği
    fig.add_trace(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(
            color=top_features['importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Önem')
        )
    ))
    
    # Layout ayarları
    fig.update_layout(
        title=f'En Önemli {top_n} Özellik (XGBoost)',
        xaxis_title='Önem Skoru',
        yaxis_title='Özellik',
        template='plotly_white',
        height=max(400, top_n * 30),  # Dinamik yükseklik
        showlegend=False
    )
    
    return fig


def display_model_performance():
    """
    Model performans sekmesini görüntüler.
    Model seçim dropdown'u, performans metrikleri, eğitim geçmişi grafiği,
    model karşılaştırma grafiği ve en iyi model gösterir.
    
    Requirements: 7.3, 7.4
    """
    from src.model_evaluation import (
        load_model_reports,
        select_best_model,
        plot_training_history,
        plot_metrics_comparison
    )
    
    st.header("📈 Model Performans Analizi")
    st.markdown("""
    Bu sekmede, eğitilmiş modellerin performans metriklerini, eğitim geçmişini ve 
    karşılaştırmalı analizlerini görüntüleyebilirsiniz.
    """)
    
    # Model raporlarının bulunduğu dizin
    reports_dir = 'models/training_history/'
    
    # Dizin yoksa oluştur ve uyarı göster
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir, exist_ok=True)
        st.warning("⚠️ Henüz kaydedilmiş model raporu yok. Lütfen önce 'Analiz' sekmesinden bir analiz çalıştırın.")
        st.info("💡 Model raporları otomatik olarak `models/training_history/` dizinine kaydedilir.")
        return
    
    # Model raporlarını yükle
    try:
        model_reports = load_model_reports(reports_dir)
    except Exception as e:
        st.error(f"❌ Model raporları yüklenirken hata oluştu: {e}")
        logger.error(f"Model raporları yükleme hatası: {e}")
        return
    
    # Rapor yoksa uyarı göster
    if not model_reports:
        st.warning("⚠️ Henüz kaydedilmiş model raporu yok. Lütfen önce 'Analiz' sekmesinden bir analiz çalıştırın.")
        st.info("💡 Model raporları otomatik olarak `models/training_history/` dizinine kaydedilir.")
        
        # Örnek bilgi göster
        with st.expander("ℹ️ Model Performans Takibi Hakkında"):
            st.markdown("""
            **Model performans takibi şunları içerir:**
            
            1. **Eğitim Metrikleri**: Her epoch'ta loss ve metrik değerleri
            2. **Performans Metrikleri**: MSE, RMSE, MAE, MAPE, R², Yönsel Doğruluk
            3. **Görselleştirmeler**: Eğitim/validasyon loss grafikleri, tahmin vs gerçek grafikler
            4. **Model Karşılaştırma**: Farklı model versiyonlarının karşılaştırılması
            5. **En İyi Model Seçimi**: Metrik bazlı otomatik model seçimi
            
            Bir analiz çalıştırdığınızda, model raporları otomatik olarak kaydedilir.
            """)
        return
    
    st.success(f"✅ {len(model_reports)} model raporu bulundu")
    
    # Model seçim dropdown'u
    st.subheader("🔍 Model Seçimi")
    model_names = list(model_reports.keys())
    
    selected_model = st.selectbox(
        "Görüntülemek istediğiniz modeli seçin:",
        options=model_names,
        index=0,
        help="Kaydedilmiş model raporlarından birini seçin"
    )
    
    if not selected_model:
        st.info("Lütfen bir model seçin.")
        return
    
    # Seçilen model raporu
    report = model_reports[selected_model]
    
    st.markdown("---")
    
    # Performans Metrikleri
    st.subheader("📊 Performans Metrikleri")
    
    if 'metrics' in report:
        metrics = report['metrics']
        
        # Metrikleri 3 sütunda göster
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="RMSE",
                value=f"{metrics.get('rmse', 0):.4f}",
                help="Root Mean Squared Error - Düşük değer daha iyidir"
            )
            st.metric(
                label="MAE",
                value=f"{metrics.get('mae', 0):.4f}",
                help="Mean Absolute Error - Düşük değer daha iyidir"
            )
        
        with col2:
            st.metric(
                label="MAPE",
                value=f"{metrics.get('mape', 0):.2f}%",
                help="Mean Absolute Percentage Error - Düşük değer daha iyidir"
            )
            st.metric(
                label="R² Score",
                value=f"{metrics.get('r2', 0):.4f}",
                help="R-squared Score - Yüksek değer daha iyidir (max: 1.0)"
            )
        
        with col3:
            st.metric(
                label="Yönsel Doğruluk",
                value=f"{metrics.get('directional_accuracy', 0):.2f}%",
                help="Fiyat yönünü doğru tahmin etme oranı"
            )
            st.metric(
                label="MSE",
                value=f"{metrics.get('mse', 0):.4f}",
                help="Mean Squared Error - Düşük değer daha iyidir"
            )
        
        # Metrik açıklamaları
        with st.expander("📖 Metrik Açıklamaları"):
            st.markdown("""
            - **RMSE (Root Mean Squared Error)**: Tahmin hatalarının karekök ortalaması. Büyük hatalara daha fazla ceza verir.
            - **MAE (Mean Absolute Error)**: Tahmin hatalarının mutlak değerlerinin ortalaması.
            - **MAPE (Mean Absolute Percentage Error)**: Yüzdesel hata oranı.
            - **R² Score**: Modelin varyansı açıklama oranı (0-1 arası, 1 mükemmel).
            - **Yönsel Doğruluk**: Fiyatın yukarı/aşağı yönünü doğru tahmin etme yüzdesi.
            - **MSE (Mean Squared Error)**: Tahmin hatalarının karelerinin ortalaması.
            """)
    else:
        st.warning("⚠️ Bu model için metrik bilgisi bulunamadı.")
    
    st.markdown("---")
    
    # Eğitim Geçmişi Grafiği
    st.subheader("📉 Eğitim Geçmişi")
    
    history_file = f"{reports_dir}{selected_model}_history.csv"
    
    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(history_file)
            
            # Eğitim geçmişi grafiğini oluştur
            fig_history = plot_training_history(history_df)
            st.pyplot(fig_history)
            
            # Eğitim detayları
            with st.expander("📋 Eğitim Detayları"):
                st.write(f"**Toplam Epoch:** {len(history_df)}")
                st.write(f"**Son Eğitim Loss:** {history_df['loss'].iloc[-1]:.4f}")
                st.write(f"**Son Validasyon Loss:** {history_df['val_loss'].iloc[-1]:.4f}")
                
                if 'mae' in history_df.columns:
                    st.write(f"**Son Eğitim MAE:** {history_df['mae'].iloc[-1]:.4f}")
                    st.write(f"**Son Validasyon MAE:** {history_df['val_mae'].iloc[-1]:.4f}")
                
                # En iyi epoch
                best_epoch = history_df['val_loss'].idxmin() + 1
                st.write(f"**En İyi Epoch:** {best_epoch} (en düşük validasyon loss)")
                
                # Overfitting kontrolü
                final_train_loss = history_df['loss'].iloc[-1]
                final_val_loss = history_df['val_loss'].iloc[-1]
                loss_diff = abs(final_val_loss - final_train_loss)
                
                if loss_diff > final_train_loss * 0.5:
                    st.warning("⚠️ Overfitting riski: Validasyon loss'u eğitim loss'undan önemli ölçüde yüksek.")
                else:
                    st.success("✅ Model iyi genelleme yapıyor.")
                
        except Exception as e:
            st.error(f"❌ Eğitim geçmişi grafiği oluşturulurken hata: {e}")
            logger.error(f"Eğitim geçmişi grafiği hatası: {e}")
    else:
        st.warning(f"⚠️ Eğitim geçmişi dosyası bulunamadı: {history_file}")
    
    st.markdown("---")
    
    # Model Karşılaştırma (birden fazla model varsa)
    if len(model_reports) > 1:
        st.subheader("🔄 Model Karşılaştırma")
        
        try:
            # Tüm modellerin metriklerini topla
            metrics_dict = {}
            for model_name, model_report in model_reports.items():
                if 'metrics' in model_report:
                    metrics_dict[model_name] = model_report['metrics']
            
            if len(metrics_dict) > 1:
                # Karşılaştırma grafiğini oluştur
                fig_comparison = plot_metrics_comparison(metrics_dict)
                st.pyplot(fig_comparison)
                
                # Karşılaştırma tablosu
                with st.expander("📊 Detaylı Karşılaştırma Tablosu"):
                    comparison_df = pd.DataFrame(metrics_dict).T
                    comparison_df = comparison_df.round(4)
                    st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("ℹ️ Karşılaştırma için en az 2 model gerekli.")
                
        except Exception as e:
            st.error(f"❌ Model karşılaştırma grafiği oluşturulurken hata: {e}")
            logger.error(f"Model karşılaştırma hatası: {e}")
        
        st.markdown("---")
        
        # En İyi Model Seçimi
        st.subheader("🏆 En İyi Model")
        
        try:
            # Farklı metrikler için en iyi modelleri bul
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**RMSE Bazlı (Düşük = İyi)**")
                best_rmse = select_best_model(metrics_dict, primary_metric='rmse', minimize=True)
                st.success(f"🥇 {best_rmse}")
                if best_rmse in metrics_dict:
                    st.write(f"RMSE: {metrics_dict[best_rmse]['rmse']:.4f}")
                
                st.markdown("**MAE Bazlı (Düşük = İyi)**")
                best_mae = select_best_model(metrics_dict, primary_metric='mae', minimize=True)
                st.success(f"🥇 {best_mae}")
                if best_mae in metrics_dict:
                    st.write(f"MAE: {metrics_dict[best_mae]['mae']:.4f}")
            
            with col2:
                st.markdown("**R² Bazlı (Yüksek = İyi)**")
                best_r2 = select_best_model(metrics_dict, primary_metric='r2', minimize=False)
                st.success(f"🥇 {best_r2}")
                if best_r2 in metrics_dict:
                    st.write(f"R²: {metrics_dict[best_r2]['r2']:.4f}")
                
                st.markdown("**Yönsel Doğruluk Bazlı (Yüksek = İyi)**")
                best_dir = select_best_model(metrics_dict, primary_metric='directional_accuracy', minimize=False)
                st.success(f"🥇 {best_dir}")
                if best_dir in metrics_dict:
                    st.write(f"Doğruluk: {metrics_dict[best_dir]['directional_accuracy']:.2f}%")
            
            # Genel öneri
            st.markdown("---")
            st.info(f"💡 **Genel Öneri:** Çoğu metrikte en iyi performansı gösteren model: **{best_rmse}**")
            
        except Exception as e:
            st.error(f"❌ En iyi model seçimi sırasında hata: {e}")
            logger.error(f"En iyi model seçimi hatası: {e}")
    
    else:
        st.info("ℹ️ Model karşılaştırma için birden fazla model raporu gerekli. Daha fazla analiz çalıştırarak farklı model versiyonları oluşturabilirsiniz.")
    
    # Model rapor bilgileri
    st.markdown("---")
    with st.expander("ℹ️ Model Rapor Bilgileri"):
        st.write(f"**Model Adı:** {selected_model}")
        if 'timestamp' in report:
            st.write(f"**Oluşturulma Zamanı:** {report['timestamp']}")
        st.write(f"**Rapor Dizini:** {reports_dir}")
        st.write(f"**Toplam Model Sayısı:** {len(model_reports)}")


if __name__ == "__main__":
    main()
