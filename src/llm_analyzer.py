"""
LLM Analiz Modülü — Groq API
Haber analizi ve LSTM tahmin açıklaması için Groq API kullanır.
Model: llama-3.3-70b-versatile (ücretsiz, Türkçe desteği iyi)

API key: https://console.groq.com adresinden ücretsiz alınabilir.
Limitler: Dakikada 30 istek, günde 14.400 istek.
"""

import json
import re
import time
import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Kullanılacak model — Groq'un en iyi ücretsiz Türkçe modeli
GROQ_MODEL = "llama-3.3-70b-versatile"


class LLMAnalyzer:
    """Groq API ile haber analizi ve tahmin açıklaması."""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: https://console.groq.com adresinden alınan API key
        """
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            self._model = GROQ_MODEL
            logger.info(f"Groq istemcisi başlatıldı. Model: {self._model}")
        except ImportError:
            raise ImportError(
                "groq kütüphanesi gerekli. "
                "Lütfen 'py -m pip install groq' komutunu çalıştırın."
            )

    def _generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Groq API'ye istek gönderir, ham metin döndürür."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1  # Düşük temperature = tutarlı çıktı
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Tek haber analizi
    # ------------------------------------------------------------------

    def analyze_news(self, title: str, content: str, symbol: str) -> Dict:
        """
        Tek bir haberi Groq LLM ile analiz eder.

        Args:
            title:   Haber başlığı
            content: Haber içeriği (ilk 400 karakter kullanılır)
            symbol:  Hisse kodu (örn: THYAO)

        Returns:
            {
                'sentiment_score': float  — -1.0 ile 1.0 arası,
                'impact':          str    — 'yüksek' | 'orta' | 'düşük',
                'direction':       str    — 'pozitif' | 'negatif' | 'nötr',
                'reasoning':       str    — kısa Türkçe açıklama,
                'time_horizon':    str    — 'kısa' | 'orta' | 'uzun'
            }
        """
        prompt = (
            f"Aşağıdaki Türkçe ekonomi haberini {symbol} hisse senedi açısından analiz et.\n\n"
            f"Başlık: {title}\n"
            f"İçerik: {content[:400]}\n\n"
            "Yalnızca aşağıdaki JSON formatında yanıt ver, başka hiçbir şey yazma:\n"
            '{"sentiment_score": -1.0 ile 1.0 arası sayı, '
            '"impact": "yüksek" veya "orta" veya "düşük", '
            '"direction": "pozitif" veya "negatif" veya "nötr", '
            '"reasoning": "maksimum 1 cümle Türkçe açıklama", '
            '"time_horizon": "kısa" veya "orta" veya "uzun"}'
        )
        try:
            text = self._generate(prompt)
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return self._neutral_result()
        except Exception as e:
            logger.warning(f"Groq haber analiz hatası: {e}")
            return self._neutral_result()

    # ------------------------------------------------------------------
    # Toplu haber analizi
    # ------------------------------------------------------------------

    def analyze_news_batch(
        self,
        news_df: pd.DataFrame,
        symbol: str,
        top_n: int = 20,
        delay: float = 2.0
    ) -> pd.DataFrame:
        """
        En yüksek relevance skorlu haberleri LLM ile analiz eder.
        Rate limit koruması için istekler arasında bekleme yapılır.

        Args:
            news_df:  Haber DataFrame'i
            symbol:   Hisse kodu (örn: THYAO.IS)
            top_n:    LLM'e gönderilecek maksimum haber sayısı
            delay:    İstekler arası bekleme süresi saniye (Groq: 2sn önerilir)

        Returns:
            news_df'e şu sütunlar eklenmiş DataFrame:
            llm_sentiment_score, llm_impact, llm_direction,
            llm_reasoning, llm_time_horizon, llm_analyzed
        """
        result_df = news_df.copy()
        result_df['llm_sentiment_score'] = 0.0
        result_df['llm_impact']          = 'düşük'
        result_df['llm_direction']       = 'nötr'
        result_df['llm_reasoning']       = ''
        result_df['llm_time_horizon']    = 'kısa'
        result_df['llm_analyzed']        = False

        if 'relevance_score' in news_df.columns:
            top_indices = news_df['relevance_score'].nlargest(top_n).index
        else:
            top_indices = news_df.head(top_n).index

        code  = symbol.replace('.IS', '').upper()
        total = len(top_indices)
        logger.info(f"Groq ile {total} haber analiz edilecek ({code})")

        for i, idx in enumerate(top_indices):
            row     = news_df.loc[idx]
            title   = str(row.get('title', ''))
            content = str(row.get('content', ''))

            result = self.analyze_news(title, content, code)

            result_df.at[idx, 'llm_sentiment_score'] = result.get('sentiment_score', 0.0)
            result_df.at[idx, 'llm_impact']          = result.get('impact', 'düşük')
            result_df.at[idx, 'llm_direction']        = result.get('direction', 'nötr')
            result_df.at[idx, 'llm_reasoning']        = result.get('reasoning', '')
            result_df.at[idx, 'llm_time_horizon']     = result.get('time_horizon', 'kısa')
            result_df.at[idx, 'llm_analyzed']         = True

            logger.debug(
                f"[{i+1}/{total}] '{title[:50]}' → "
                f"{result.get('direction')} ({result.get('sentiment_score', 0):.2f})"
            )

            if i < total - 1:
                time.sleep(delay)

        analyzed = int(result_df['llm_analyzed'].sum())
        logger.info(f"Groq analizi tamamlandı: {analyzed}/{len(result_df)} haber")
        return result_df

    # ------------------------------------------------------------------
    # LSTM tahmin açıklaması
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        symbol: str,
        lstm_change_pct: float,
        features: Dict
    ) -> str:
        """
        LSTM tahminini teknik göstergeler bağlamında LLM ile açıklar.

        Args:
            symbol:          Hisse kodu
            lstm_change_pct: LSTM'in tahmin ettiği yüzdelik değişim
            features:        Son gün teknik göstergeler dict'i

        Returns:
            Türkçe açıklama metni
        """
        trend         = "yukarı trend" if features.get('ema_20', 0) > features.get('ema_50', 0) else "aşağı trend"
        direction_str = "yükseliş"     if lstm_change_pct >= 0 else "düşüş"

        prompt = (
            f"{symbol} hisse senedi için LSTM modeli şu tahmini üretti:\n"
            f"Beklenen fiyat değişimi: {lstm_change_pct:+.2f}% ({direction_str})\n\n"
            "Güncel teknik göstergeler:\n"
            f"- RSI (14): {features.get('rsi_14', 'N/A')}\n"
            f"- MACD: {features.get('macd', 'N/A')}\n"
            f"- EMA20/EMA50 durumu: {trend}\n"
            f"- Son duygu skoru (haber bazlı): {features.get('sentiment_compound', 'N/A')}\n\n"
            "Bu tahmini teknik analiz açısından 2-3 cümleyle Türkçe olarak açıkla. "
            "Yatırım tavsiyesi verme, sadece teknik durumu yorumla."
        )
        try:
            return self._generate(prompt, max_tokens=200)
        except Exception as e:
            logger.warning(f"Groq açıklama hatası: {e}")
            return (
                f"LSTM modeli {lstm_change_pct:+.2f}% "
                f"{'yükseliş' if lstm_change_pct >= 0 else 'düşüş'} tahmin etti."
            )

    # ------------------------------------------------------------------
    # Hibrit skor birleştirme (statik — API gerektirmez)
    # ------------------------------------------------------------------

    @staticmethod
    def merge_with_bert_scores(
        sentiment_df: pd.DataFrame,
        news_df_with_llm: pd.DataFrame,
        bert_weight: float = 0.5,
        llm_weight: float  = 0.5
    ) -> pd.DataFrame:
        """
        BERT/Lexicon compound skorlarını LLM skorlarıyla ağırlıklı birleştirir.
        LLM tarafından analiz edilmemiş satırlarda sadece BERT skoru kullanılır.
        """
        merged = sentiment_df.copy()

        if 'llm_sentiment_score' not in news_df_with_llm.columns:
            return merged

        llm_scores   = news_df_with_llm['llm_sentiment_score'].values
        llm_analyzed = (
            news_df_with_llm['llm_analyzed'].values
            if 'llm_analyzed' in news_df_with_llm.columns
            else [True] * len(news_df_with_llm)
        )

        bert_compounds = merged['compound'].values.copy()
        hybrid = [
            bert_weight * b + llm_weight * float(llm_scores[i])
            if i < len(llm_scores) and llm_analyzed[i] else b
            for i, b in enumerate(bert_compounds)
        ]

        merged['compound_bert'] = bert_compounds
        merged['compound']      = hybrid
        return merged

    @staticmethod
    def _neutral_result() -> Dict:
        return {
            'sentiment_score': 0.0,
            'impact':          'düşük',
            'direction':       'nötr',
            'reasoning':       'Analiz yapılamadı.',
            'time_horizon':    'kısa'
        }


# Geriye dönük uyumluluk — eski GeminiAnalyzer adını da çalıştır
GeminiAnalyzer = LLMAnalyzer
