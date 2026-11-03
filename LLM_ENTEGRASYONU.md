# 🤖 LLM Entegrasyonu (Groq)

## Genel Bakış

Proje, haber analizi ve LSTM tahmin açıklaması için **Groq API** kullanır.  
Model: `llama-3.3-70b-versatile` (ücretsiz, Türkçe desteği iyi)

---

## Kurulum

```bash
py -m pip install groq
```

---

## API Key Alma (Ücretsiz)

1. https://console.groq.com adresine git
2. "Create API Key" tıkla
3. Anahtarı kopyala (örn: `gsk_...`)

**Ücretsiz Limitler:**
- Dakikada 30 istek
- Günde 14.400 istek

---

## Kullanım

### Streamlit Arayüzü

1. `py -m streamlit run src/app.py` ile başlat
2. Sidebar → **Gelişmiş Ayarlar** → **Groq LLM Desteği**
3. Checkbox'ı işaretle
4. API key'ini gir
5. Parametreleri ayarla:
   - **LLM'e Gönderilecek Haber Sayısı**: 5–30 (varsayılan 20)
   - **BERT Ağırlığı**: 0.1–0.9 (varsayılan 0.5)
6. "Analiz Et" butonuna tıkla

### Programatik Kullanım

```python
from src.llm_analyzer import LLMAnalyzer
import pandas as pd

# İstemci başlat
llm = LLMAnalyzer(api_key="gsk_...")

# Tek haber analizi
result = llm.analyze_news(
    title="THYAO ilk çeyrekte rekor kar açıkladı",
    content="Türk Hava Yolları kargo gelirlerinde artış...",
    symbol="THYAO"
)
# {
#     'sentiment_score': 0.8,
#     'impact': 'yüksek',
#     'direction': 'pozitif',
#     'reasoning': 'Kar artışı hisse için olumlu sinyal.',
#     'time_horizon': 'kısa'
# }

# Toplu haber analizi
news_df = pd.DataFrame({
    'title': ['Haber 1', 'Haber 2'],
    'content': ['İçerik 1...', 'İçerik 2...'],
    'relevance_score': [0.8, 0.6]
})

analyzed = llm.analyze_news_batch(
    news_df,
    symbol='THYAO.IS',
    top_n=20,
    delay=2.0  # Groq rate limit için önerilen
)

# LSTM tahmin açıklaması
explanation = llm.explain_prediction(
    symbol='THYAO',
    lstm_change_pct=+2.5,
    features={
        'rsi_14': 65.3,
        'macd': 0.012,
        'ema_20': 150.2,
        'ema_50': 148.1,
        'sentiment_compound': 0.4
    }
)
```

---

## Analiz Akışı

```
[200 Haber] → BERT/Lexicon (tümü, hızlı)
                    ↓
           [relevance_score > 0.5]
                    ↓
       Top 20 Haber → Groq (derin analiz)
                    ↓
  Hibrit Skor = BERT_ağırlık × BERT + (1 - BERT_ağırlık) × Groq
                    ↓
           [LSTM Eğitimi]
                    ↓
    [LSTM Tahmin] → Groq Açıklaması
```

---

## LLM Çıktıları

`analyze_news_batch` sonrası eklenen sütunlar:

| Sütun | Açıklama |
|---|---|
| `llm_sentiment_score` | -1.0 ile 1.0 arası skor |
| `llm_impact` | yüksek / orta / düşük |
| `llm_direction` | pozitif / negatif / nötr |
| `llm_reasoning` | Türkçe kısa açıklama |
| `llm_time_horizon` | kısa / orta / uzun vade |
| `llm_analyzed` | True/False |

---

## Performans

| Aşama | Süre |
|---|---|
| 20 haber Groq analizi | ~15–20 saniye |
| 200 haber BERT analizi | 2–3 dakika (CPU) |
| LSTM tahmin açıklaması | ~2 saniye |

---

## Sorun Giderme

| Hata | Çözüm |
|---|---|
| `401 Unauthorized` | https://console.groq.com adresinden yeni key oluştur |
| `429 Rate Limit` | `top_n` düşür veya `delay` artır (örn: 2.0 → 4.0) |
| `groq module not found` | `py -m pip install groq` |

---

## Devre Dışı Bırakma

Groq'u kullanmak istemezsen sidebar'daki checkbox'ı işaretleme veya API key'i boş bırak.  
LSTM ve XGBoost normal şekilde çalışmaya devam eder.

---

## İlgili Dosyalar

- `src/llm_analyzer.py` — `LLMAnalyzer` sınıfı
- `src/app.py` — Groq UI ve analiz entegrasyonu
- `test_groq.py` — API key bağlantı test scripti
