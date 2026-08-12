# 🤖 Gemini LLM Entegrasyonu

## Ne Eklendi?

Projeye **Google Gemini API** desteği eklendi. LSTM ile birlikte çalışarak:
1. En ilgili haberleri derin analiz eder
2. BERT/Lexicon skorlarıyla hibrit skor üretir
3. LSTM tahminini açıklar

---

## Kurulum

```bash
py -m pip install google-genai
```

✅ Kuruldu: `google-genai 2.10.0`

---

## API Key Alma (Ücretsiz)

1. https://aistudio.google.com adresine git
2. "Get API key" → "Create API key" tıkla
3. Anahtarı kopyala (örn: `AIzaSyC...`)

**Ücretsiz Limitler:**
- Dakikada 15 istek
- Günde 1500 istek

---

## Kullanım

### 1. Streamlit Arayüzü

1. `py -m streamlit run src/app.py` ile uygulamayı başlat
2. Ana menüden "Tahmin Uygulamasına Git"
3. Sidebar → **Gelişmiş Ayarlar**
4. "🤖 Gemini ile Gelişmiş Haber Analizi" checkbox'ını işaretle
5. API key'ini gir
6. Parametreleri ayarla:
   - **Gemini'ye Gönderilecek Haber Sayısı**: 5-30 (varsayılan 20)
   - **BERT Ağırlığı**: 0.1-0.9 (varsayılan 0.5)
7. "Analiz Et" butonuna tıkla

### 2. Programatik Kullanım

```python
from src.llm_analyzer import GeminiAnalyzer
import pandas as pd

# Gemini başlat
gemini = GeminiAnalyzer(api_key="AIzaSyC...")

# Tek haber analizi
result = gemini.analyze_news(
    title="THYAO ilk çeyrekte rekor kar açıkladı",
    content="Türk Hava Yolları kargo gelirlerinde artış...",
    symbol="THYAO"
)
print(result)
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

analyzed = gemini.analyze_news_batch(
    news_df,
    symbol='THYAO.IS',
    top_n=20,
    delay=1.0
)
print(analyzed[['title', 'llm_direction', 'llm_sentiment_score']])

# LSTM tahmin açıklaması
explanation = gemini.explain_prediction(
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
print(explanation)
# "LSTM modeli %2.5 yükseliş tahmin ediyor. RSI aşırı alım
#  bölgesine yaklaşsa da EMA20/50 yukarı kesiş devam ediyor..."
```

---

## Nasıl Çalışır?

### Analiz Akışı

```
[200 Haber] → BERT/Lexicon (tümü, hızlı)
                    ↓
           [relevance_score > 0.5]
                    ↓
         Top 20 Haber → Gemini (derin analiz, ~20-30 saniye)
                    ↓
    Hibrit Skor = 0.5×BERT + 0.5×Gemini
                    ↓
           [LSTM Eğitimi]
                    ↓
      [LSTM Tahmin] → Gemini Açıklaması
```

### Gemini'nin Avantajları

1. **Bağlam anlama**: "kargo geliri arttı" → THYAO'ya pozitif
2. **Nüans algılama**: "beklenenden düşük büyüme" → negatif (lexicon "büyüme" deyince pozitif der)
3. **Zaman ufku**: kısa/orta/uzun vade ayrımı
4. **Etki seviyesi**: yüksek/orta/düşük
5. **Açıklama**: LSTM tahminini Türkçe açıklar

### Rate Limit Yönetimi

- Top 20 haber = 20 istek
- Her istek arası 1 saniye bekleme
- Toplam süre: ~20-25 saniye
- Dakikada 15 istek limiti aşılmaz

---

## Çıktılar

### 1. Haber Analizi Sonuçları

Gemini'den gelen sütunlar (`news_df_llm`):
- `llm_sentiment_score`: -1.0 ile 1.0 arası skor
- `llm_impact`: yüksek/orta/düşük
- `llm_direction`: pozitif/negatif/nötr
- `llm_reasoning`: Türkçe açıklama (1 cümle)
- `llm_time_horizon`: kısa/orta/uzun vade
- `llm_analyzed`: True/False (LLM tarafından analiz edildi mi?)

### 2. Hibrit Duygu Skoru

BERT ve Gemini skorları ağırlıklı ortalama ile birleştirilir:

```python
compound_final = 0.5 × compound_bert + 0.5 × llm_sentiment_score
```

Kullanıcı ağırlıkları ayarlayabilir (0.1-0.9).

### 3. LSTM Tahmin Açıklaması

```
🤖 Gemini Tahmin Yorumu:
LSTM modeli +2.3% yükseliş öngörüyor. RSI 65 seviyesinde ve
EMA20/50 yukarı trendi devam ediyor. Son haberlerde pozitif duygu
hakim, ancak kısa vadede kar realizasyonu riski mevcut.
```

---

## Performans

| Aşama | Süre |
|---|---|
| 20 haber Gemini analizi | 20-30 saniye |
| 200 haber BERT analizi | 2-3 dakika (CPU) |
| Hibrit birleştirme | < 1 saniye |
| LSTM tahmin açıklaması | 2-3 saniye |

**Toplam ek süre**: ~25-35 saniye

---

## Sorun Giderme

### "API key geçersiz"
- https://aistudio.google.com adresinden yeni key oluştur
- Key'i kopyalarken başında/sonunda boşluk olmamalı

### "Rate limit aşıldı"
- `top_n` parametresini düşür (20 → 10)
- `delay` parametresini artır (1.0 → 2.0)

### "Gemini yanıt vermiyor"
- İnternet bağlantısını kontrol et
- Gemini API status: https://status.cloud.google.com

### "Protobuf uyarısı"
- Çalışmaya devam eder, görmezden gelebilirsin
- Düzeltmek için: `py -m pip install protobuf==4.25.9`

---

## Devre Dışı Bırakma

Gemini'yi kullanmak istemezsen:
- Streamlit'te checkbox'ı işaretleme
- Veya API key'i boş bırak

LSTM ve XGBoost normal şekilde çalışmaya devam eder.

---

## Dosya Değişiklikleri

**Yeni Dosyalar:**
- `src/llm_analyzer.py`: Gemini entegrasyon modülü
- `GEMINI_ENTEGRASYONU.md`: Bu dokümantasyon

**Güncellenen Dosyalar:**
- `src/app.py`: Sidebar UI + analiz akışına Gemini adımı eklendi
- `requirements.txt`: `google-genai>=1.0.0` eklendi

**Değişmeyen:**
- LSTM, XGBoost, BERT, veri toplama modülleri aynı

---

## Özet

✅ Kurulum: `py -m pip install google-genai`  
✅ API Key: https://aistudio.google.com (ücretsiz)  
✅ Kullanım: Streamlit'te checkbox işaretle + API key gir  
✅ Performans: +25-35 saniye ek süre  
✅ Avantaj: Daha doğru haber analizi + LSTM açıklaması  

**Sonuç**: LSTM sayısal gücünü koruyor, Gemini onu daha akıllı hale getiriyor! 🚀
