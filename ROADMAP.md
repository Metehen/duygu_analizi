# 🗺️ Proje Yol Haritası

Türkçe Ekonomi Haberleri Duygu Analizi & Hisse Tahmin Sistemi

> Durum ikonları: ✅ Tamamlandı · 🔄 Devam Ediyor · 📋 Planlandı · 💡 Fikir

---

## ✅ Tamamlananlar

- [x] RSS beslemelerinden haber toplama
- [x] Türkçe BERT ve lexicon tabanlı duygu analizi
- [x] Teknik göstergeler (EMA, RSI, MACD)
- [x] LSTM fiyat tahmin modeli
- [x] XGBoost yön tahmin modeli
- [x] Basit geriye dönük test (backtesting)
- [x] Streamlit web arayüzü
- [x] Hisse özel haber filtreleme
- [x] RSS kaynak seçimi (sektör bazlı)
- [x] Model kaydetme / önbellekten yükleme
- [x] Groq LLM entegrasyonu (haber analizi + tahmin açıklaması)
- [x] Model performans takip sekmesi
- [x] Haber okuma sayfası (hisse bazlı + genel piyasa)
- [x] Kapsamlı test suite (pytest + hypothesis)
- [x] requirements.txt sabit versiyonlara dönüştürüldü + requirements-dev.txt oluşturuldu
- [x] Merkezi loglama: dosyaya yazar (app.log), rotasyon (5MB/3 yedek), oturum ID
- [x] Özel exception sınıfları (StockDataError, InsufficientDataError vb.) + modül seviyesi girdi doğrulama
- [x] UX: analiz tamamlandığında st.toast bildirimi, tüm parametreler session state ile hatırlanıyor
- [x] Haber-fiyat gecikme analizi (cross-correlation, otomatik + manuel lag, sidebar kontrolü)
- [x] LSTM Monte Carlo Dropout güven aralığı (`predict_lstm_with_uncertainty`, %90 CI band, CV uyarısı)
- [x] Yeni teknik göstergeler: Bollinger Bands×5, Stochastic×2, ATR×2, OBV — LSTM özellik sayısı 7→19'a çıktı
- [x] Model versiyonlama — her eğitilen modelin metadata'sı (parametreler, lag, özellikler, CV) JSON'a kaydediliyor, UI'da gösteriliyor, model silme arayüzü eklendi
- [x] Veri kalıcılığı — SQLite (`src/database.py`) ile haber/fiyat önbelleği, sidebar'da süre ayarı, analize otomatik entegre. Analiz sonuçları kaydediliyor, geçmiş analiz karşılaştırma tablosu + metrik trendi grafiği

---

## 📋 Aşama 1 — Temel Kalite (Kısa Vadeli)

### 1.1 Hata Yönetimi
- [x] Kritik hataları (veri yok, model çöktü) ile uyarıları ayırt et
- [x] Her hata için kullanıcıya ne yapması gerektiğini söyle
- [x] Modül seviyesinde girdi doğrulama (`fetch_stock_data` boş dönünce erken çık)

### 1.2 Loglama
- [x] Log çıktısını `app.log` dosyasına yaz
- [x] Log rotasyonu ekle (5MB, 3 yedek dosya)
- [x] Oturum bazlı log ID'si (hangi analiz ne zaman çalıştı)

### 1.3 Bağımlılık Sabitleme
- [x] `requirements.txt` açık aralıklarını (`>=`) sabit versiyonlara (`==`) dönüştür
- [x] `requirements-dev.txt` oluştur (pytest, hypothesis sadece geliştirme için)

### 1.4 Küçük UX İyileştirmeleri
- [x] Analiz tamamlandığında tarayıcı sekmesine bildirim (st.toast)
- [x] Son kullanılan parametreleri session'da hatırla (sayfa yenilenince sıfırlanmasın)
- [ ] Uzun analizlerde kalan süre tahmini göster

---

## 📋 Aşama 2 — Model Kalitesi (Orta Vadeli)

### 2.1 Haber-Fiyat Gecikme Analizi ⭐
- [x] Cross-correlation analizi ile optimal gecikme (lag) bul
- [x] Şu an sabit 0 olan lag değerini otomatik hesapla
- [x] Sidebar'da lag parametresini göster ve manuel ayarlamaya izin ver

### 2.2 LSTM Tahmin Güveni
- [x] Monte Carlo Dropout ile belirsizlik aralığı hesapla
- [x] Grafikte güven bandı göster (%90 CI, şeffaf alan)
- [x] Düşük güvenli tahminleri kullanıcıya uyar (CV > %5)

### 2.3 Otomatik Özellik Seçimi ⚠️ Düşük Öncelik
- [ ] ~~XGBoost feature importance sonucuna göre düşük önemli özellikleri filtrele~~
- [ ] ~~LSTM giriş boyutunu otomatik küçült~~
- [ ] ~~Eğitim hızı ve doğruluk üzerindeki etkiyi logla~~

> **Not:** XGBoost ve LSTM farklı görevler yapıyor (yön vs fiyat), önem sıralamaları örtüşmeyebilir.
> Mevcut ~15 özellik zaten küçük, darboğaz değil. Riskli mimarisel değişiklik gerektirir. Atlandı.

### 2.4 Ensemble Model ⛔ Atlandı
> **Karar:** Projenin asıl odağı LSTM fiyat tahmini. XGBoost yardımcı analiz aracı olarak kalır
> ama tahmin ağırlığı %100 LSTM'de olacak. Ensemble karmaşıklık ekler, kazanım belirsiz.

### 2.5 Daha Fazla Teknik Gösterge
- [x] Bollinger Bands (upper, middle, lower, width, pct_b)
- [x] Stochastic Oscillator (%K, %D)
- [x] ATR (Average True Range) + ATR% — volatilite ölçümü
- [x] OBV (On-Balance Volume) + log normalize

---

## 📋 Aşama 3 — Profesyonel Altyapı (Uzun Vadeli)

### 3.1 Model Versiyonlama
- [x] Her eğitilen modelin metadata'sını kaydet (parametreler, veri aralığı, özellikler)
- [x] Model karşılaştırma tablosunu genişlet
- [x] Model silme / arşivleme arayüzü

### 3.2 Portföy Analizi
- [ ] Tek hisse yerine birden fazla hisse girebilme
- [ ] Portföy genelinde duygu skoru özeti
- [ ] "En olumlu görünen hisse" önerisi
- [ ] Hisseler arası korelasyon matrisi

### 3.3 Zamanlayıcı ve Otomatik Analiz
- [ ] APScheduler ile belirli saatlerde otomatik analiz
- [ ] "Son güncelleme: 09:00" bilgisini arayüzde göster
- [ ] Önemli haber geldiğinde (yüksek sentiment değişimi) uyarı

### 3.4 Veri Kalıcılığı
- [x] Çekilen haberleri SQLite'a kaydet (tekrar tekrar RSS çekmeyi azalt)
- [x] Fiyat verilerini cache'le (Yahoo Finance istek limitini aşmamak için)
- [x] Geçmiş analizleri karşılaştırabilmek için sonuçları sakla

### 3.5 REST API Katmanı (FastAPI)
- [ ] `/analyze` — hisse analizi başlat
- [ ] `/predict` — tahmin döndür
- [ ] `/sentiment` — duygu skoru döndür
- [ ] `/news` — hisse haberlerini döndür
- [ ] Streamlit ile paralel çalışabilir mimari

---

## 💡 Gelecek Fikirler (Kesin Değil)

### Veri Kaynakları
- [ ] KAP (Kamuyu Aydınlatma Platformu) entegrasyonu — resmi şirket açıklamaları
- [ ] Twitter/X API — sosyal medya duygu analizi
- [ ] Döviz kuru ve faiz verisi (TCMB API)
- [ ] Altın fiyatı korelasyonu

### Model İyileştirmeleri
- [ ] Transformer tabanlı zaman serisi modeli (TFT — Temporal Fusion Transformer)
- [ ] Piyasa rejimi tespiti (bull/bear/sideways) — farklı rejimlerde farklı model
- [ ] Anomali tespiti — olağandışı fiyat hareketlerini önceden işaretle

### Kullanıcı Deneyimi
- [ ] Kullanıcı hesabı ve kayıtlı portföy
- [ ] E-posta / Telegram bildirimi
- [ ] Mobil uyumlu arayüz
- [ ] Dark mode

### Gerçek Zamanlılık
- [ ] WebSocket ile canlı fiyat akışı
- [ ] Anlık RSS polling (5 dakikada bir)
- [ ] "Piyasa açık / kapalı" durumu göstergesi

---

## 📊 Başarı Metrikleri

Projenin ne kadar ilerlediğini ölçmek için:

| Metrik | Şu An | Hedef |
|---|---|---|
| LSTM Yönsel Doğruluk | ~55% | >60% |
| Haber Gecikme Optimizasyonu | Yok | Uygulandı |
| Test Coverage | ~60% | >80% |
| Desteklenen Hisse Sayısı | 5 (varsayılan) | Sınırsız |
| API Endpoint | Yok | 4+ endpoint |
| Ortalama Analiz Süresi | ~5-8 dk | <3 dk |

---

## 📝 Katkı Notları

Yeni bir özellik eklerken:
1. Önce bu dosyaya ekle (💡 fikir olarak)
2. Geliştirmeye başlarken 📋 planlandı yap
3. Tamamlanınca ✅ işaretle ve "Tamamlananlar" bölümüne taşı

---

*Son güncelleme: Eylül 2026 — Aşama 1+2 tamamlandı, Aşama 3.1+3.4 tamamlandı*
