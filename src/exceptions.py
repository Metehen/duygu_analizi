# Copyright (C) 2026  Metehen
# Licensed under GNU AGPL v3. See LICENSE file for details.
"""
Proje genelinde kullanılan özel exception sınıfları.

Hiyerarşi:
    AppError (taban)
    ├── DataError          — veri yokluğu, geçersiz format
    │   ├── StockDataError — hisse fiyat verisi sorunları
    │   └── NewsDataError  — haber toplama sorunları
    ├── ModelError         — model eğitim/tahmin sorunları
    └── ConfigError        — yapılandırma sorunları

Kullanım:
    from src.exceptions import StockDataError, ModelError
    raise StockDataError("THYAO.IS için veri çekilemedi", symbol="THYAO.IS")

Streamlit'te yakalamak için:
    try:
        ...
    except StockDataError as e:
        st.error(e.user_message())
    except AppError as e:
        st.error(e.user_message())
"""


class AppError(Exception):
    """Tüm uygulama hatalarının taban sınıfı."""

    # Alt sınıflar bunu override ederek sabit öneri metni ekleyebilir
    _default_hint: str = ""

    def __init__(self, message: str, hint: str = "", **context):
        """
        Args:
            message:  Teknik hata mesajı (loglara gider)
            hint:     Kullanıcıya gösterilecek öneri (opsiyonel)
            **context: Ek bağlam bilgisi (loglara eklenir, örn: symbol='THYAO.IS')
        """
        super().__init__(message)
        self.message = message
        self.hint    = hint or self._default_hint
        self.context = context

    def user_message(self) -> str:
        """Streamlit'te gösterilecek kullanıcı dostu mesaj."""
        msg = self.message
        if self.hint:
            msg += f"\n\n💡 **İpucu:** {self.hint}"
        return msg

    def __str__(self) -> str:
        ctx = "  |  ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message}  [{ctx}]" if ctx else self.message


# ──────────────────────────────────────────────────────────
# Veri hataları
# ──────────────────────────────────────────────────────────

class DataError(AppError):
    """Genel veri hatası."""
    _default_hint = "Tarih aralığını veya sembolü kontrol edin."


class StockDataError(DataError):
    """Hisse fiyat verisi çekme/işleme hatası."""
    _default_hint = (
        "Hisse sembolünün '.IS' ile bittiğinden emin olun (örn: THYAO.IS). "
        "Yahoo Finance'e erişilebildiğini kontrol edin."
    )


class InsufficientDataError(DataError):
    """Model eğitimi için yeterli veri yok."""
    _default_hint = (
        "Daha geniş bir tarih aralığı seçin. "
        "LSTM lookback değerini düşürmeyi deneyin."
    )


class NewsDataError(DataError):
    """Haber toplama/işleme hatası."""
    _default_hint = (
        "RSS kaynaklarına erişim sağlanamıyor olabilir. "
        "En az bir kaynak seçili olduğundan emin olun."
    )


# ──────────────────────────────────────────────────────────
# Model hataları
# ──────────────────────────────────────────────────────────

class ModelError(AppError):
    """Genel model hatası."""
    _default_hint = "Epoch sayısını veya lookback değerini değiştirmeyi deneyin."


class ModelTrainingError(ModelError):
    """Model eğitim hatası."""
    _default_hint = (
        "Eğitim verisi çok az olabilir. "
        "Tarih aralığını genişletin veya lookback değerini küçültün."
    )


class ModelPredictionError(ModelError):
    """Model tahmin hatası."""
    _default_hint = "Model yeniden eğitilmeyi gerektirebilir."


# ──────────────────────────────────────────────────────────
# Yapılandırma hataları
# ──────────────────────────────────────────────────────────

class ConfigError(AppError):
    """Yapılandırma hatası."""
    _default_hint = "config.py dosyasını kontrol edin."
