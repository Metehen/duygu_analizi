"""
Merkezi loglama modülü.

Özellikler:
- Hem konsola hem dosyaya yazar (app.log)
- RotatingFileHandler: max 5MB, 3 yedek dosya
- Her uygulama oturumuna benzersiz session_id atar
- Tüm modüller get_logger() ile buradan logger alır

Kullanım:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Mesaj")
"""

import logging
import os
import sys
import uuid
from logging.handlers import RotatingFileHandler
from datetime import datetime

# ──────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────
LOG_FILE    = "app.log"
MAX_BYTES   = 5 * 1024 * 1024   # 5 MB
BACKUP_COUNT = 3                 # app.log, app.log.1, app.log.2, app.log.3
LOG_FORMAT  = "%(asctime)s | %(session_id)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ──────────────────────────────────────────────
# Oturum ID — uygulama başına bir kez oluşturulur
# ──────────────────────────────────────────────
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


class _SessionFilter(logging.Filter):
    """Her log kaydına session_id alanını ekler."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = SESSION_ID
        return True


def _build_root_logger() -> None:
    """
    Kök logger'ı bir kez yapılandırır.
    İkinci çağrılarda handler eklenmez (idempotent).
    """
    root = logging.getLogger()

    # Zaten yapılandırıldıysa tekrar yapılandırma
    if getattr(root, "_app_configured", False):
        return

    root.setLevel(logging.DEBUG)  # Handler'lar kendi seviyelerini filtreler

    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    session_filter = _SessionFilter()

    # ── Konsol handler (INFO ve üzeri) ──────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    console_handler.addFilter(session_filter)

    # ── Dosya handler (DEBUG ve üzeri, rotate) ──
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(session_filter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root._app_configured = True  # type: ignore[attr-defined]

    root.info(
        "Loglama başlatıldı | session_id=%s | log_file=%s",
        SESSION_ID,
        os.path.abspath(LOG_FILE)
    )


def get_logger(name: str) -> logging.Logger:
    """
    Verilen isimde bir logger döndürür.
    İlk çağrıda kök logger'ı da yapılandırır.

    Args:
        name: Genellikle __name__ (modül adı)

    Returns:
        Yapılandırılmış logging.Logger instance'ı

    Örnek:
        logger = get_logger(__name__)
        logger.info("Haber toplama başladı")
        logger.warning("RSS kaynağı yanıt vermedi: %s", url)
        logger.error("Model eğitim hatası", exc_info=True)
    """
    _build_root_logger()
    return logging.getLogger(name)


def get_session_id() -> str:
    """Mevcut oturum ID'sini döndürür (UI'da göstermek için)."""
    return SESSION_ID
