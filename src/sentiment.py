"""
Duygu Analizi Modülü
Türkçe metinler için duygu analizi yapar.
BERT veya sözlük tabanlı yöntem kullanır.
"""

from src.logger import get_logger
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Türkçe duygu analizi için sınıf."""
    
    def __init__(self, method: str = 'bert', model_name: Optional[str] = None):
        """
        Duygu analizi sınıfını başlatır.
        
        Args:
            method: 'bert' veya 'lexicon'
            model_name: BERT model adı (opsiyonel, varsayılan config'den alınır)
        """
        self.method = method.lower()
        self.model = None
        self.tokenizer = None
        
        if self.method == 'bert':
            try:
                self._load_bert_model(model_name)
                logger.info(f"BERT modeli yüklendi: {model_name or 'default'}")
            except Exception as e:
                logger.warning(f"BERT modeli yüklenemedi: {e}. Sözlük tabanlı yönteme geçiliyor.")
                self.method = 'lexicon'
        
        if self.method == 'lexicon':
            self._load_lexicon()
            logger.info("Sözlük tabanlı yöntem kullanılıyor")
    
    def _load_bert_model(self, model_name: Optional[str] = None):
        """
        BERT modelini yükler.
        
        Args:
            model_name: Model adı (opsiyonel)
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # Config'den model adını al
            if model_name is None:
                from config import BERT_MODEL
                model_name = BERT_MODEL
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            
            # GPU varsa kullan
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
        except ImportError as e:
            raise ImportError(
                "transformers ve torch kütüphaneleri gerekli. "
                "Lütfen 'pip install transformers torch' komutunu çalıştırın."
            ) from e
    
    def _load_lexicon(self):
        """
        Sözlük tabanlı yöntem için pozitif/negatif kelime listelerini yükler.
        """
        # Basit Türkçe pozitif/negatif kelime listeleri
        self.positive_words = {
            'iyi', 'güzel', 'harika', 'mükemmel', 'başarılı', 'kazanç', 'artış',
            'yükseliş', 'büyüme', 'gelişme', 'olumlu', 'pozitif', 'karlı', 'avantaj',
            'fırsat', 'güçlü', 'sağlam', 'istikrarlı', 'rekor', 'yüksek', 'artan',
            'iyileşme', 'toparlanma', 'canlanma', 'patlama', 'sıçrama', 'yükseldi',
            'arttı', 'kazandı', 'başardı', 'geçti', 'aştı', 'ulaştı'
        }
        
        self.negative_words = {
            'kötü', 'berbat', 'başarısız', 'kayıp', 'düşüş', 'azalma', 'gerileme',
            'olumsuz', 'negatif', 'zararlı', 'dezavantaj', 'risk', 'tehlike', 'zayıf',
            'kırılgan', 'istikrarsız', 'düşük', 'azalan', 'kötüleşme', 'çöküş',
            'kriz', 'sorun', 'problem', 'sıkıntı', 'düştü', 'azaldı', 'kaybetti',
            'başarısız', 'geriledi', 'düşüşte', 'kaybında'
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Tek bir metin için duygu analizi yapar.
        
        Args:
            text: Analiz edilecek metin
        
        Returns:
            Dictionary: {'positive': float, 'negative': float, 'neutral': float, 'compound': float}
        """
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            # Boş metin için nötr skor döndür
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        try:
            if self.method == 'bert':
                return self._analyze_bert(text)
            else:
                return self._analyze_lexicon(text)
        except Exception as e:
            logger.error(f"Duygu analizi hatası: {e}")
            # Hata durumunda nötr skor döndür
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
    
    def _analyze_bert(self, text: str) -> Dict[str, float]:
        """
        BERT modeli ile duygu analizi yapar.
        
        Args:
            text: Analiz edilecek metin
        
        Returns:
            Duygu skorları
        """
        import torch
        
        # Metni tokenize et
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # GPU'ya taşı
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Tahmin yap
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Model çıktısına göre skorları hesapla
        # Çoğu BERT sentiment modeli 3 sınıf döndürür: [negative, neutral, positive]
        if len(probs) == 3:
            negative, neutral, positive = probs
        elif len(probs) == 2:
            # Binary classification: [negative, positive]
            negative, positive = probs
            neutral = 0.0
        else:
            # Beklenmeyen durum, nötr döndür
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        # Compound skoru hesapla (-1 ile 1 arası)
        compound = float(positive - negative)
        
        return {
            'positive': float(positive),
            'negative': float(negative),
            'neutral': float(neutral),
            'compound': compound
        }
    
    def _analyze_lexicon(self, text: str) -> Dict[str, float]:
        """
        Sözlük tabanlı yöntemle duygu analizi yapar.
        
        Args:
            text: Analiz edilecek metin
        
        Returns:
            Duygu skorları
        """
        # Metni küçük harfe çevir ve kelimelere ayır
        words = text.lower().split()
        
        if len(words) == 0:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        # Pozitif ve negatif kelime sayılarını say
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            # Hiç duygu kelimesi yoksa nötr
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        # Skorları normalize et
        positive_score = positive_count / len(words)
        negative_score = negative_count / len(words)
        neutral_score = 1.0 - (positive_score + negative_score)
        
        # Compound skoru hesapla
        compound = (positive_count - negative_count) / len(words)
        # -1 ile 1 arası sınırla
        compound = max(-1.0, min(1.0, compound * 5))  # 5 ile çarparak hassasiyeti artır
        
        return {
            'positive': float(positive_score),
            'negative': float(negative_score),
            'neutral': float(max(0.0, neutral_score)),
            'compound': float(compound)
        }
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Birden fazla metin için duygu analizi yapar.
        
        Args:
            texts: Metin listesi
        
        Returns:
            DataFrame with sentiment scores
        """
        results = []
        
        for text in texts:
            scores = self.analyze(text)
            results.append(scores)
        
        return pd.DataFrame(results)
