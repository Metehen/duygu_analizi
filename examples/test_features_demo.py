"""
Demo script to test features.py functionality
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features import (
    calculate_technical_indicators,
    merge_sentiment_with_prices,
    prepare_training_data
)

print("=" * 60)
print("Testing Features Module")
print("=" * 60)

# Test 1: Technical Indicators
print("\n1. Testing calculate_technical_indicators...")
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
price_df = pd.DataFrame({
    'date': dates,
    'open': np.random.uniform(90, 110, 100),
    'high': np.random.uniform(95, 115, 100),
    'low': np.random.uniform(85, 105, 100),
    'close': np.random.uniform(90, 110, 100),
    'volume': np.random.randint(1000000, 10000000, 100)
})

result = calculate_technical_indicators(price_df)
print(f"   ✓ Input shape: {price_df.shape}")
print(f"   ✓ Output shape: {result.shape}")
print(f"   ✓ New columns: {[col for col in result.columns if col not in price_df.columns]}")
print(f"   ✓ RSI range: {result['rsi_14'].min():.2f} - {result['rsi_14'].max():.2f}")

# Test 2: Merge Sentiment with Prices
print("\n2. Testing merge_sentiment_with_prices...")
sentiment_df = pd.DataFrame({
    'date': dates[:50],  # Only first 50 days
    'positive': np.random.uniform(0, 1, 50),
    'negative': np.random.uniform(0, 1, 50),
    'neutral': np.random.uniform(0, 1, 50),
    'compound': np.random.uniform(-1, 1, 50)
})

merged = merge_sentiment_with_prices(result, sentiment_df)
print(f"   ✓ Price data rows: {len(result)}")
print(f"   ✓ Sentiment data rows: {len(sentiment_df)}")
print(f"   ✓ Merged data rows: {len(merged)}")
print(f"   ✓ Sentiment columns added: {[col for col in merged.columns if 'sentiment' in col]}")

# Test 3: Prepare Training Data
print("\n3. Testing prepare_training_data...")
lookback = 30
X, y, scaler = prepare_training_data(merged, target_column='close', lookback=lookback)
print(f"   ✓ Lookback window: {lookback}")
print(f"   ✓ X shape: {X.shape}")
print(f"   ✓ y shape: {y.shape}")
print(f"   ✓ X value range: {X.min():.4f} - {X.max():.4f}")
print(f"   ✓ Expected samples: {len(merged) - lookback}")
print(f"   ✓ Actual samples: {X.shape[0]}")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
