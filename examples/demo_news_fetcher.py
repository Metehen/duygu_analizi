"""
Haber Toplama Modülü Demo

Bu script, news_fetcher modülünün temel kullanımını gösterir.
"""

import sys
import os

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from news_fetcher import fetch_rss_news, save_news, load_news


def main():
    """Ana demo fonksiyonu."""
    
    print("=" * 60)
    print("Haber Toplama Modülü Demo")
    print("=" * 60)
    
    # Örnek RSS feed URL'leri
    rss_urls = [
        'https://www.bloomberght.com/rss',
        'https://www.dunya.com/rss',
    ]
    
    print("\n1. RSS beslemelerinden haber çekiliyor...")
    print(f"   Kaynaklar: {len(rss_urls)} adet")
    
    # Haberleri çek
    news_df = fetch_rss_news(rss_urls, max_articles=50)
    
    print(f"\n   ✓ Toplam {len(news_df)} haber çekildi")
    
    if not news_df.empty:
        print("\n2. İlk 3 haber:")
        print("-" * 60)
        for idx, row in news_df.head(3).iterrows():
            print(f"\n   Başlık: {row['title'][:60]}...")
            print(f"   Kaynak: {row['source']}")
            print(f"   Tarih: {row['date']}")
            print(f"   İçerik: {row['content'][:100]}...")
        
        print("\n3. Haberleri kaydetme...")
        
        # CSV formatında kaydet
        csv_path = 'data/demo_news.csv'
        save_news(news_df, csv_path, format='csv')
        print(f"   ✓ CSV formatında kaydedildi: {csv_path}")
        
        # JSON formatında kaydet
        json_path = 'data/demo_news.json'
        save_news(news_df, json_path, format='json')
        print(f"   ✓ JSON formatında kaydedildi: {json_path}")
        
        print("\n4. Haberleri yükleme...")
        
        # CSV'den yükle
        loaded_csv = load_news(csv_path, format='csv')
        print(f"   ✓ CSV'den {len(loaded_csv)} haber yüklendi")
        
        # JSON'dan yükle
        loaded_json = load_news(json_path, format='json')
        print(f"   ✓ JSON'dan {len(loaded_json)} haber yüklendi")
        
        print("\n5. İstatistikler:")
        print("-" * 60)
        print(f"   Toplam haber sayısı: {len(news_df)}")
        print(f"   Benzersiz kaynak sayısı: {news_df['source'].nunique()}")
        print(f"   Tarih aralığı: {news_df['date'].min()} - {news_df['date'].max()}")
        print(f"   Ortalama başlık uzunluğu: {news_df['title'].str.len().mean():.0f} karakter")
        print(f"   Ortalama içerik uzunluğu: {news_df['content'].str.len().mean():.0f} karakter")
        
        print("\n6. Kaynaklara göre dağılım:")
        print("-" * 60)
        source_counts = news_df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   {source}: {count} haber")
    
    else:
        print("\n   ⚠ Hiç haber çekilemedi. RSS beslemeleri kontrol edilmeli.")
    
    print("\n" + "=" * 60)
    print("Demo tamamlandı!")
    print("=" * 60)


if __name__ == '__main__':
    main()
