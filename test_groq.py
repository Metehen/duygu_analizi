"""
Groq API key testi.
Kullanım: py test_gemini.py

API key almak için: https://console.groq.com
"""

api_key = input("Groq API key'ini gir: ").strip()

if not api_key:
    print("❌ API key boş.")
    input("Çıkmak için Enter'a bas...")
    exit(1)

print("\nTest ediliyor...")

try:
    from groq import Groq
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Sadece 'Çalışıyor' yaz."}],
        max_tokens=10
    )
    yanit = response.choices[0].message.content.strip()
    print(f"✅ Bağlantı başarılı!")
    print(f"   Model yanıtı: {yanit}")
    print(f"\n✅ llm_analyzer.py hazır. Streamlit uygulamasında Groq API key'ini gir.")

except Exception as e:
    err = str(e)
    if "401" in err:
        print("❌ API key geçersiz. https://console.groq.com adresinden yeni key al.")
    elif "429" in err:
        print("⚠️  Rate limit. Birkaç saniye bekle ve tekrar dene.")
    elif "groq" in err.lower():
        print(f"❌ Kütüphane hatası: {e}")
        print("   Çözüm: py -m pip install groq")
    else:
        print(f"❌ Hata: {e}")

input("\nÇıkmak için Enter'a bas...")
