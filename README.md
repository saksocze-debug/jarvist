# MARK — 4D Holografik Gemini Asistanı

Tarayıcıda çalışan, sesli konuşan, 4D tesseract hologramlı asistan. API anahtarı sadece sunucuda durur.

## Render'a yayınlama
1. Bu klasörü GitHub reposuna yükle.
2. Render → New → Blueprint → repoyu seç (render.yaml otomatik okunur).
3. `GEMINI_API_KEY` değerini gir (https://aistudio.google.com/apikey).
4. Deploy bitince `https://mark-hologram.onrender.com` herkese açıktır.

İsteğe bağlı değişkenler: `GEMINI_MODEL` (varsayılan gemini-2.5-flash), `RATE_LIMIT_PER_MIN` (IP başına, varsayılan 15).

## Yerelde deneme
    pip install -r requirements.txt
    GEMINI_API_KEY=anahtarin uvicorn main:app --reload

Mikrofon (Web Speech API) en iyi Chrome/Edge/Safari'de çalışır ve HTTPS ister; Render HTTPS verir.
