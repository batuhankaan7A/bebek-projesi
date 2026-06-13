# BebeSes — Yapay Zeka Bebek Ses Monitörü

> Ebeveyn kontrol paneli · WCAG 2.1 AA · n8n entegrasyonu · FastAPI backend

---

## Proje Mimarisi

```
┌─────────────────────┐     HTTPS POST /predict     ┌──────────────────────┐
│   Frontend          │ ─────────────────────────►  │   Backend (FastAPI)  │
│   Netlify / Vercel  │ ◄─────────────────────────  │   Render / Railway   │
│   index.html        │     { label, confidence }   │   main.py            │
└─────────────────────┘                             └──────────────────────┘
         │                                                    │
         │ Webhook POST (ağlama tespit edilince)              │ uyku_bebek.h5
         ▼                                                    │ uyku_bebek_classes.npy
┌─────────────────────┐
│   n8n               │
│   (self-host / cloud)│
│   → Telegram        │
│   → WhatsApp        │
│   → E-posta         │
└─────────────────────┘
```

---

## Hızlı Başlangıç

### 1. Frontend — Netlify Deploy

```bash
# Repo'ya index.html ve netlify.toml yükle
git add index.html netlify.toml
git commit -m "feat: ebeveyn paneli"
git push

# Netlify'da:
# New site → Import from GitHub → publish directory: .  → Deploy
```

### 2. Backend — Render.com Deploy

```bash
# backend/ klasörünü ayrı bir repoya koy
cd backend/
git init && git add . && git commit -m "init"
git remote add origin https://github.com/kullanici/bebesebes-api.git
git push -u origin main

# Render.com:
# New Web Service → Connect repo → Runtime: Python
# Build: pip install -r requirements.txt
# Start: uvicorn main:app --host 0.0.0.0 --port $PORT
# Ortam değişkeni: ALLOWED_ORIGINS=https://bebesebes.netlify.app
```

**ÖNEMLİ:** `uyku_bebek.h5` ve `uyku_bebek_classes.npy` dosyalarını `backend/` klasörüne kopyala!

### 3. Frontend'e Backend URL'si Ekle

`index.html` → Ayarlar sayfası → Backend API URL alanına Render URL'sini yapıştır:
```
https://bebesebes-api.onrender.com
```

---

## n8n Kurulumu (Bildirim Sistemi)

### Gerekli Akış:
1. **Webhook** nodu → URL'yi kopyala → panele yapıştır
2. **IF** nodu → `body.event === "cry"` kontrolü  
3. **Switch** nodu:  
   - `cry` / `pain` → Telegram Bot (acil)  
   - `hungry` → WhatsApp (uyarı)  
   - `test` → sadece loga yaz  
4. **Telegram** nodu → Bot Token + Chat ID

### n8n'e gelen payload örneği:
```json
{
  "baby_name": "Ayşe",
  "event": "cry",
  "label": "Genel Ağlama",
  "confidence": 87,
  "timestamp": "2026-06-13T14:32:00Z",
  "source": "bebesebes_dashboard"
}
```

---

## WCAG 2.1 AA Uyumluluk Kontrol Listesi

| Kriter | Durum | Açıklama |
|--------|-------|----------|
| 1.1.1 Non-text Content | ✅ | Tüm ikonlar aria-hidden, fonksiyonel elemanlar etiketli |
| 1.3.1 Info & Relationships | ✅ | Semantik HTML5 (header, nav, main, footer) |
| 1.4.3 Contrast (Minimum) | ✅ | Tüm metin/arka plan oranı ≥ 4.5:1 |
| 1.4.4 Resize Text | ✅ | rem birimleri, responsive |
| 2.1.1 Keyboard | ✅ | Tüm interaktif elemanlar klavyeyle erişilebilir |
| 2.4.1 Bypass Blocks | ✅ | Skip link mevcut |
| 2.4.3 Focus Order | ✅ | Mantıksal tab sırası |
| 2.4.7 Focus Visible | ✅ | :focus-visible outline tüm elemanlarda |
| 3.1.1 Language of Page | ✅ | `lang="tr"` |
| 4.1.2 Name Role Value | ✅ | ARIA role/label/pressed/current |
| 4.1.3 Status Messages | ✅ | aria-live bölgeleri (polite + assertive) |

---

## Klasör Yapısı

```
bebek-monitor/
├── index.html          ← Frontend (Netlify'a deploy et)
├── netlify.toml        ← Netlify yapılandırması
├── README.md
└── backend/
    ├── main.py         ← FastAPI sunucu
    ├── requirements.txt
    └── render.yaml     ← Render yapılandırması
    [aynı klasöre koy:]
    ├── uyku_bebek.h5
    └── uyku_bebek_classes.npy
```
