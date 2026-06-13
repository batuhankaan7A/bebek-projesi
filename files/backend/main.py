"""
BebeSes – FastAPI Backend
Deploy: Render.com veya Railway.app
Model: bebek_uyku_modeli.h5  |  Etiketler: classes.npy
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import librosa
import os
import tempfile
from datetime import datetime

# ─── TENSORFLOW VE YAMNET YÜKLEME ──────────────────────
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    
    # 1. Kendi Eğittiğin Modeli Yükle
    MODEL_PATH  = os.getenv("MODEL_PATH", "bebek_uyku_modeli.h5")
    LABELS_PATH = os.getenv("LABELS_PATH", "classes.npy")
    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = list(np.load(LABELS_PATH, allow_pickle=True))
    
    # 2. YAMNet Özellik Çıkarıcıyı Yükle (test_et.py'deki gibi)
    print("YAMNet Özellik Çıkarıcı Yükleniyor...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    MODEL_LOADED = True
    print(f"✅ Model yüklendi: {MODEL_PATH}")
    print(f"✅ Sınıflar: {labels}")
except Exception as e:
    print(f"⚠️  Model yüklenemedi ({e}). Mock mod aktif.")
    MODEL_LOADED = False
    labels = ["calm", "sleep", "hungry", "cry", "pain"]

# ─── APP ─────────────────────────────────────────────
app = FastAPI(
    title="BebeSes API",
    description="Bebek ses analiz API'si — YAMNet + bebek_uyku_modeli.h5",
    version="1.0.0"
)

# CORS Ayarları
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── YARDIMCI FONKSİYONLAR ────────────────────────────
def extract_yamnet_features(file_path: str) -> np.ndarray:
    """YAMNet ile ses dosyasından özellikleri çıkar (test_et.py formatı)"""
    # Ses tam olarak YAMNet'in beklediği gibi 16000 Hz ve Mono yüklenir
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # YAMNet'e gönderip embedding'leri al
    scores, embeddings, spectrogram = yamnet_model(y)
    
    # Embedding'lerin ortalamasını alarak kendi modelimize (1, 1024) formatında ver
    features = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)
    return features


def mock_predict() -> dict:
    """Model yokken gerçekçi rastgele tahmin üret (geliştirme modu)."""
    import random
    weights = [0.35, 0.25, 0.18, 0.14, 0.08]
    cls = random.choices(labels, weights=weights[:len(labels)])[0]
    confidence = round(random.uniform(0.62, 0.97), 3)

    raw = np.random.dirichlet(np.ones(len(labels)) * 0.5)
    raw[labels.index(cls)] = confidence
    raw = raw / raw.sum()
    probs = {l: round(float(v), 4) for l, v in zip(labels, raw)}

    return {
        "label": cls,
        "confidence": confidence,
        "probabilities": probs,
        "model": "mock",
        "timestamp": datetime.utcnow().isoformat()
    }


# ─── ROUTE'LAR ────────────────────────────────────────
@app.get("/", tags=["Genel"])
def root():
    return {
        "service": "BebeSes API",
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "classes": labels
    }


@app.get("/health", tags=["Genel"])
def health():
    return {"status": "ok", "model": MODEL_LOADED, "time": datetime.utcnow().isoformat()}


# ÖNEMLİ: Ön yüz sesi 'ses_dosyasi' adıyla gönderiyor, burası da 'ses_dosyasi' almalı.
@app.post("/predict/audio", tags=["Tahmin"])
async def predict_audio(ses_dosyasi: UploadFile = File(...)):
    contents = await ses_dosyasi.read()
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="Ses dosyası çok küçük veya boş.")

    if not MODEL_LOADED:
        return mock_predict()

    # Frontend'den gelen webm/wav dosyasını güvenli okumak için geçici dosyaya yazıyoruz
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(contents)
        tmp_path = temp_audio.name

    try:
        # 1. YAMNet Özellik Çıkarımı
        features = extract_yamnet_features(tmp_path)
        
        # 2. Model Tahmini
        preds    = model.predict(features, verbose=0)[0]
        idx      = int(np.argmax(preds))
        
        # İşlem bitince geçici dosyayı temizle
        os.remove(tmp_path)
        
        return {
            "label":         labels[idx],
            "confidence":    round(float(preds[idx]), 4),
            "probabilities": {l: round(float(p), 4) for l, p in zip(labels, preds)},
            "model":         "bebek_uyku_modeli.h5",
            "timestamp":     datetime.utcnow().isoformat()
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")


@app.get("/classes", tags=["Model"])
def get_classes():
    return {"classes": labels, "count": len(labels)}


# ─── ÇALIŞTIRMA ────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
