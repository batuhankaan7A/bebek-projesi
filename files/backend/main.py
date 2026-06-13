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
import io
import os
import json
from datetime import datetime

# TensorFlow/Keras import (opsiyonel - model yoksa mock döner)
try:
    import tensorflow as tf
    MODEL_PATH  = os.getenv("MODEL_PATH", "bebek_uyku_modeli.h5")
    LABELS_PATH = os.getenv("LABELS_PATH", "classes.npy")
    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = list(np.load(LABELS_PATH, allow_pickle=True))
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
    description="Bebek ses analiz API'si — bebek_uyku_modeli.h5 modeli",
    version="1.0.0"
)

# CORS: Netlify/Vercel frontend domain'ini buraya ekle
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── YARDIMCI FONKSİYONLAR ────────────────────────────
SAMPLE_RATE = 22050
DURATION    = 2       # saniye
N_MFCC      = 40


def extract_features(audio_bytes: bytes) -> np.ndarray:
    """Ham ses bytes'ından MFCC özellikleri çıkar."""
    audio_io = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_io, sr=SAMPLE_RATE, duration=DURATION)
    # Pad/trim
    target = SAMPLE_RATE * DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)


def mock_predict() -> dict:
    """Model yokken gerçekçi rastgele tahmin üret (geliştirme modu)."""
    import random
    weights = [0.35, 0.25, 0.18, 0.14, 0.08]
    cls = random.choices(labels, weights=weights[:len(labels)])[0]
    confidence = round(random.uniform(0.62, 0.97), 3)

    # Tüm sınıflar için normalize dağılım
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
        "classes": labels,
        "docs": "/docs"
    }


@app.get("/health", tags=["Genel"])
def health():
    return {"status": "ok", "model": MODEL_LOADED, "time": datetime.utcnow().isoformat()}


@app.post("/predict/audio", tags=["Tahmin"])
async def predict_audio(file: UploadFile = File(...)):
    """
    Ses dosyası yükle → sınıf tahmini al.
    Kabul edilen formatlar: .wav, .mp3, .ogg, .webm
    """
    contents = await file.read()
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="Ses dosyası çok küçük veya boş.")

    if not MODEL_LOADED:
        return mock_predict()

    try:
        features = extract_features(contents)
        preds    = model.predict(features, verbose=0)[0]
        idx      = int(np.argmax(preds))
        return {
            "label":         labels[idx],
            "confidence":    round(float(preds[idx]), 4),
            "probabilities": {l: round(float(p), 4) for l, p in zip(labels, preds)},
            "model":         "bebek_uyku_modeli.h5",
            "timestamp":     datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")


class LiveRequest(BaseModel):
    source: str = "live_mic"
    timestamp: float = 0.0


@app.post("/predict", tags=["Tahmin"])
def predict_live(req: LiveRequest):
    """
    Frontend'den canlı analiz isteği.
    Gerçek uygulamada ses stream'i işler; şimdi mock/model döner.
    """
    if not MODEL_LOADED:
        return mock_predict()
    # Gerçek ses stream entegrasyonu buraya eklenebilir
    return mock_predict()


@app.get("/classes", tags=["Model"])
def get_classes():
    return {"classes": labels, "count": len(labels)}


# ─── ÇALIŞTIRMA ────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
