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
    
    # 2. YAMNet Özellik Çıkarıcıyı Yükle
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
    description="Bebek ses analiz API'si — YAMNet + Çift Çıktılı bebek_uyku_modeli.h5",
    version="1.0.0"
)

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
    """YAMNet ile ses dosyasından özellikleri çıkar"""
    audio_numpy, sr = librosa.load(file_path, sr=16000, mono=True)
    wav_data = tf.convert_to_tensor(audio_numpy, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    
    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    features = np.expand_dims(mean_embedding, axis=0) # (1, 1024)
    return features

# ─── ROUTE'LAR ────────────────────────────────────────
@app.get("/", tags=["Genel"])
def root():
    return {"service": "BebeSes API", "status": "ok", "model_loaded": MODEL_LOADED}

@app.post("/predict/audio", tags=["Tahmin"])
async def predict_audio(ses_dosyasi: UploadFile = File(...)):
    contents = await ses_dosyasi.read()
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="Ses dosyası çok küçük veya boş.")

    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model yüklü değil.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(contents)
        tmp_path = temp_audio.name

    try:
        # 1. YAMNet Özellik Çıkarımı
        features = extract_yamnet_features(tmp_path)
        
        # 2. Model Tahmini - İŞTE DÜZELTİLEN YER BURASI (ÇİFT ÇIKTI)
        cry_pred, reason_pred = model.predict(features, verbose=0)
        
        # Sadece 9'lu sınıfın (nedenlerin) yüzdelerini alıyoruz
        preds = reason_pred[0]
        idx = int(np.argmax(preds))
        
        is_crying = bool(cry_pred[0][0] > 0.5)
        
        os.remove(tmp_path)
        
        return {
            "label":         labels[idx],
            "confidence":    round(float(preds[idx]), 4),
            "probabilities": {l: round(float(p), 4) for l, p in zip(labels, preds)},
            "is_crying":     is_crying,
            "cry_prob":      round(float(cry_pred[0][0]), 4),
            "timestamp":     datetime.utcnow().isoformat()
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
