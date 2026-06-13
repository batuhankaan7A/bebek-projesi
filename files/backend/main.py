"""
BebeSes – FastAPI Backend (DÜZELTİLMİŞ ÇİFT ÇIKTILI VERSİYON)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import librosa
import os
import tempfile
from datetime import datetime

# Konsol uyarılarını gizle (test_et.py'deki gibi)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    
    MODEL_PATH  = os.getenv("MODEL_PATH", "bebek_uyku_modeli.h5")
    LABELS_PATH = os.getenv("LABELS_PATH", "classes.npy")
    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = list(np.load(LABELS_PATH, allow_pickle=True))
    
    print("YAMNet Özellik Çıkarıcı Yükleniyor...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    MODEL_LOADED = True
    print("✅ Model ve YAMNet başarıyla yüklendi!")
except Exception as e:
    print(f"⚠️ Model yüklenemedi: {e}")
    MODEL_LOADED = False
    labels = ["calm", "sleep", "hungry", "cry", "pain"]

app = FastAPI(title="BebeSes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_and_resample_audio(file_path):
    """Sesi YAMNet'in istediği 16000 Hz formata getirir (test_et.py'deki fonksiyon)"""
    audio_numpy, sr = librosa.load(file_path, sr=16000, mono=True)
    audio = tf.convert_to_tensor(audio_numpy, dtype=tf.float32)
    return audio

@app.get("/")
def root():
    return {"status": "ok", "message": "BebeSes API Çalışıyor"}

@app.post("/predict/audio")
async def predict_audio(ses_dosyasi: UploadFile = File(...)):
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model yüklü değil.")

    contents = await ses_dosyasi.read()
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="Ses dosyası çok küçük.")

    # Gelen sesi geçici bir dosyaya kaydet (librosa okuyabilsin diye)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(contents)
        tmp_path = temp_audio.name

    try:
        # 1. Özellik Çıkarımı (YAMNet)
        wav_data = load_and_resample_audio(tmp_path)
        scores, embeddings, spectrogram = yamnet_model(wav_data)
        
        if embeddings.shape[0] == 0:
            raise Exception("Sesten özellik çıkarılamadı.")
            
        mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        features = np.expand_dims(mean_embedding, axis=0) # (1, 1024)
        
        # 2. Model Tahmini (ÇİFT ÇIKTI) - İŞTE KRİTİK NOKTA!
        cry_pred, reason_pred = model.predict(features, verbose=0)
        
        # Ağlama durumu
        crying_probability = float(cry_pred[0][0])
        is_crying = bool(crying_probability > 0.5)
        
        # Detaylı İhtimaller (9 sınıf)
        preds = reason_pred[0]
        en_yuksek_idx = int(np.argmax(preds))
        en_olasi_neden = labels[en_yuksek_idx]
        
        # Olasılıkları sözlük yapısına çevir (Ön yüz için)
        probs_dict = {str(label): float(prob) for label, prob in zip(labels, preds)}
        
        os.remove(tmp_path) # Temizlik
        
        return {
            "label": en_olasi_neden, # Frontend bunu "tahmini_neden" olarak bekliyor olabilir
            "tahmini_neden": en_olasi_neden, # Garanti olsun diye iki isimle de gönderiyoruz
            "confidence": float(preds[en_yuksek_idx]),
            "neden_olasiligi": float(preds[en_yuksek_idx]) * 100,
            "is_crying": is_crying,
            "cry_prob": crying_probability,
            "probabilities": probs_dict
        }
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")
