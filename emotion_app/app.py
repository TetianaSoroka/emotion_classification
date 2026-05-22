import streamlit as st
import librosa
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.title("Визначення емоційного забарвлення мовлення")
st.write("Завантажте аудіофайл")

def extract_features_from_buffer(y, sr):
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = np.hstack([
        np.mean(mels_db, axis=1), np.std(mels_db, axis=1),
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(zcr, axis=1), np.std(zcr, axis=1),
        np.mean(rms, axis=1), np.std(rms, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(centroid, axis=1), np.std(centroid, axis=1)
    ])
    return features

@st.cache_resource
def load_assets():
    model = joblib.load('emotion_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

emotions = {
0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry', 4: 'fearful', 5: 'disgust', 6: 'surprised'
}

uploaded_file = st.file_uploader("Виберіть файл (WAV, MP3)", type=['wav', 'mp3'])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Проаналізувати голос"):
        with st.spinner("Обробка..."):
            y, sr = librosa.load(uploaded_file, sr=None)
            
            feat = extract_features_from_buffer(y, sr)
            
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            
            probabilities = model.predict_proba(feat_scaled)[0] 
            
            max_idx = np.argmax(probabilities)
            
            class_id = model.classes_[max_idx]
            
            result_text = emotions.get(class_id, str(class_id))
            
            confidence = probabilities[max_idx] * 100 

            st.success(f"Результат: **{result_text.upper()}**")
            st.metric(label="Впевненість моделі", value=f"{confidence:.2f}%")
            st.progress(int(confidence))

            with st.expander("Подивитися детальний розподіл ймовірностей"):
                labels = [emotions.get(c, c) for c in model.classes_]
    
                fig, ax = plt.subplots()
                x_positions = np.arange(len(labels))

                bars = ax.bar(x_positions,  probabilities, color='#5B7E3C')
                ax.set_xticks(x_positions)
                ax.set_xticklabels(labels, rotation=0, fontsize=10)
                ax.set_ylabel('Ймовірність')
                
                st.pyplot(fig)