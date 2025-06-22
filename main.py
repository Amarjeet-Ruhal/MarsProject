import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

# Load model and label encoder
def get_type_from_filename(filename):
    if len(filename.split("-")) < 2:
        return "song"
    vocal_channel = filename.split("-")[1]
    return "speech" if vocal_channel == "01" else "song"


# Extract MFCC features
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)  # Resample
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Padding or truncating
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        return mfcc
    except Exception as e:
        print(f"[ERROR] {file_path} => {e}")
        return None

# UI Layout
st.set_page_config(page_title="Speech Emotion Classifier", page_icon="🎙️")

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a WAV audio file to predict the emotion.")

# File upload
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
if uploaded_file is not None:
    file_type = get_type_from_filename(uploaded_file.name)
    if file_type == "speech":
        model = load_model("model/speech_model/emotion_model_speech.keras")
        le = joblib.load("model/speech_model/label_encoder_speech.pkl")
    else:
        model = load_model("model/song_model/emotion_model_song.keras")
        le = joblib.load("model/song_model/label_encoder_song.pkl")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Feature extraction
    with st.spinner("Extracting features and predicting..."):
        mfcc = extract_features(uploaded_file)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # shape = (1, 40, 174, 1)
        prediction = model.predict(mfcc)
        predicted_label = le.inverse_transform([np.argmax(prediction)])

    # Display result
    st.success(f"🎯 Predicted Emotion: **{predicted_label[0].capitalize()}**")
    