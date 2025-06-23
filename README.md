# 🎧 Mars Project

This project implements an end-to-end pipeline for classifying **emotions** from both **speech** and **song** audio files using deep learning. The models are trained separately on the **RAVDESS** dataset and served via a **Streamlit web app** that accepts `.wav` files and outputs the predicted emotion.

---

## 🔍 Project Overview

The objective is to classify human emotions from raw audio signals using features like MFCCs. The system detects emotions such as:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

Two independent models were trained:
- **Speech Model** (vocal channel = 01)
- **Song Model** (vocal channel = 02)

---

## 📂 Dataset

[RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/records/1188976#.XCx-tc9KhQI)

- **Audio_Speech_Actors_01-24.zip** — 1440 speech files
- **Audio_Song_Actors_01-24.zip** — 1012 song files

Each file name contains metadata in the following format:

**Example**: `02-01-06-01-02-01-12.wav`
- Audio-only (02), Speech (01), Fearful (06), Normal Intensity (01), Statement 2, Repetition 1, Actor 12 (Female)

---

## 🧪 Model Architecture

### ✅ Feature Extraction
- **MFCCs** (Mel Frequency Cepstral Coefficients)
- Zero-padding/truncation to consistent shape
- Normalization

### ✅ Model (CNN)
- Conv2D + MaxPooling + Dropout
- Flatten + Dense + Softmax

### Key Strategy
- I trained two different models one for speech and one for song.
- This was done as one single model was giving very bad results
- Its accuracy was around 70%
- After using this strategy I can give best results
- The uploaded file will automatically can call one of the models

### 🧠 Output Classes:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 📊 Evaluation Metrics

- ✅ **F1 Score** ≥ 80%
- ✅ **Per-class accuracy** ≥ 75%
- ✅ **Overall accuracy** ≥ 80%
- ✅ **Confusion Matrix** reported

Evaluation is performed on:
- Validation set (20%)
- Hidden test set (for final scoring)

---

## 🛠️ Project Structure
| Folder/File        | Description                                            |
| ------------------ | ------------------------------------------------------ |
| `model`    | Training notebook, model, and label encoder |
| `main.py`           | Streamlit app for emotion prediction                   |
| `requirements.txt` | Python dependencies                                    |
| `README.md`        | Project documentation                                  |


---

## 💻 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Trained Model
Trained models are in model file

### 3. Launch Web App
```bash
streamlit run app.py
```

---
### 📹 Demo Video
- 👉 Click here to watch the demo(https://app.screencastify.com/watch/wmun8xi6voyd5niaq80e)
---
🤝 Team & Credits
-Dataset: RAVDESS - Livingstone & Russo (2018)
-Built using Python, TensorFlow, librosa, Streamlit

📜 License
This project is for academic and educational use only.
