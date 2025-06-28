A machine learning project to classify human emotions from voice using audio signals.

## 🔍 Supported Emotions
- Angry
- Happy
- Neutral
- Sad
- Disgust

## 🛠️ Tech Stack
- Python, TensorFlow
- `librosa` for audio preprocessing
- MLP model
- `Streamlit` for UI

## 📦 Setup
```bash
git clone <repo-url>
cd voice-emotion-recognition
pip install -r requirements.txt
```

## 📊 Train the Model
```bash
python src/train_model.py
```

## 💻 Run the App
```bash
streamlit run web_app/app.py
```

## 📁 Data
Use the RAVDESS dataset: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

## 🔗 Credits
- RAVDESS dataset
- Cardiff NLP for emotion label mapping

---

This completes a full working project setup for emotion recognition from voice!
