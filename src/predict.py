import numpy as np
from tensorflow.keras.models import load_model
from src.extract_features import extract_mfcc
from src.utils import decode_label

model = load_model("models/emotion_model.h5")

def predict_emotion(audio_path):
    features = extract_mfcc(audio_path)
    pred = model.predict(np.expand_dims(features, axis=0))
    emotion_index = np.argmax(pred)
    return decode_label(emotion_index)
