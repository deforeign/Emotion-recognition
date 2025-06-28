from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from src.extract_features import extract_mfcc

EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

le = LabelEncoder()
le.fit(EMOTIONS)

def load_data(data_dir):
    X, y = [], []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                label = get_label_from_filename(file)
                if label in EMOTIONS:
                    feature = extract_mfcc(os.path.join(root, file))
                    X.append(feature)
                    y.append(label)
    return np.array(X), le.transform(y)

def get_label_from_filename(filename):
    # Example for RAVDESS: filename format -> '03-01-05-01-02-01-12.wav'
    emo_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry',
               '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    emotion_id = filename.split("-")[2]
    return emo_map.get(emotion_id, 'unknown')


def decode_label(label_index):
    return le.inverse_transform([label_index])[0]
