from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from src.extract_features import extract_mfcc

# Supported CREMA-D emotions
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']  



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
    if not X:
        print(f"[ERROR] No valid .wav files found in {data_dir}.")
    else:
        print(f"[INFO] Loaded {len(X)} samples.")
    return np.array(X), le.transform(y)


def get_label_from_filename(filename):
    # TESS filename format: OAF_back_angry.wav
    emotion = filename.split('_')[-1].split('.')[0].lower()
    return emotion if emotion in EMOTIONS else 'unknown'



def decode_label(label_index):
    return le.inverse_transform([label_index])[0]