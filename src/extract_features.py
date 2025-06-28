import librosa
import numpy as np

def extract_mfcc(file_path, sr=16000, n_mfcc=40):
    # Load original sampling rate
    y, orig_sr = librosa.load(file_path, sr=None, duration=3, offset=0.5)

    # Resample if needed
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)
