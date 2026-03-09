import librosa
import numpy as np

def extract_audio_features(file):

    y, sr = librosa.load(file)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    pitch = librosa.yin(y, fmin=50, fmax=300)

    energy = np.mean(librosa.feature.rms(y=y))

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        [np.mean(pitch)],
        [energy]
    ])

    return features