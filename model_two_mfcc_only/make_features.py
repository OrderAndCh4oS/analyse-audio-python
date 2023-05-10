import pickle

import librosa.feature
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf


def get_feature(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    df = pd.DataFrame(mfccs)
    df = df.astype(float)
    scaled_df = preprocessing.scale(df)
    normalised_mfccs = preprocessing.normalize(scaled_df)

    return normalised_mfccs


def make_features():
    with open('../pickles/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    xs = []
    ys = []
    for label, y, sr in data:
        feature = get_feature(y, sr)
        if len(feature[0]) != 1293:
            continue
        xs.append(feature)
        ys.append(label)
    xs = np.array(xs)
    ys = tf.one_hot(np.array(ys), 10)
    # ys = np.array(ys)
    print(xs.shape)
    print(ys.shape)

    with open('features.pkl', 'wb') as f:
        pickle.dump((xs, ys), f)


if __name__ == "__main__":
    make_features()
