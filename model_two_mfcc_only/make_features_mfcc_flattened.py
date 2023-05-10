import pickle

import librosa.feature
import numpy as np
import tensorflow as tf


def get_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = mfcc.flatten()

    return mfcc


def make_features():
    with open('../pickles/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    xs = []
    ys = []
    for label, y, sr in data:
        feature = get_feature(y, sr)
        if len(feature) != 16809:
            continue
        xs.append(feature)
        ys.append(label)
    xs = np.array(xs)
    ys = tf.one_hot(np.array(ys), 10)
    # ys = np.array(ys)
    print(xs.shape)
    print(ys.shape)

    with open('features-flattened.pkl', 'wb') as f:
        pickle.dump((xs, ys), f)


if __name__ == "__main__":
    make_features()
