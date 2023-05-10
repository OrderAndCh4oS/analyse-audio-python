import pickle

import librosa.feature
import numpy as np
import tensorflow as tf


def get_feature(y, sr):
    mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr))
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    return np.concatenate((mfcc_mean, mfcc_min, mfcc_max))


def make_features():
    with open('../pickles/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    xs = []
    ys = []
    for label, y, sr in data:
        feature = get_feature(y, sr)
        print(feature.shape)
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
