import pickle

import librosa.feature
import numpy as np
import pandas as pd
from sklearn import preprocessing


def get_feature(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    df = pd.DataFrame(mfccs)
    df = df.astype(float)
    scaled_df = preprocessing.scale(df)
    normalised_mfccs = preprocessing.normalize(scaled_df)

    return normalised_mfccs


def make_training_data():
    with open('./features.pkl', 'rb') as f:
        xs, ys = pickle.load(f)
    total = len(xs)
    permutations = np.random.permutation(total)
    features = np.array(xs)[permutations]
    labels = np.array(ys)[permutations]
    features_train = features[0:700]
    labels_train = labels[0:700]
    features_validate = features[700:850]
    labels_validate = labels[700:850]
    features_test = features[850:total]
    labels_test = labels[850:total]
    data_set = (
        (labels_train, features_train),
        (labels_validate, features_validate),
        (labels_test, features_test)
    )
    with open('data_set.pkl', 'wb') as f:
        pickle.dump(data_set, f)


if __name__ == "__main__":
    make_training_data()
