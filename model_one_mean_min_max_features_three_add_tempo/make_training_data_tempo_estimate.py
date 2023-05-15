import pickle
import numpy as np
import librosa.feature
import tensorflow as tf

def get_mfcc(y, sr):
    return np.array(librosa.feature.mfcc(y=y, sr=sr))


def get_mel_spectrogram(y, sr):
    return np.array(librosa.feature.melspectrogram(y=y, sr=sr))


def get_chroma_vector(y, sr):
    return np.array(librosa.feature.chroma_stft(y=y, sr=sr))


def get_tonnetz(y, sr):
    return np.array(librosa.feature.tonnetz(y=y, sr=sr))


def get_zero_crossings(y):
    return librosa.zero_crossings(y, pad=False)


def get_zero_crossing_rate(y):
    return librosa.feature.zero_crossing_rate(y)


def get_tempo(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return np.array(librosa.feature.tempo(onset_envelope=onset_env, sr=sr))


def get_feature(y, sr):
    mfcc = get_mfcc(y, sr)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    mfcc_feature = np.concatenate((mfcc_mean, mfcc_min, mfcc_max))

    mel_spectrogram = get_mel_spectrogram(y, sr)
    mel_spectrogram_mean = mel_spectrogram.mean(axis=1)
    mel_spectrogram_min = mel_spectrogram.min(axis=1)
    mel_spectrogram_max = mel_spectrogram.max(axis=1)
    mel_spectrogram_feature = np.concatenate((mel_spectrogram_mean, mel_spectrogram_min, mel_spectrogram_max))

    chroma = get_chroma_vector(y, sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_min = chroma.min(axis=1)
    chroma_max = chroma.max(axis=1)
    chroma_feature = np.concatenate((chroma_mean, chroma_min, chroma_max))

    tonnetz = get_tonnetz(y, sr)
    tonnetz_mean = tonnetz.mean(axis=1)
    tonnetz_min = tonnetz.min(axis=1)
    tonnetz_max = tonnetz.max(axis=1)
    tonnetz_feature = np.concatenate((tonnetz_mean, tonnetz_min, tonnetz_max))

    zero_crossing_rate = get_zero_crossing_rate(y)
    zero_crossing_rate_mean = zero_crossing_rate.mean(axis=1)
    zero_crossing_rate_min = zero_crossing_rate.min(axis=1)
    zero_crossing_rate_max = zero_crossing_rate.max(axis=1)
    zero_crossing_feature = np.concatenate((
        zero_crossing_rate_mean,
        zero_crossing_rate_min,
        zero_crossing_rate_max
    ))

    """
    Adding tempo seems to have a negative impact on results dropped from ~60% to low fifties
    It also takes ages to process the tempo
    """
    tempo = min(get_tempo(y, sr) / 1000, 1)  # max tempo 1000 bpm or level out at 1

    return np.concatenate((
        chroma_feature,
        mel_spectrogram_feature,
        mfcc_feature,
        tonnetz_feature,
        zero_crossing_feature,
        tempo
    ))


def make_training_data():
    with open('../pickles/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    xs = []
    ys = []
    for label, y, sr in data:
        features = get_feature(y, sr)
        xs.append(features)
        ys.append(label)
    xs = np.array(xs)
    ys = tf.one_hot(np.array(ys), 10)
    # ys = np.array(ys)
    print(xs.shape)
    print(ys.shape)

    permutations = np.random.permutation(999)
    features = np.array(xs)[permutations]
    labels = np.array(ys)[permutations]
    features_train = features[0:700]
    labels_train = labels[0:700]
    features_validate = features[700:850]
    labels_validate = labels[700:850]
    features_test = features[850:999]
    labels_test = labels[850:999]
    data_set = (
        (labels_train, features_train),
        (labels_validate, features_validate),
        (labels_test, features_test)
    )
    with open('data_set.pkl', 'wb') as f:
        pickle.dump(data_set, f)


if __name__ == "__main__":
    make_training_data()
