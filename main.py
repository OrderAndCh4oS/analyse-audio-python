import matplotlib.pyplot as plt
import librosa
import librosa.display


def load_audio(base_path, file):
    audio_path = f'{base_path}/{file}'
    x, sr = librosa.load(audio_path)
    print(type(x), type(sr))
    print(x.shape, sr)

    return x, sr


def visualise_wave(x, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)


if __name__ == '__main__':
    base_path = 'data/genres_original'
    (x, sr) = load_audio(base_path, 'blues/blues.00000.wav')
    plt.figure(figsize=(14, 5))
    visualise_wave(x, sr)
    plt.show()

