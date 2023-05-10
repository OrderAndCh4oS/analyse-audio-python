import os
import pickle

import librosa

labels = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

base_path = './data/genres_original'

data = []

for i, label in enumerate(labels):
    for filename in os.scandir(f'{base_path}/{label}'):
        if filename.is_file():
            if filename.name == '.DS_Store':
                continue
            x, sr = librosa.load(f'{base_path}/{label}/{filename.name}')
            data.append((i, x, sr))

with open('pickles/processed_data.pkl', 'wb') as f:
    pickle.dump(data, f)
