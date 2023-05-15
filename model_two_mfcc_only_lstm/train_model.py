import pickle

import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, Dropout, LSTM
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from utilities.training_plots import plot_loss, plot_categorical_accuracy


def load_data():
    with open('data_set.pkl', 'rb') as f:
        data_set = pickle.load(f)

    return data_set


"""
Testing out using normalised mfcc with LSTMs. results aren't great, probably worse than flattening as it is

Idea from here:
https://www.servomagazine.com/magazine/article/music-genre-classification-using-lstm

Worth checking what data they're actually extracting and if they manipulate it at all. 
"""
def build_model():
    inputs = Input(shape=(20, 1293), name="feature")
    x = LSTM(1024, return_sequences=True, input_shape=(20, 1293), name="lstm_1")(inputs)
    x = LSTM(1024, name="lstm_2")(x)
    x = Dense(2048, activation="relu", name="dense_1")(x)
    x = Dense(1024, activation="relu", name="dense_2")(x)
    x = Dense(512, activation="relu", name="dense_3")(x)
    x = Dropout(0.1, name="dropout_2")(x)
    x = Dense(64, activation="relu", name="dense_4")(x)
    x = Dropout(0.1, name="dropout_1")(x)
    outputs = Dense(10, activation="softmax", name="predictions")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()],
    )
    model.summary()

    return model


def plot_history(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.show()


def train(model):
    return model.fit(
        x=data_set[0][1].tolist(),
        y=data_set[0][0].tolist(),
        verbose=1,
        validation_data=(data_set[1][1].tolist(), data_set[1][0].tolist()),
        epochs=15
    )


def test_data(model, data_set):
    return model.evaluate(x=data_set[2][1].tolist(), y=data_set[2][0].tolist(), verbose=0)


def get_max_index(arr):
    return max(range(len(arr)), key=arr.__getitem__)


if __name__ == "__main__":
    data_set = load_data()
    model = build_model()

    # plot_model(model, show_shapes=True)
    history = train(model)

    plot_categorical_accuracy(history)
    plot_loss(history)
    # plot_history(history)

    score = test_data(model, data_set)
    print('Accuracy : ' + str(score[1] * 100) + '%')
    print(data_set[2][1][0].shape)
    prediction = model.predict(tf.constant([data_set[2][1][3]]))

    print(f'Label: {data_set[2][0][3]}')
    print(f'Label Index: {get_max_index(data_set[2][0][3])}')

    print(f'Prediction {prediction[0]}')
    print(f'Prediction Index {get_max_index(prediction[0])}')
