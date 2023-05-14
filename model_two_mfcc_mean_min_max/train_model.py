import pickle

import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam

from utilities.training_plots import plot_categorical_accuracy, plot_loss


def load_data():
    with open('data_set.pkl', 'rb') as f:
        data_set = pickle.load(f)

    return data_set


def build_model():
    inputs = Input(shape=60, name="feature")
    x = Dense(40, activation="relu", name="dense_1")(inputs)
    x = Dense(20, activation="relu", name="dense_2")(x)
    outputs = Dense(10, activation="softmax", name="predictions")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()],
    )
    model.summary()

    return model


def train():
    return model.fit(
        x=data_set[0][1].tolist(),
        y=data_set[0][0].tolist(),
        verbose=1,
        validation_data=(data_set[1][1].tolist(), data_set[1][0].tolist()),
        epochs=45
    )


def test_data():
    return model.evaluate(x=data_set[2][1].tolist(), y=data_set[2][0].tolist(), verbose=0)


def get_max_index(arr):
    return max(range(len(arr)), key=arr.__getitem__)


if __name__ == "__main__":
    data_set = load_data()
    model = build_model()

    # plot_model(model, show_shapes=True)
    history = train()

    plot_categorical_accuracy(history)
    plot_loss(history)
    # plot_history(history)

    score = test_data()
    print('Accuracy : ' + str(score[1] * 100) + '%')
    print(data_set[2][1][0].shape)
    prediction = model.predict(tf.constant([data_set[2][1][3]]))

    print(f'Label: {data_set[2][0][3]}')
    print(f'Label Index: {get_max_index(data_set[2][0][3])}')

    print(f'Prediction {prediction[0]}')
    print(f'Prediction Index {get_max_index(prediction[0])}')
