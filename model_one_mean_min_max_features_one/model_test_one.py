import pickle

from keras import Input, Model
from keras.layers import Dense, Dropout
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Adam

from utilities.training_plots import plot_sparse_categorical_accuracy, plot_loss


def load_data():
    with open('data_set.pkl', 'rb') as f:
        data_set = pickle.load(f)

    return data_set


def build_model():
    inputs = Input(shape=498, name="feature")
    x = Dense(512, activation="relu", name="dense_1")(inputs)
    x = Dropout(0.1, name="dropout_1")(x)
    x = Dense(256, activation="relu", name="dense_2")(x)
    x = Dropout(0.1, name="dropout_2")(x)
    x = Dense(128, activation="relu", name="dense_3")(x)
    outputs = Dense(10, activation="softmax", name="predictions")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()],
    )
    model.summary()

    return model


def train():
    return model.fit(
        x=data_set[0][1].tolist(),
        y=data_set[0][0].tolist(),
        verbose=1,
        validation_data=(data_set[1][1].tolist(), data_set[1][0].tolist()),
        epochs=30
    )


def test_data():
    return model.evaluate(x=data_set[2][1].tolist(), y=data_set[2][0].tolist(), verbose=0)


if __name__ == "__main__":
    data_set = load_data()
    model = build_model()

    # plot_model(model, show_shapes=True)
    history = train()

    plot_sparse_categorical_accuracy(history)
    plot_loss(history)
    # plot_history(history)

    score = test_data()
    print('Accuracy : ' + str(score[1] * 100) + '%')
