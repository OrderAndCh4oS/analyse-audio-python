import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_decision_forests as tfdf

plt.style.use('dark_background')


def load_data():
    with open('data_set.pkl', 'rb') as f:
        data_set = pickle.load(f)

    return data_set


def build_model():
    """
    https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
    """

    model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)
    model.compile(metrics=["accuracy"])

    return model


def train():
    return model.fit(
        x=data_set[0][1].tolist(),
        y=data_set[0][0].tolist(),
    )


def test_data():
    return model.evaluate(x=data_set[2][1].tolist(), y=data_set[2][0].tolist())


if __name__ == "__main__":
    data_set = load_data()
    model = build_model()
    history = train()
    evaluation = model.evaluate(x=data_set[1][1].tolist(), y=data_set[1][0].tolist(), return_dict=True)

    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

    # print('Predict 1:', model.predict(x=np.array([data_set[1][1][0]])), data_set[1][0][0])

    logs = model.make_inspector().training_logs()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")

    plt.show()
