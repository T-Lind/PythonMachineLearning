import keras
import tensorflow as tf
from keras import activations
from tensorflow import Tensor

from Equations import logistic
import matplotlib.pyplot as plt
import random

num_iters = 100
seq_len = 10

def generate_data(rate):
    y_vals = [0.5]
    for _ in range(num_iters - 1):
        y_vals.append(logistic(y_vals[-1], rate))
    return y_vals, rate


def generate_sequence(input):
    return input[-seq_len:]

def modified_relu(input: Tensor) -> Tensor:
    """
    Relu function + 2.0
    :param input:
    :return:
    """
    return activations.leaky_relu(input) + tf.constant(3.0)

def generate_n_sequences(n_sequences):
    data, labels = [], []
    for _ in range(n_sequences):
        seq, rate = generate_data(random.random() + 3)
        data.append(generate_sequence(seq))
        labels.append(rate)
    return data, labels


# Example rate of 3 and 4
plt.plot(generate_data(3.4)[0], label='r=3.4')
plt.plot(generate_data(3.75)[0], label='r=3.75')
plt.legend()

model = keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(seq_len,)),
    tf.keras.layers.Dense(128, activation=modified_relu),
    tf.keras.layers.Dense(256, activation="sigmoid"),
    tf.keras.layers.Dense(512, activation="sigmoid"),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(512, activation="sigmoid"),
    tf.keras.layers.Dense(128, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation=modified_relu)
])

model.compile(optimizer='nadam',
              loss='mse',
              metrics=['accuracy'])

X_train, y_train = generate_n_sequences(100000)
seq, rate = generate_data(random.random() + 3)
pred_tensor = generate_sequence(seq)
print(rate)
print(model.predict((pred_tensor,)))


X_test, y_test = generate_n_sequences(100)

model.fit(X_train, y_train, epochs=10)
print(model.evaluate(X_test, y_test))
print(y_test)
plt.show()
