# Load the TensorBoard notebook extension
from functools import partial

import tensorflow as tf
import datetime

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(28*28, activation='selu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='selu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

s = 20*len(x_train) // 32
learning_r = tf.keras.optimizers.schedules.ExponentialDecay(0.02, s, 0.1)
optimizer= tf.keras.optimizers.RMSprop(
    learning_rate=learning_r,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
)



model = create_model()
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(x=x_train,
          y=y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

