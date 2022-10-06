import tensorflow as tf
from tensorflow.python.keras.models import clone_model
from repnet.RepNet import RepNet
from repnet.RepTree import repnet, MinBranch

# Test REPNET on MNIST

MIN_SUITABLE_ACC = 0.95

mnist = tf.keras.datasets.mnist

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Base model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='selu'),
  tf.keras.layers.Dense(64, activation='selu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


main = MinBranch(5, lambda x: 0.05, weights=model.get_weights())

accuracy = 0
past_acc = 0

while accuracy < MIN_SUITABLE_ACC:
    past_accuracy = model.evaluate(test_images, test_labels, verbose=2)[1]
    past_weights = model.get_weights()

    model.fit(train_images, train_labels, epochs=1)
    accuracy = model.evaluate(test_images, test_labels, verbose=2)[1]





