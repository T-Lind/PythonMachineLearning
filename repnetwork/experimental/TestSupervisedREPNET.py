import tensorflow as tf
from tensorflow.python.keras.models import clone_model
from dev_repnet.RepTree import repnet, MinBranch, SupervisedTree

# Test REPNET on MNIST

MIN_SUITABLE_ACC = 0.95

mnist = tf.keras.datasets.mnist

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Base model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='selu'),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# main = MinBranch(5, lambda x: 0.05, weights=model.get_weights())
main = SupervisedTree(10, lambda x: 0.02, model, train_images, train_labels, test_images, test_labels)

accuracy = 0

while accuracy < MIN_SUITABLE_ACC:
    print(main)
    accuracy = main.update_branch_ends()
    print(accuracy)
