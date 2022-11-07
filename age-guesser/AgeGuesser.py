import tensorflow as tf
import keras
import numpy as np

train_in = np.random.randint(0, 90, (100000, 1))
train_out = train_in.flatten()

test_in = np.random.randint(0, 90, (10000, 1))
test_out = test_in.flatten()

initializer = tf.keras.initializers.Ones()
values = initializer(shape=(1, 1))

# Use ReLU initializer and set all the weights to a value of 1, so it doesn't even need to be trained.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 1)),
    tf.keras.layers.Dense(1, activation="ReLU", kernel_initializer=initializer)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])


# Age guesser UI
exit_condition = False

print("\nAge Guesser:\n----------------------")
while not exit_condition:
    input_age = int(input("\nEnter your integer age: "))
    print(f"Predicted age: {int(model.predict((input_age,)))} [calculated as int(model.predict((input_age,)))]")

    exit_condition = False if input("Continue? (Y/n) ") == "Y" else True
