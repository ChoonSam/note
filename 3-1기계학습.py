import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape([-1, 784])
test_images = test_images.reshape([-1, 784])
train_images = train_images / 255.
test_images = test_images / 255.
print(train_images[0])

i = 0
for image in train_images:
    image.shape

    plt.imshow(train_images[i].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

    print(train_labels[i])

    i = i + 1
    if i == 10:
        break

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32, input_shape=(28, 28, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=100, epochs=10, validation_data=(test_images, test_labels))

result = model.evaluate(test_images, test_labels)
print('최적화 완료!')
print("정확도 : ", result[1] * 100, ("%"))