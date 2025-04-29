import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import gzip

# Untuk membaca file .idx
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, size = int.from_bytes(f.read(4), 'big'), int.from_bytes(f.read(4), 'big')
        rows, cols = int.from_bytes(f.read(4), 'big'), int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
        return data / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, size = int.from_bytes(f.read(4), 'big'), int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Fungsi untuk menampilkan gambar beserta labelnya
def show_images(images, labels, sample_size=5):
    plt.figure(figsize=(10, 2))
    for i in range(sample_size):
        plt.subplot(1, sample_size, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"label:{labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Path data
data_path = '../data/number/'
train_images = load_mnist_images(os.path.join(data_path, 'train-images.idx3-ubyte'))
train_labels = load_mnist_labels(os.path.join(data_path, 'train-labels.idx1-ubyte'))
test_images = load_mnist_images(os.path.join(data_path, 't10k-images.idx3-ubyte'))
test_labels = load_mnist_labels(os.path.join(data_path, 't10k-labels.idx1-ubyte'))

# Tambahkan dimensi channel (28x28 -> 28x28x1)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# One-hot encoding labels
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), batch_size=64)

model.save('../models/number_model.keras')

print("Model disimpan di 'models/number_model.keras'")

images = load_mnist_images('../data/number/t10k-images.idx3-ubyte')
labels = load_mnist_labels('../data/number/t10k-labels.idx1-ubyte')

show_images(images, labels, sample_size=5)

