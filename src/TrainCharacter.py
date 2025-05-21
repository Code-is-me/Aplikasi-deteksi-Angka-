import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Direktori dataset
data_dir = '../data/character/'
classes = ['add', 'dec', 'div', 'eq', 'mul', 'sub']

# Fungsi untuk memuat dan memproses gambar
def load_images(data_dir, classes):
    images, labels = [], []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f'Warning: Direktori {class_dir} tidak ditemukan.')
            continue
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f'Warning: Gambar {file_path} tidak dapat dimuat.')
                continue
            image = cv2.resize(image, (28, 28))
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=-1)
            images.append(image)
            labels.append(label)
    return np.array(images), to_categorical(labels, num_classes=len(classes))

# Muat data gambar
X, y = load_images(data_dir, classes)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

np.save('../data/character/X_train.npy', X_train)
np.save('../data/character/y_train.npy', y_train)
np.save('../data/character/X_val.npy', X_val)
np.save('../data/character/y_val.npy', y_val)

# Membuat model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Menyimpan model
model.save('../models/char_model.h5')
print("Model disimpan di 'models/char_model.h5'")

# Membuat grafik akurasi model
plt.plot(history.history['accuracy'], 'b--', label='accuracy')
plt.plot(history.history['val_accuracy'], 'y-.', label='(val_accuracy')
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy character')
plt.legend()
plt.show()