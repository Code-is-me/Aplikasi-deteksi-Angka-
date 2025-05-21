import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# ====== Batesin Pengambilan dataset yang digunakan ======

def limit_images_per_class(dataset_path, max_per_class=420):
    for label_dir in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, label_dir)
        if not os.path.isdir(full_path):
            continue
        image_files = [f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > max_per_class:
            to_keep = set(random.sample(image_files, max_per_class))
            for img in image_files:
                if img not in to_keep:
                    os.remove(os.path.join(full_path, img))

dataset_path = "../data/data-gabungan/"
limit_images_per_class(dataset_path, max_per_class=420)

# ====== Mulai ngeload datasetnya ======

img_size = (28, 28)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode="int"
)

# Normalisasi dan prefetch
AUTOTUNE = tf.data.AUTOTUNE

def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

class_names = train_ds.class_names 
label_to_index = {name: i for i, name in enumerate(class_names)}
print("Label ke index:", label_to_index)
num_classes = len(class_names)

train_ds = train_ds.map(normalize_img).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(normalize_img).cache().prefetch(buffer_size=AUTOTUNE)

# ====== Bangun model CNN klasifikasi ======

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ====== Train dan simpan model ======

model.fit(train_ds, validation_data=val_ds, epochs=10)

# Simpan model
model.save("../models/combined_model3.h5")
print("Model berhasil disimpan sebagai combined_model3.h5")
