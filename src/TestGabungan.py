import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==== Load Model ====
model_path = '../models/combined_model3.h5'
model = load_model(model_path)

# ==== Daftar Label Sesuai Dataset Gabungan ====
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/', '*', '-']

# ==== Fungsi Preprocessing ====
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Gagal membaca gambar: {image_path}")
        return None
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    image = image.astype('float32') / 255.0 # 0-1
    image = np.expand_dims(image, axis=-1)  # (28, 28, 1)
    image = np.expand_dims(image, axis=0)   # (1, 28, 28, 1)
    return image

# ==== Uji Gambar Tunggal ====
def predict_single(image_path):
    image = preprocess_image(image_path)
    if image is None:
        return
    pred = model.predict(image)
    label_index = np.argmax(pred)
    predicted_label = class_names[label_index]
    confidence = np.max(pred)
    print(f"🖼️ {os.path.basename(image_path)} → Prediksi: {predicted_label} (Confidence: {confidence:.2f})")

# ================================
# ==== Ubah Path di Bawah Ini ====
# ================================

# Untuk satu gambar
predict_single('../data/data-gabungan/add/34.jpg')

# Untuk satu folder
# predict_from_folder('../data/character')
