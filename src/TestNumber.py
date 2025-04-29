from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import streamlit as st
from PIL import Image

# Memuat model yang telah disimpan
model = load_model('models/number_model.h5')
print("Model berhasil dimuat!")

# Fungsi untuk memuat data uji
def load_mnist_data(image_path, label_path):
    with open(image_path, 'rb') as img_file, open(label_path, 'rb') as lbl_file:
        img_file.read(16)  
        lbl_file.read(8)    
        
        images = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(-1, 28, 28, 1).astype('float32') / 255.0
        labels = np.frombuffer(lbl_file.read(), dtype=np.uint8)
        
        return images, labels

# Memuat data uji
X_test, y_test = load_mnist_data(
    'data/number/t10k-images.idx3-ubyte',
    'data/number/t10k-labels.idx1-ubyte'
)

print(f"Data uji dimuat: {X_test.shape}, Labels: {y_test.shape}")

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

accuracy = np.sum(predicted_labels == y_test) / len(y_test)
print(f"Akurasi pada data uji: {accuracy * 100:.2f}%")

# Menampilkan beberapa hasil prediksi secara visual
# def show_sample_predictions(images, labels, predictions, sample_size=10):
#     plt.figure(figsize=(10, 4))
#     for i in range(sample_size):
#         plt.subplot(2, 5, i + 1)
#         plt.imshow(images[i].reshape(28, 28), cmap='gray')
#         plt.title(f"Label: {labels[i]}\nPred: {predictions[i]}")
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# show_sample_predictions(X_test, y_test, predicted_labels)


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


# Fungsi Preprocessing Image dengan Canny Edge dan Cropping
def preprocess_image(image: np.ndarray) -> tuple:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter kontur kecil untuk menghindari noise
        if w >= 5 and h >= 10:
            cropped_image = edges[y:y+h, x:x+w]

            scale = 28 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Menambahkan padding untuk memastikan gambar 28x28 dan tetap di tengah
            padded_image = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

            # Normalisasi dan menambahkan channel dimension
            padded_image = padded_image.astype('float32') / 255.0
            padded_image = np.expand_dims(padded_image, axis=-1) 
            padded_image = np.expand_dims(padded_image, axis=0)   
            
            processed_images.append((padded_image, x))
    
    # Mengurutkan gambar berdasarkan posisi X (kiri ke kanan) untuk pembacaan angka yang benar
    processed_images = [img for img, _ in sorted(processed_images, key=lambda item: item[1])]
    
    return processed_images, edges

def predict_numbers(images: list) -> list:
    predictions = []
    for image in images:
        prediction = model.predict(image)
        predicted_label = np.argmax(prediction)
        predictions.append(predicted_label)
    return predictions

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_images, edges = preprocess_image(gray_frame)
    predictions = predict_numbers(processed_images)
    
    predicted_text = ''.join(map(str, predictions)) if predictions else '?'
    height = edges.shape[0]
    cv2.putText(edges, f'Prediksi: {predicted_text}', (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Hasil Canny Edge + Prediksi", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()