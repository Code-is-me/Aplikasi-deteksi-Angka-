import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2

model = keras.models.load_model('../models/char_model.h5')

X_val = np.load('../data/character/X_val.npy')
y_val = np.load('../data/character/y_val.npy')

label_to_symbol = {
    0: '+', 
    1: '.',
    2: '/',
    3: '=',
    4: '*',
    5: '-' 
}

# Lakukan prediksi pada data validasi
predictions = model.predict(X_val)

# Tampilkan beberapa prediksi dan label aslinya
# for i in range(10):
#     image = X_val[i].reshape(28, 28)
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.title(f'Predicted: {label_to_symbol[np.argmax(predictions[i])]}, Actual: {label_to_symbol[np.argmax(y_val[i])]}')
#     plt.show()

print("Prediksi selesai.")

# plt.tight_layout()
# plt.show()


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


def preprocess_image(image: np.ndarray) -> tuple:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 5 and h >= 10:
            cropped_image = edges[y:y+h, x:x+w]
            scale = 28 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            padded_image = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

            padded_image = padded_image.astype('float32') / 255.0
            padded_image = np.expand_dims(padded_image, axis=-1)
            padded_image = np.expand_dims(padded_image, axis=0)
            processed_images.append((padded_image, x))
    processed_images = [img for img, _ in sorted(processed_images, key=lambda item: item[1])]
    return processed_images, edges

# Fungsi untuk memprediksi simbol menggunakan model
def predict_symbols(images: list) -> list:
    predictions = []
    for image in images:
        prediction = model.predict(image)
        predicted_symbol = label_to_symbol[np.argmax(prediction)]
        predictions.append(predicted_symbol)
    return predictions

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_images, edges = preprocess_image(gray_frame)
    predictions = predict_symbols(processed_images)

    predicted_text = ''.join(predictions) if predictions else '?'
    height = edges.shape[0]
    cv2.putText(edges, f'Prediksi: {predicted_text}', (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hasil Canny Edge + Prediksi', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()