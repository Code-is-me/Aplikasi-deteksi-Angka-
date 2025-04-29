import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(page_title="EduMath", layout="wide")

# Load models
model_path = os.path.join(os.path.dirname(__file__), 'models', 'number_model.h5')
number_model = load_model(model_path)

char_model_path = os.path.join(os.path.dirname(__file__), 'models', 'char_model.h5')
char_model = load_model(char_model_path)

# Mapping karakter matematika
label_to_symbol = {
    0: '+',  
    1: '.',  
    2: '/',  
    3: '=',  
    4: '*',  
    5: '-'   
}

def preprocess_image(image: np.ndarray) -> tuple:
    # Tingkatkan kontras
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 3 and h >= 5: 
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

def predict_symbols(images: list) -> list:
    predictions = []
    for image in images:
        number_prediction = number_model.predict(image)
        char_prediction = char_model.predict(image)

        number_confidence = np.max(number_prediction)
        char_confidence = np.max(char_prediction)
        
        if number_confidence > char_confidence and number_confidence > 0.7:
            predictions.append(str(np.argmax(number_prediction)))
        elif char_confidence > 0.7:
            predictions.append(label_to_symbol[np.argmax(char_prediction)])
        else:
            predictions.append('?')
    
    return predictions

st.title("EduMath - Deteksi Angka dan Karakter MTK")
st.write("Gunakan kamera untuk mendeteksi angka dan simbol matematika secara real-time.")

if st.button('Mulai Menghitung'):
    cap = cv2.VideoCapture(0)
    frame_skip = 5  # Prediksi tiap 5 frame
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_images, edges = preprocess_image(gray_frame)
        predictions = predict_symbols(processed_images)

        predicted_text = ''.join(predictions) if predictions else '?'
        cleaned_text = predicted_text.replace('?', '').replace('=', '')

        try:
            result = eval(cleaned_text)
        except (SyntaxError, ZeroDivisionError, NameError):
            result = "Tidak Valid"

        height = edges.shape[0]
        cv2.putText(edges, f'Hasil: {result}', (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Hasil Canny Edge + Prediksi', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

st.write("Klik tombol di atas untuk mulai mendeteksi.")