import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
import re

# Set page config
st.set_page_config(page_title="EDUMATH", layout="wide", initial_sidebar_state="collapsed") 

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Hide the top navbar */
        header {visibility: hidden;}
        
        /* Optional: Hide the footer */
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Load models
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'number_model.h5')
number_model = load_model(model_path)

char_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'char_model.h5')
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
    # 1. Tingkatkan kontras
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    # 2. Blur ringan buat kurangi noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))

    # 3. Thresholding biner (langsung hitam-putih)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Canny Edge untuk deteksi tepi
    edges = cv2.Canny(thresh, 50, 150)

    # 5. Sedikit perbesar tepi biar lebih nyambung
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 6. Cari kontur eksternal saja
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 7. Filter kontur kecil (skip noise kecil)
        if w >= 4 and h >= 8:
            cropped_image = edges[y:y+h, x:x+w]

            # 8. Resize proporsional ke 28x28
            scale = 28 / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 9. Padding supaya pas 28x28
            padded_image = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

            # 10. Normalisasi ke [0,1] dan expand dims
            padded_image = padded_image.astype('float32') / 255.0
            padded_image = np.expand_dims(padded_image, axis=-1)  # (28,28,1)
            padded_image = np.expand_dims(padded_image, axis=0)   # (1,28,28,1)

            processed_images.append((padded_image, x))

    # 11. Urutkan kontur dari kiri ke kanan
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

def clean_prediction(predicted_text):
    # Hanya izinkan angka 0-9, + - * / . (titik untuk desimal)
    allowed_chars = '0123456789+-*/.'
    cleaned = ''.join(c for c in predicted_text if c in allowed_chars)

    # Tambahan: hilangkan jika awalan/operator tidak valid
    cleaned = re.sub(r'^[*/+.-]+', '', cleaned)  # hilangkan operator aneh di awal
    cleaned = re.sub(r'[*/+.-]+$', '', cleaned)  # hilangkan operator aneh di akhir
    cleaned = re.sub(r'[^0-9][*/+.-]{2,}', '', cleaned)  # hilangkan operator berurutan aneh

    return cleaned

# ========================== Streamlit UI ==========================
col1, col2, col3 = st.columns([1, 8, 1])
with col1:
    back_button = st.button("Kembali", type="secondary")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_skip = 10
frame_count = 0

with col2:
    st_frame = st.empty()
    while True:
        if back_button:
            cap.release()
            cv2.destroyAllWindows()
            st.switch_page("home.py")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        h, w, _ = frame.shape
        zoom_factor = 1.5  # Misal mau zoom 1.5x
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        frame = frame[start_y:start_y + new_h, start_x:start_x + new_w]
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_images, edges = preprocess_image(gray_frame)
        predictions = predict_symbols(processed_images)

        print(predictions)

        predicted_text = ''.join(predictions) if predictions else '?'
        cleaned_text = clean_prediction(predicted_text)

        try:
            result = eval(cleaned_text)
        except (SyntaxError, ZeroDivisionError, NameError):
            result = "Tidak Valid"

        height = edges.shape[0]
        cv2.putText(edges, f'Hasil: {result}', (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        output_frame = cv2.resize(frame, (960, 540))
        st_frame.image(edges, channels="GRAY", use_container_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()