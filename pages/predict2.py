import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Set page config
st.set_page_config(page_title="EDUMATH Real-Time", layout="wide", initial_sidebar_state="collapsed")

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'number_model.h5')
char_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'char_model.h5')

number_model = load_model(model_path, safe_mode=True)
char_model = load_model(char_model_path, safe_mode=True)

# Mapping karakter matematika
label_to_symbol = {
    0: '+',
    1: '.',
    2: '/',
    3: '=',
    4: '*',
    5: '-'
}

# Fungsi prediksi
def predict_image(img, model, is_char=False):
    img = cv2.resize(img, (28, 28))  # Ukuran sesuai model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=[0, -1])  # (1, 28, 28, 1)

    pred = model.predict(img, verbose=0)
    class_id = np.argmax(pred)

    if is_char:
        return label_to_symbol.get(class_id, '?')
    else:
        return str(class_id)

# Inisialisasi ekspresi
expression = ""

# Real-time kamera
st.title("EDUMATH - Real-Time Math Recognition")

run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

if run:
    st.warning("Letakkan satu karakter dalam kotak biru!")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Ambil ROI tengah
        h, w, _ = frame.shape
        size = 200
        x1 = w // 2 - size // 2
        y1 = h // 2 - size // 2
        x2 = x1 + size
        y2 = y1 + size

        # Gambar kotak
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = frame[y1:y2, x1:x2]

        # Prediksi
        number_pred = predict_image(roi, number_model, is_char=False)
        char_pred = predict_image(roi, char_model, is_char=True)

        # Deteksi karakter yang dipilih
        current_item = None
        if char_pred in ['+', '-', '*', '/', '=']:
            current_item = char_pred
        else:
            current_item = number_pred

        # Tambahkan ke ekspresi jika belum karakter sama
        if len(expression) == 0 or expression[-1] != current_item:
            expression += current_item

        # Jika ketemu '=', langsung evaluasi ekspresi
        if '=' in expression:
            try:
                clean_expr = expression.replace('=', '')
                result = eval(clean_expr)
                cv2.putText(frame, f"{clean_expr} = {result}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except Exception as e:
                cv2.putText(frame, "Error menghitung!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Reset ekspresi setelah hitung
            expression = ""

        # Tampilkan ekspresi berjalan di kamera
        cv2.putText(frame, f"Expr: {expression}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Tampilkan prediksi sekarang
        cv2.putText(frame, f"Number: {number_pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Char: {char_pred}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # Update ke streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    st.info("Aktifkan kamera dengan mencentang 'Start Camera'.")
    camera.release()
