import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
import re
import base64

# Set page config
st.set_page_config(page_title="EDUMATH", layout="wide", initial_sidebar_state="collapsed") 

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def image_to_base64(image_filename):
    path = os.path.join(os.path.dirname(__file__), "..", "assets", image_filename)
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
# Load assets
ic_bg_base64 = image_to_base64("bg-count.png")
ic_camera_base64 = image_to_base64("ic_camera.png")

st.markdown("""
    <style>
        /* Hide the top navbar */
        header {visibility: hidden;}
        
        /* Optional: Hide the footer */
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{ic_bg_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
""", unsafe_allow_html=True)

# ==== Load Model ==== 
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'combined_model3.h5')
model = load_model(model_path)

# Label output model
index_to_label = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '/', 12: '*', 13: '-'
}

# Preprocessing dan segmentasi seperti sebelumnya
def preprocess_char(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=(0, -1))  # (1, 28, 28, 1)

def segment_characters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])

    for x, y, w, h in bounding_boxes:
        pad = 3
        x, y, w, h = x - pad, y - pad, w + 2*pad, h + 2*pad
        x = max(x, 0)
        y = max(y, 0)
        if w*h < 500:
            continue
        char = image[y:y+h, x:x+w]
        char = cv2.copyMakeBorder(char, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
        char_images.append(char)

    return char_images

def predict_expression(frame):
    chars = segment_characters(frame)
    expression = ""

    for char_img in chars:
        processed = preprocess_char(char_img)
        pred = model.predict(processed, verbose=0)
        predicted_index = np.argmax(pred)
        confidence = pred[0][predicted_index]

        if confidence > 0.7:
            predicted_label = index_to_label[predicted_index]
            print(f"✔ Predicted: {predicted_label} (Confidence: {confidence:.2f})")
            expression += predicted_label
        else:
            print(f"✖ Confidence rendah: {confidence:.2f}, karakter dilewati.")

    return expression

# ========================== Streamlit UI ==========================
col1, col2, col3 = st.columns([1, 2, 1])
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

        display_frame = frame.copy()
        expression = predict_expression(frame)

        try:
            result = eval(expression)
            display_text = f"Ekspresi: {expression}   Hasil: {result}"
        except:
            display_text = f"Ekspresi: {expression}   Hasil: Error"

        cv2.putText(display_frame, display_text, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(display_rgb, channels="RGB", use_container_width=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()