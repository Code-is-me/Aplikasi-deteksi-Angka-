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
    # image = cv2.imread(image_path)
    chars = segment_characters(frame)
    expression = ""

    # for i, char_img in enumerate(chars):
    #     cv2.imshow(f"Char {i}", char_img)
    for char_img in chars:
        processed = preprocess_char(char_img)
        number_prediction = number_model.predict(processed, verbose=0)
        char_prediction = char_model.predict(processed, verbose=0)

        number_index = np.argmax(number_prediction)
        char_index = np.argmax(char_prediction)

        number_confidence = number_prediction[0][number_index]
        char_confidence = number_prediction[0][char_index]

        if number_confidence > 0.7:
            predicted_number = str(number_index)
            print(f"✔ Predicted: {predicted_number} (Confidence: {number_confidence:.2f})")
            expression += predicted_number
        elif char_confidence > 0.7:
            predicted_char = label_to_symbol[char_index]
            print(f"✔ Predicted: {predicted_char} (Confidence: {char_confidence:.2f})")
            expression += predicted_char
        else:
            print(f"✖ Confidence rendah: {number_confidence:.2f} & {char_confidence:.2f}, karakter dilewati.")

    return expression
# ========================== Streamlit UI ==========================
col1, col2, col3 = st.columns([1, 3, 1])
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

        # Ganti cv2.imshow() dengan ini:
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(display_rgb, channels="RGB", use_container_width=True)

        # Tampilkan frame
        # cv2.imshow("Live Camera - Tekan 's' untuk scan", frame)

        key = cv2.waitKey(1) & 0xFF
        # if key == ord('s'):
        #     print("📸 Scan ekspresi...")
        #     print("🧠 Ekspresi Terdeteksi:", expression)
        #     try:
        #         result = eval(expression)
        #         print("💡 Hasil Evaluasi:", result)
        #     except Exception as e:
        #         print("❌ Gagal evaluasi ekspresi:", e)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     real_time_prediction()
    
    # image_path = 'img/sub.jpg'
    # result = predict_expression(image_path)
    # try:
    #     result = eval(result)
    #     print("💡 Hasil Evaluasi:", result)
    # except Exception as e:
    #     print("❌ Gagal evaluasi ekspresi:", e)
    #     result = None