import streamlit as st
import base64
from PIL import Image

# Set page config
st.set_page_config(page_title="EDUMATH", layout="wide", initial_sidebar_state="collapsed")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# Load assets
ic_tittle_base64 = image_to_base64("assets/tittle.png")
ic_elipse_base64 = image_to_base64("assets/elipse.png")
ic_camera_base64 = image_to_base64("assets/ic_camera.png")
ic_medali_base64 = image_to_base64("assets/ic_medali.png")
ic_calculator_base64 = image_to_base64("assets/ic_calculator.png")
ic_minus_base64 = image_to_base64("assets/ic_minus.png")
ic_mult_base64 = image_to_base64("assets/ic_mult.png")
ic_plus_base64 = image_to_base64("assets/ic_plus.png")
ic_div_base64 = image_to_base64("assets/ic_div.png")
ic_bubble_base64 = image_to_base64("assets/bubble.png")
ic_bubble2_base64 = image_to_base64("assets/bubble2.png")
ic_bubble3_base64 = image_to_base64("assets/bubble3.png")

st.markdown("""
    <style>
        /* Hide the top navbar */
        header {visibility: hidden;}
        
        /* Optional: Hide the footer */
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="image-container">
        <img src="data:image/png;base64,{ic_tittle_base64}" width="800">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="p-1">
        <h1 class="subtitle ">Hitung dengan Cepat,</h1>
        <h1 class="subtitle m-0">Belajar dengan Mudah!</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="center desc" style="margin-bottom: 30px;">Ambil langkah untuk menghitung operasi dasar matematika dengan cara yang inovatif.<br>Yuk, coba sekarang!</p>', unsafe_allow_html=True)

if st.button("Mulai Menghitung", key="start_button", type="primary"):
    st.switch_page("pages/predict4.py")

st.markdown(
    f"""
    <div class="frame-container">
        <div class="frame-feature pink">
            <img style="margin-right: 13px;"src="data:image/png;base64,{ic_camera_base64}" width="50">
            <p>Gunakan kamera untuk memindai soal.</p>
        </div>
        <div class="frame-feature kuning">
            <img style="margin-right: 10px;" src="data:image/png;base64,{ic_medali_base64}" width="50">
            <p>Cepat, mudah, dan praktis.</p>
        </div>
        <div class="frame-feature teal">
            <img style="margin-right: 10px;" src="data:image/png;base64,{ic_calculator_base64}" width="50">
            <p>Hitung matematika dasar secara otomatis.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="elipse-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_elipse_base64}" width="800">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="mult-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_mult_base64}" width="130">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="plus-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_plus_base64}" width="115">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="div-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_div_base64}" width="115">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="minus-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_minus_base64}" width="120">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="bubble-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_bubble_base64}" width="48">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="bubble2-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_bubble_base64}" width="30">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="bubble3-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_bubble2_base64}" width="15">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="bubble4-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_bubble2_base64}" width="15">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="bubble5-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_bubble3_base64}" width="15">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="bubble6-background" style="gap: 0rem !important;">
        <img src="data:image/png;base64,{ic_bubble3_base64}" width="9">
    </div>
    """,
    unsafe_allow_html=True
)