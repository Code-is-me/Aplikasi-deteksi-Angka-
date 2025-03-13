import streamlit as st

# Page configuration
st.set_page_config(page_title="EduMath", layout="wide")

# Title Section with Custom Font & Size
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

        .title {
            text-align: center;
            color: #FF6F61;
            font-size: 48px;
            font-weight: 700;
            font-family: 'Montserrat', sans-serif;
        }

        .subtitle {
            text-align: center;
            font-size: 60px;
            color: #333;
            font-family: 'Montserrat', sans-serif;
            margin-bottom: 20px;
        }

        .description {
            text-align: center;
            font-size: 20px;
            color: #555;
            font-family: 'Montserrat', sans-serif;
            margin-top: 10px;
            line-height: 1.6;
        }

        .button-container {
            text-align: center;
            margin-top: 30px;
        }

        .start-button {
            display: inline-block;
            padding: 14px 24px;
            font-size: 18px;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
            color: white;
            background-color: #FF6F61;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            text-align: center;
        }

        .start-button:hover {
            background-color: #E65C50;
            transform: scale(1.05);
        }
    </style>

    <h1 class="title">EDUMATH</h1>
    <h2 class="subtitle">Hitung dengan Cepat, Belajar dengan Mudah!</h2>
    <p class="description">
        Ambil langkah untuk menghitung operasi dasar matematika dengan cara yang inovatif. <br>
        Yuk, coba sekarang!
    </p>

    <div class="button-container">
        <button class="start-button">Mulai Menghitung</button>
    </div>
    """,
    unsafe_allow_html=True
)

# Features section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .feature-container {
            display: flex;
            justify-content: center;
            gap: 24px;
            flex-wrap: wrap;
            margin-top: 30px;
        }

        .feature-box {
            width: 350px;
            padding: 24px;
            border-radius: 16px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            font-family: 'Montserrat', sans-serif;
            box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease-in-out;
            cursor: pointer;
        }

        .feature-box:hover {
            transform: scale(1.05);
            box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.2);
        }

        .red { background-color: #FFEBEE; color: #D32F2F; }
        .yellow { background-color: #FFF8E1; color: #F57C00; }
        .blue { background-color: #E3F2FD; color: #1976D2; }
    </style>

    <div class="feature-container">
        <div class="feature-box red">📷 Gunakan kamera untuk memindai soal.</div>
        <div class="feature-box yellow">🏅 Cepat, mudah, dan praktis.</div>
        <div class="feature-box blue">🧮 Hitung matematika dasar secara otomatis.</div>
    </div>
    """,
    unsafe_allow_html=True
)
