import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tavsiyeler import oneriler

st.set_page_config(
    page_title="Bitki Sagligi Tespit Platformu",
    layout="wide"
)

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Model yuklenemedi: {e}")

try:
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
except Exception as e:
    st.error(f"labels.txt bulunamadi: {e}")

st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
        color: #1B5E20 !important;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    .main-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #1B5E20;
    }
    
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #4CAF50 !important;
        margin-bottom: 50px;
        font-weight: 500;
    }

    div[data-testid="stRadio"] label {
        font-size: 20px !important;
        font-weight: bold !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%) !important;
        color: #FFFFFF !important;
        border-radius: 30px !important;
        border: none !important;
        height: 65px;
        width: 100%;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 6px 20px rgba(27, 94, 32, 0.25);
        transition: all 0.3s ease-in-out;
        margin-top: 20px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(27, 94, 32, 0.45);
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%) !important;
    }
    
    .panel-container {
        border: 2px solid #1B5E20;
        padding: 35px;
        border-radius: 20px;
        background-color: #F1F8E9;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.04);
        margin-top: 30px;
    }
    
    .section-title {
        font-size: 26px;
        font-weight: 800;
        border-bottom: 3px solid #1B5E20;
        padding-bottom: 12px;
        margin-bottom: 25px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-text {
        font-size: 22px;
        line-height: 1.8;
        color: #1B5E20;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">BITKI HASTALIGI TESPITI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"></div>', unsafe_allow_html=True)

col_center = st.columns([1, 4, 1])[1]

with col_center:
    girdi_turu = st.radio(
        "Görsel Veri Giriş Yöntemi:", 
        ["Dosya Yukle", "Kamera Entegrasyonu"], 
        horizontal=True
    )

    image = None

    if girdi_turu == "Dosya Yukle":
        uploaded_file = st.file_uploader("Analiz edilecek yaprak fotografini secin", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_file = st.camera_input("Sistem kamerasini kullanarak fotograf cekin")
        if camera_file:
            image = Image.open(camera_file).convert("RGB")

    if image:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(image, caption="Giris Yapilan Yapi Analiz Görseli", use_container_width=True)
        
        if st.button("ANALIZI BASLAT"):
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array.astype(np.float32), axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            prediction = np.argmax(output)
            confidence = np.max(output) * 100
            disease_class = labels[prediction] if len(labels) > 0 else "Bilinmeyen Sinif"
            
            temiz_isim = disease_class.replace("___", " - ").replace("_", " ")
            
            st.markdown(f"""
            <div class="panel-container">
                <div class="section-title">Analiz Bulgulari</div>
                <div class="result-text">
                    <p><b>Saptanan Patoloji / Durum:</b> {temiz_isim}</p>
                    <p><b>Tehsis Dogruluk Orani:</b> %{confidence:.2f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            tavsiye_metni = oneriler.get(disease_class, "Ilgili sinif icin sistemde spesifik bir uzman tavsiyesi tanimlanmamistir.")
            
            st.markdown(f"""
            <div class="panel-container">
                <div class="section-title">Uzman Müdahale Stratejisi</div>
                <div class="result-text" style='font-style: italic;'>
                    <p>{tavsiye_metni}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)