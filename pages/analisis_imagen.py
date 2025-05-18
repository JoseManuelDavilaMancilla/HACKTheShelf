import streamlit as st
from models.image_analyzer import analyze_image
from PIL import Image

import os

st.set_page_config(
    page_title="Analisis de Imagen",
    page_icon="📸",
)

st.title('Análisis')

st.write("Sube la imagen del anaquel a analizar")
st.write("El modelo analizará la imagen y te mostrará los resultados de la detección de anomalías.")
st.write("Recuerda que el modelo puede tardar un poco en procesar la imagen, por favor ten paciencia.")

st.write("A continuación puedes ver ejemplos de imagenes que puedes subir para analizar:\n")
col1, col2 = st.columns(2)
with col1:
    st.image(Image.open("assets\\IMG_2716.jpg").rotate(270, expand=True), use_container_width=True)
with col2:
    st.image("assets\\anaquel.jpg", use_container_width=True)

st.divider()

uploaded_file = st.file_uploader("Subir Imagen", accept_multiple_files=False, type=["jpg"])

if uploaded_file:
    st.success("Foto subida con éxito")

    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
        image_path = f"temp_{uploaded_file.name}"
    image_name = os.path.splitext(image_path)[0]
    st.image(image_path)
    analysis_results = analyze_image(image_path)
    st.write(analysis_results)

    #st.pyplot(fig2)

    # Despliega 2 graficas del resultado del YOLO y un menu de seleccion para escojer los objetos con streamlit

