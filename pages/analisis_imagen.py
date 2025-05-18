import streamlit as st
import models.pipeline_crop as pcrop
import api.mongo_connection as mc
import models.image_analyzer as ima

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
    DATABASE = "files_hackathon"
    COLLECTION = "anaquel_estante"
    st.success("Foto subida con éxito")

    # Save image to a temporary file
    image_path = f"temp_{uploaded_file.name}"
    with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Option 1: Insert raw image bytes (if MongoDB stores binary)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Insert into MongoDB
    document_id = mc.insert_image_data(image_path, DATABASE, COLLECTION)
    analisis_imagen = ima.analyze_image(document_id, DATABASE, COLLECTION)
    # Show image and results
    st.image(image_path)
    #boxes, rectangulo_grande, posicion_por_rect, espacios_vacios = pcrop.toda_la_info(image_bytes)   
    #st.divider()
    #st.write(posicion_por_rect)
    #st.write(espacios_vacios)
    #pcrop.visualizar(image_bytes)

    #
    #st.write(analysis_results)


