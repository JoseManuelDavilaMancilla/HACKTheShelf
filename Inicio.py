"""
---
title: "Bienvenida"
icon: "🏠"
---
"""
import streamlit as st

st.set_page_config(
        page_title="Hackathon",
        page_icon="🚨",
)

st.title('Hackathon2025 FEMSA')

st.divider()

st.subheader("Detección de anomalías y análisis de visión computacional")
st.write("Utilizamos visión computacional con el objetivo de detectar posibles fallas en los anaqueles.")

st.write("Nuestras funcionalidades:")
st.write("Filtrar las imágenes con acomodaciones incorrectas dentro de los anaqueles .")
st.write("Detectar posibles productos faltantes dentro de los estantes .")
st.write("Diferenciar entre productos de la misma cadena en presentaciones de distintos tamaños.")
st.write("Categorizar de manera exitosa las distintas agrupaciones de productos.")

st.divider()

rows = 2
columns = 2

for row in range(rows):
    cols = st.columns(columns)
    for col_index, col in enumerate(cols):
        with col:
            container = st.container(border=True)
            with container:
                if row == 0 and col_index == 0:
                    st.write("**xd**")
                    st.write("1. s.")
                    
                elif row == 0 and col_index == 1:
                    st.write(" **xdd**")
                    st.write("1.")
                   
                elif row == 1 and col_index == 0:
                    st.write("**xdddd**")
                    st.write("1.")