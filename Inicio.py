import streamlit as st
import requests
import sys
import types
import torch

# Fake _path_ to prevent Streamlit from trying to walk into torch.classes
# Solo aplica el parche si no existe ya
if not hasattr(torch.classes, "_path_"):
    torch.classes._path_ = types.SimpleNamespace(_path=[])

# --- Page Config ---
st.set_page_config(
    page_title="Hackathon FEMSA 2025",
    page_icon="🚨",
    layout="wide"
)
from streamlit_lottie import st_lottie
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space


# --- Load Lottie Animation ---
@st.cache_data
def load_lottie(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Custom CSS ---
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: #E60012 !important;
        }
        [data-testid="stVerticalBlock"] {
            background-color: rgba(255, 255, 255, 0.97);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            margin: 1rem;
        }
        h1, h2, h3 {
            color: #E60012;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
col1, col2 = st.columns([2, 1])
with col1:
    st.title("Hackathon FEMSA 2025 🚀")
    st.markdown("### Soluciones inteligentes para tiendas OXXO")
with col2:
    anim = load_lottie("https://assets1.lottiefiles.com/packages/lf20_x62chJ.json")
    st_lottie(anim, height=180)

st.divider()

# --- Module 1 ---
st.subheader("📸 Detección de anomalías mediante visión computacional")
st.write("Usamos visión por computadora para detectar fallas en la organización de los anaqueles de tiendas OXXO.")

st.markdown("""
**Funcionalidades:**
- 🔍 Filtrado de imágenes con acomodo incorrecto.
- 📉 Detección de productos faltantes.
- 📏 Diferenciación de presentaciones similares.
- 🧠 Clasificación por categorías.
""")

st.divider()
st.markdown("### 🧪 Módulos en desarrollo:")

# --- Cards for Modules ---
cols = st.columns(2)

with cols[0]:
    with st.container(border=True):
        st.markdown("**🔎 Módulo de detección de errores**")
        st.write("Detecta estantes mal organizados.")

with cols[1]:
    with st.container(border=True):
        st.markdown("**📉 Análisis de productos faltantes**")
        st.write("Identifica productos ausentes en anaqueles.")

cols = st.columns(2)

with cols[0]:
    with st.container(border=True):
        st.markdown("**📏 Clasificación de presentaciones**")
        st.write("Distingue entre presentaciones similares.")

with cols[1]:
    with st.container(border=True):
        st.markdown("**🧠 Agrupación de productos**")
        st.write("Agrupa productos automáticamente por categoría.")

st.divider()

# --- Optional Metrics Section ---
st.markdown("### 📊 Métricas clave")
c1, c2, c3 = st.columns(3)
c1.metric("Imágenes analizadas", "1,200", "+12%")
c2.metric("Errores detectados", "415", "+5%")
c3.metric("Productos faltantes", "92", "-3%")
style_metric_cards()

add_vertical_space(2)
st.markdown("---")
st.markdown("<p style='text-align:center;'>🔴 Powered by visión computacional + FEMSA 2025</p>", unsafe_allow_html=True)
