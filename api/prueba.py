import requests
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

ACCOUNT_ID =  st.secrets["CLOUDFLARE"]["ACCOUNT_ID"]
API_TOKEN = st.secrets['CLOUDFLARE']['API_KEY']

url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/llava-hf/llava-1.5-7b-hf"
headers = {
    "Authorization": f"Bearer {API_TOKEN}", 
    "Content-Type": "application/json"
}
## Cargar la imagen como bytes
with open("assets\Captura de pantalla 2025-05-17 141352.png", "rb") as f:
    image_bytes = f.read()

# Convertir la imagen a una lista de enteros (valores entre 0-255)
image_array = list(image_bytes)

# Construir el payload completo como JSON
payload = {
    "image": image_array,
    "prompt": "Extract the letters from the product",
    "raw": True,
    "max_tokens": 256
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
