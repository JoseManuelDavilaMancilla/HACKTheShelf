import requests
import streamlit as st
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os
import json

ACCOUNT_ID =  st.secrets["CLOUDFLARE"]["ACCOUNT_ID"]
API_TOKEN = st.secrets['CLOUDFLARE']['API_KEY']

# Create empty JSON file
with open("descripciones.json", "w") as f:
    json.dump([], f)

url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/llava-hf/llava-1.5-7b-hf"
headers = {
    "Authorization": f"Bearer {API_TOKEN}", 
    "Content-Type": "application/json"
}
## Cargar la imagen como bytes
img_dir = os.listdir("assets")
for img in img_dir:
    image_path = os.path.join("assets", img)
        
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Convertir la imagen a una lista de enteros (valores entre 0-255)
    image_array = list(image_bytes)

    # Construir el payload completo como JSON
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a store shelf product recognition system. You will receive as input a single product image cropped from a shelf. Your response must only contain the name of the product or brand. If you are not sure about the product, please answer unknown.\n"
                )
            },
            {
                "role": "user",
                "content": "Identify the image provided:"
            }
        ],
        "image": image_array,
        "raw": True,
        "max_tokens": 256,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

    result = response.json()
    # Guardar la respuesta como línea separada en un archivo JSON
    with open("descripciones.json", "a") as f:
        json.dump(result, f)
        f.write("\n")