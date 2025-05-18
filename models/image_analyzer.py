import json
import requests
import base64
from PIL import Image
from io import BytesIO
import streamlit as st
import models.pipeline_crop as pcrop
import api.mongo_connection as mc
import logging
import os
    
def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, Image.Image): 
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise TypeError("Expected str or PIL.Image.Image")
    
def cloudflare_llavahf(image_list):
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
  
    for img in image_list:
        with open(img, "rb") as f:
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

#_______________________________________

def cloudflare_llama(description : str) -> str:

    inputs = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"}
    ]
    }

    account_id = st.secrets["CLOUDFLARE"]["ACCOUNT_ID"]
    API_BASE_URL = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/'
    headers = {
        "Authorization": f"Bearer {st.secrets['CLOUDFLARE']['API_KEY']}"
    }

    def run(model, inputs):
        response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=inputs)
        return response.json()
    
    output = run("@cf/meta/llama-2-7b-chat-int8", inputs)

    logging.info(f"Output: {output}")
    return output

#--------------------------------------------------------------------------------
def analyze_image(document_id, DATABASE, COLLECTION):
    image_bytes = mc.get_image_data(document_id, DATABASE, COLLECTION)
    yolo_data = pcrop.toda_la_info(image_bytes)
    yolo_mongo = mc.insert_yolo_data(document_id, yolo_data, DATABASE, COLLECTION)
    return yolo_mongo
    