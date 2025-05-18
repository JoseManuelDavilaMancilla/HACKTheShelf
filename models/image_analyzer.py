import json
import requests
import base64
import streamlit as st
from PIL import Image
from io import BytesIO
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
    
def cloudflare_llavahf(image_path : str) -> str:
    imagen = Image.open(image_path)
    inputs = [
        {
            "image": encode_image(image=imagen),
            "max_tokens": 512,
            "temperature": 0.7,
            "prompt": "Analyze the content of the image.", #Prompt feo (optimizar)
            "raw": False,
            "seed": 42 
        }
    ]

    # Getting the base64 string
    account_id = st.secrets["CLOUDFLARE"]["ACCOUNT_ID"]
    API_BASE_URL = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/'
    headers = {
        "Authorization": f"Bearer {st.secrets['CLOUDFLARE']['API_KEY']}"
    }

    def run(model, inputs):
        response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=inputs)
        return response.json()
    
    output = run("@cf/llava-hf/llava-1.5-7b-hf", inputs)

    logging.info(f"Output: {output}")
    return output

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

def analyze_image(image_path : str) -> str:
    DATABASE = "files_hackathon"
    COLLECTION = "anaquel_estante"
    # obtienes la imagen
    #cropped_dict = pcrop.proceso_general(image_path, 0.2, "models\\best.pt")
    #document = mc.insert_image_data(image_path, DATABASE, COLLECTION )
    imagen = mc.get_image_data("6829a0b0526bcfb9cb109397", DATABASE, COLLECTION)
    #cropped_dict = pcrop.proceso_general(imagen, 0.2, "models\\best.pt")
    #boxed = pcrop.get_boxes(imagen, 0.1,"models\\best.pt" )
    data, rectangulo_grande = pcrop.proceso_general(imagen, "models\\best.pt")
    print(data[0])
    
analyze_image("assets\\IMG_2716.jpg")