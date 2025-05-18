import requests
import streamlit as st

ACCOUNT_ID =  st.secrets["CLOUDFLARE"]["ACCOUNT_ID"]
API_TOKEN = st.secrets['CLOUDFLARE']['API_KEY']


url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/meta/llama-2-7b-chat-int8"
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "messages": [
    
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"}
    ]
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
