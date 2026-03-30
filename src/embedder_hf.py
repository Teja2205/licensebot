import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN  = os.getenv("HF_TOKEN", "")
MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def get_embedding(text):
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    response = requests.post(
        MODEL_URL,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    result = response.json()
    arr    = np.array(result, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    elif arr.ndim == 3:
        arr = arr.mean(axis=1).mean(axis=0)
    return [float(x) for x in arr.flatten()]

def get_embeddings(texts):
    return [get_embedding(text) for text in texts]
