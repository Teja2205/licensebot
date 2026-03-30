import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def get_embedding(text):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    response = requests.post(
        MODEL_URL,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    result = response.json()
    # HF API returns nested list for sentences — flatten to single vector
    if isinstance(result, list):
        arr = np.array(result)
        if arr.ndim == 2:
            # Multiple token embeddings — mean pool them
            return arr.mean(axis=0).tolist()
        elif arr.ndim == 1:
            return arr.tolist()
    return result

def get_embeddings(texts):
    return [get_embedding(text) for text in texts]
