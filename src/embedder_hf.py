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
    embedding = response.json()
    if isinstance(embedding[0], list):
        embedding = np.mean(embedding, axis=0)
    return embedding

def get_embeddings(texts):
    return [get_embedding(text) for text in texts]
