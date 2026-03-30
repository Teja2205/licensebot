import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")

def get_embedding(text):
    from huggingface_hub import InferenceClient
    client = InferenceClient(
        token=HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    result = client.feature_extraction(text)
    arr = np.array(result, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    elif arr.ndim == 3:
        arr = arr.mean(axis=1).mean(axis=0)
    return [float(x) for x in arr.flatten()]

def get_embeddings(texts):
    return [get_embedding(text) for text in texts]
