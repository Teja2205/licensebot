import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from loader import load_documents, split_documents

MODEL_NAME = "all-MiniLM-L6-v2"

def get_embeddings(chunks):
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded: {MODEL_NAME}")

    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
    return embeddings

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = get_embeddings(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Text preview: {chunk.page_content[:60]}...")
        print(f"Embedding preview: {embedding[:5]}")