import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loader import load_documents, split_documents

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

def build_vector_store(chunks):
    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Convert to float32 — FAISS requirement
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    print(f"\nVector store built successfully")
    print(f"Total vectors stored: {index.ntotal}")
    return index, chunks, model

def search(query, index, chunks, model, top_k=3):
    # Embed the user's question
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS for closest chunks
    distances, indices = index.search(query_embedding, top_k)

    print(f"\nQuery: {query}")
    print(f"\nTop {top_k} relevant chunks found:\n")

    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        distance = distances[0][i]
        print(f"--- Result {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Distance: {distance:.4f}")
        print(f"Content: {chunk.page_content[:150]}...")
        print()
        results.append(chunk)
    return results

if __name__ == "__main__":
    # Build the store
    docs = load_documents()
    chunks = split_documents(docs)
    index, chunks, model = build_vector_store(chunks)

    # Test with sample queries
    search("when do licenses need to be renewed?", index, chunks, model)
    search("what happens if someone installs unlicensed software?", index, chunks, model)