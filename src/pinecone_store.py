import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinecone import Pinecone
from dotenv import load_dotenv
from loader import load_documents, split_documents
from embedder_hf import get_embedding, get_embeddings

load_dotenv()

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "licensebot")

def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")
    pc    = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX)
    return index

def upsert_documents(chunks, model=None):
    index  = get_pinecone_index()
    texts  = [chunk.page_content for chunk in chunks]
    embeddings = get_embeddings(texts)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{os.path.basename(chunk.metadata['source'])}_chunk_{i}"
        vectors.append({
            "id":     vector_id,
            "values": list(embedding),
            "metadata": {
                "text":   chunk.page_content,
                "source": os.path.basename(chunk.metadata["source"]),
                "page":   chunk.metadata.get("page", 1)
            }
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // batch_size + 1} — {len(batch)} vectors")

    print(f"\nTotal vectors upserted: {len(vectors)}")
    return len(vectors)

def search_pinecone(query, model=None, top_k=3, threshold=0.1):
    index           = get_pinecone_index()
    query_embedding = get_embedding(query)

    results = index.query(
        vector=list(query_embedding),
        top_k=top_k,
        include_metadata=True
    )

    filtered = []
    for match in results["matches"]:
        if match["score"] >= threshold:
            filtered.append({
                "content": match["metadata"]["text"],
                "source":  match["metadata"]["source"],
                "score":   round(match["score"], 4),
                "page":    match["metadata"].get("page", 1)
            })
    return filtered

def get_index_stats():
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats['total_vector_count']}")
    return stats
