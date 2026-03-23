import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from loader import load_documents, split_documents

load_dotenv()

MODEL_NAME     = "all-MiniLM-L6-v2"
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "licensebot")

def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")
    pc    = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX)
    return index

def upsert_documents(chunks, model):
    """Embed chunks and upsert into Pinecone"""
    index  = get_pinecone_index()
    texts  = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{os.path.basename(chunk.metadata['source'])}_chunk_{i}"
        vectors.append({
            "id":     vector_id,
            "values": embedding.tolist(),
            "metadata": {
                "text":   chunk.page_content,
                "source": os.path.basename(chunk.metadata["source"]),
                "page":   chunk.metadata.get("page", 1)
            }
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // batch_size + 1} — {len(batch)} vectors")

    print(f"\nTotal vectors upserted: {len(vectors)}")
    return len(vectors)

def search_pinecone(query, model, top_k=3, threshold=0.1):
    """Search Pinecone for relevant chunks"""
    index          = get_pinecone_index()
    query_embedding = model.encode([query])[0].tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    filtered = []
    for match in results["matches"]:
        if match["score"] >= threshold:
            filtered.append({
                "content":  match["metadata"]["text"],
                "source":   match["metadata"]["source"],
                "score":    round(match["score"], 4),
                "page":     match["metadata"].get("page", 1)
            })
    return filtered

def get_index_stats():
    """Show how many vectors are stored"""
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats['total_vector_count']}")
    return stats

if __name__ == "__main__":
    print("Loading documents...")
    model  = SentenceTransformer(MODEL_NAME)
    docs   = load_documents()
    chunks = split_documents(docs)

    print("\nUpserting to Pinecone...")
    upsert_documents(chunks, model)

    print("\nIndex stats:")
    get_index_stats()

    print("\nTesting search...")
    results = search_pinecone("when do licenses need to be renewed?", model)
    for r in results:
        print(f"\nSource: {r['source']} (score: {r['score']})")
        print(f"Content: {r['content'][:100]}...")