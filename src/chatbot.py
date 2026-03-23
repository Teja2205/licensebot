import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loader import load_documents, split_documents

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
LLM_MODEL = "llama3.2"

def build_vector_store(chunks):
    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    return index, chunks, model

def retrieve(query, index, chunks, model, top_k=2):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "content": chunks[idx].page_content,
            "source": chunks[idx].metadata["source"],
            "distance": distances[0][i]
        })
    return results

def ask(query, index, chunks, model):
    # Step 1 — retrieve relevant chunks
    results = retrieve(query, index, chunks, model)

    # Step 2 — build context from chunks
    context = ""
    sources = []
    for r in results:
        context += f"\n---\n{r['content']}\n"
        sources.append(r["source"])

    # Step 3 — build prompt
    prompt = f"""You are LicenseBot, an AI assistant that answers questions about software licensing policies.
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."
Always mention which document your answer comes from.

Context:
{context}

Question: {query}

Answer:"""

    # Step 4 — send to Ollama
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    # Step 5 — print answer with sources
    print(f"\nQuestion: {query}")
    print(f"\nAnswer: {answer}")
    print(f"\nSources: {list(set(sources))}")
    print("\n" + "="*60)
    return answer

if __name__ == "__main__":
    # Build knowledge base
    print("Loading documents and building vector store...")
    docs = load_documents()
    chunks = split_documents(docs)
    index, chunks, model = build_vector_store(chunks)
    print("Ready!\n" + "="*60)

    # Test questions
    answer=ask("when do licenses need to be renewed?", index, chunks, model)
    print(answer)
    ask("what happens if someone violates the license policy?", index, chunks, model)
    ask("who approves renewals over $10,000?", index, chunks, model)
    ask("what is the weather today?", index, chunks, model)