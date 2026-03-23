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

# ─────────────────────────────────────────────
# Component 1 & 2 — Build Vector Store
# ─────────────────────────────────────────────
def build_vector_store(chunks):
    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    return index, chunks, model

# ─────────────────────────────────────────────
# Component 3 — Retrieval with Distance Threshold
# ─────────────────────────────────────────────
def retrieve(query, index, chunks, model, top_k=2, threshold=1.2):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        distance = distances[0][i]
        # Only include chunks below the distance threshold
        if distance < threshold:
            results.append({
                "content": chunks[idx].page_content,
                "source":  os.path.basename(chunks[idx].metadata["source"]),
                "distance": round(float(distance), 4)
            })
    return results

# ─────────────────────────────────────────────
# Component 5 — Conversation Memory
# ─────────────────────────────────────────────
def build_messages(history, context, query):
    system_prompt = """You are LicenseBot, an AI assistant that answers questions about software licensing policies.
Use ONLY the context provided to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."
Always mention which document your answer comes from.
Keep answers concise and professional."""

    messages = [{"role": "system", "content": system_prompt}]

    # Inject conversation history
    for turn in history:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    # Add current question with context
    current = f"""Context:
{context}

Question: {query}"""

    messages.append({"role": "user", "content": current})
    return messages

# ─────────────────────────────────────────────
# Component 6 — Source Citations
# ─────────────────────────────────────────────
def format_sources(results):
    if not results:
        return "No relevant sources found."
    seen = set()
    lines = []
    for i, r in enumerate(results):
        if r["source"] not in seen:
            seen.add(r["source"])
            lines.append(f"  [{i+1}] {r['source']} (relevance score: {r['distance']})")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# Core ask function
# ─────────────────────────────────────────────
def ask(query, index, chunks, model, history):
    # Retrieve relevant chunks
    results = retrieve(query, index, chunks, model)

    # Build context string
    context = ""
    if results:
        for r in results:
            context += f"\n---\n{r['content']}\n"
    else:
        context = "No relevant context found."

    # Build messages with memory
    messages = build_messages(history, context, query)

    # Stream answer from Llama
    print(f"\n🤖 LicenseBot: ", end="", flush=True)
    answer = ""
    for chunk in ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True
    ):
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        answer += piece

    # Print citations
    print(f"\n\n📎 Sources:\n{format_sources(results)}")
    print("\n" + "─"*60)

    return answer

# ─────────────────────────────────────────────
# Component 7 — Interactive Terminal Loop
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print("  LicenseBot — AI Licensing Policy Assistant")
    print("  Powered by Llama 3.2 + FAISS + Sentence Transformers")
    print("="*60)
    print("\nLoading knowledge base...")

    docs = load_documents()
    chunks = split_documents(docs)
    index, chunks, model = build_vector_store(chunks)

    print("✅ Knowledge base ready!")
    print("💬 Ask me anything about licensing policies.")
    print("   Type 'quit' to exit | Type 'history' to see conversation\n")
    print("─"*60)

    # Conversation memory — stores all previous turns
    history = []

    while True:
        try:
            query = input("\n👤 You: ").strip()

            # Handle special commands
            if not query:
                continue
            if query.lower() == "quit":
                print("\nGoodbye!")
                break
            if query.lower() == "history":
                if not history:
                    print("No conversation history yet.")
                else:
                    print("\n--- Conversation History ---")
                    for i, turn in enumerate(history):
                        print(f"\nQ{i+1}: {turn['question']}")
                        print(f"A{i+1}: {turn['answer'][:100]}...")
                continue

            # Get answer
            answer = ask(query, index, chunks, model, history)

            # Save turn to memory
            history.append({
                "question": query,
                "answer":   answer
            })

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

if __name__ == "__main__":
    main()