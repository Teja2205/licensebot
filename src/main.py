import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ollama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone_store import upsert_documents, search_pinecone, get_index_stats
from loader import load_documents, split_documents

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL  = "llama3.2"

# ─────────────────────────────────────────────
# Initialize Model
# ─────────────────────────────────────────────
def get_model():
    return SentenceTransformer(MODEL_NAME)

# ─────────────────────────────────────────────
# Component 5 — Conversation Memory
# ─────────────────────────────────────────────
def build_messages(history, context, query):
    system_prompt = """You are LicenseBot, an AI assistant that answers questions strictly based on provided context documents.

Rules you must follow:
1. ONLY use information from the context provided. Never use outside knowledge.
2. If the answer is not in the context, respond with exactly: "I don't have enough information to answer that based on the available documents."
3. Never speculate, never use training knowledge, never add information not in the context.
4. Always cite which document your answer comes from.
5. Keep answers concise and professional."""

    messages = [{"role": "system", "content": system_prompt}]

    for turn in history:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

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
    seen  = set()
    lines = []
    for i, r in enumerate(results):
        if r["source"] not in seen:
            seen.add(r["source"])
            score = r.get("score", r.get("distance", 0))
            lines.append(f"  [{i+1}] {r['source']} (relevance: {score})")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# Core ask function
# ─────────────────────────────────────────────
def ask(query, model, history):
    # Retrieve from Pinecone
    results = search_pinecone(query, model)

    # Build context
    context = ""
    if results:
        for r in results:
            context += f"\n---\n{r['content']}\n"
    else:
        context = "No relevant context found."

    # Build messages with memory
    messages = build_messages(history, context, query)

    # Stream answer
    print(f"\n🤖 LicenseBot: ", end="", flush=True)
    answer = ""
    for chunk in ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True
    ):
        piece   = chunk["message"]["content"]
        print(piece, end="", flush=True)
        answer += piece

    print(f"\n\n📎 Sources:\n{format_sources(results)}")
    print("\n" + "─"*60)
    return answer

# ─────────────────────────────────────────────
# Component 7 — Interactive Terminal Loop
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print("  LicenseBot — AI Licensing Policy Assistant")
    print("  Powered by Llama 3.2 + Pinecone + Sentence Transformers")
    print("="*60)
    print("\nInitializing...")

    model = get_model()

    # Load and upsert base docs on first run
    print("Syncing knowledge base with Pinecone...")
    docs   = load_documents()
    chunks = split_documents(docs)
    upsert_documents(chunks, model)
    get_index_stats()

    print("\n✅ Ready!")
    print("💬 Ask me anything about licensing policies.")
    print("   Type 'quit' to exit | Type 'history' to see conversation\n")
    print("─"*60)

    history = []

    while True:
        try:
            query = input("\n👤 You: ").strip()

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

            answer = ask(query, model, history)
            history.append({"question": query, "answer": answer})

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

if __name__ == "__main__":
    main()