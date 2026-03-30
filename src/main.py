import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from groq import Groq
from dotenv import load_dotenv
from pinecone_store import upsert_documents, search_pinecone, get_index_stats
from loader import load_documents, split_documents

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    return messages

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
