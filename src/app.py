import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import ollama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone_store import upsert_documents, search_pinecone, get_index_stats
from loader import load_documents, split_documents, load_pdf_file
from main import build_messages, format_sources

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(
    page_title="LicenseBot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LicenseBot")
st.caption("AI-powered assistant for software licensing policies")
st.divider()

if "messages"       not in st.session_state:
    st.session_state.messages = []
if "history"        not in st.session_state:
    st.session_state.history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

@st.cache_resource
def load_model_and_sync():
    model  = SentenceTransformer(MODEL_NAME)
    docs   = load_documents()
    chunks = split_documents(docs)
    upsert_documents(chunks, model)
    return model

with st.spinner("Loading knowledge base..."):
    model = load_model_and_sync()

st.success("✅ Knowledge base ready — ask me anything!")

with st.sidebar:
    st.header("📂 Knowledge Base")
    st.caption("Upload PDFs to expand LicenseBot's knowledge")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.uploaded_files
        ]
        if new_files:
            with st.spinner("Processing uploaded PDFs..."):
                for pdf_file in new_files:
                    pdf_docs   = load_pdf_file(pdf_file)
                    pdf_chunks = split_documents(pdf_docs)
                    upsert_documents(pdf_chunks, model)
                    st.session_state.uploaded_files.append(pdf_file.name)
                    st.success(f"✅ Added: {pdf_file.name}")

    st.divider()
    st.subheader("📄 Loaded Documents")
    for doc in ["software_license_policy.txt",
                "compliance_rules.txt",
                "renewal_terms.txt"]:
        st.markdown(f"• {doc}")
    for fname in st.session_state.uploaded_files:
        st.markdown(f"• 📄 {fname}")

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about licensing policies..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = search_pinecone(prompt, model)
            context = ""
            if results:
                for r in results:
                    context += f"\n---\n{r['content']}\n"
            else:
                context = "No relevant context found."

            messages = build_messages(
                st.session_state.history,
                context,
                prompt
            )

            answer      = ""
            placeholder = st.empty()
            for chunk in ollama.chat(
                model="llama3.2",
                messages=messages,
                stream=True
            ):
                piece    = chunk["message"]["content"]
                answer  += piece
                placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)

            with st.expander("📎 Sources"):
                st.text(format_sources(results))

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer
    })
    st.session_state.history.append({
        "question": prompt,
        "answer":   answer
    })