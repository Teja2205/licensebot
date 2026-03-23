import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loader import load_documents, split_documents, load_pdf_file
from main import build_vector_store, retrieve, build_messages, format_sources

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LicenseBot",
    page_icon="🤖",
    layout="centered"
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🤖 LicenseBot")
st.caption("AI-powered assistant for software licensing policies")
st.divider()

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "messages"       not in st.session_state:
    st.session_state.messages = []
if "history"        not in st.session_state:
    st.session_state.history = []
if "index"          not in st.session_state:
    st.session_state.index = None
if "chunks"         not in st.session_state:
    st.session_state.chunks = []
if "model"          not in st.session_state:
    st.session_state.model = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ─────────────────────────────────────────────
# Load Base Knowledge Base Once
# ─────────────────────────────────────────────
@st.cache_resource
def load_base_knowledge():
    docs   = load_documents()
    chunks = split_documents(docs)
    index, chunks, model = build_vector_store(chunks)
    return index, chunks, model

# ─────────────────────────────────────────────
# Sidebar — Document Upload
# ─────────────────────────────────────────────
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
                # Start with base knowledge
                base_index, base_chunks, model = load_base_knowledge()

                # Load base docs
                all_chunks = list(base_chunks)

                # Add each new PDF
                for pdf_file in new_files:
                    pdf_docs   = load_pdf_file(pdf_file)
                    pdf_chunks = split_documents(pdf_docs)
                    all_chunks.extend(pdf_chunks)
                    st.session_state.uploaded_files.append(pdf_file.name)
                    st.success(f"✅ Added: {pdf_file.name}")

                # Rebuild vector store with all chunks
                new_index, new_chunks, new_model = build_vector_store(all_chunks)
                st.session_state.index  = new_index
                st.session_state.chunks = new_chunks
                st.session_state.model  = new_model

    # Show loaded documents
    st.divider()
    st.subheader("📄 Loaded Documents")

    # Always show base docs
    base_docs = ["software_license_policy.txt",
                 "compliance_rules.txt",
                 "renewal_terms.txt"]
    for doc in base_docs:
        st.markdown(f"• {doc}")

    # Show uploaded PDFs
    for fname in st.session_state.uploaded_files:
        st.markdown(f"• 📄 {fname}")

    # Clear conversation button
    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

# ─────────────────────────────────────────────
# Load Knowledge Base (base or with PDFs)
# ─────────────────────────────────────────────
if st.session_state.index is None:
    with st.spinner("Loading knowledge base..."):
        index, chunks, model = load_base_knowledge()
        st.session_state.index  = index
        st.session_state.chunks = chunks
        st.session_state.model  = model

st.success("✅ Knowledge base ready — ask me anything!")

# ─────────────────────────────────────────────
# Display Conversation History
# ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask about licensing policies..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role":    "user",
        "content": prompt
    })

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = retrieve(
                prompt,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.model
            )

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

            # Stream response
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

            # Show sources
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