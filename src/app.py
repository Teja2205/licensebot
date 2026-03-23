import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from main import build_vector_store, ask
from loader import load_documents, split_documents

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
# Load Knowledge Base Once (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_knowledge_base():
    docs = load_documents()
    chunks = split_documents(docs)
    index, chunks, model = build_vector_store(chunks)
    return index, chunks, model

with st.spinner("Loading knowledge base..."):
    index, chunks, model = load_knowledge_base()

st.success("✅ Knowledge base ready — ask me anything about licensing policies!")

# ─────────────────────────────────────────────
# Session State — Conversation Memory
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

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

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Get answer from LicenseBot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Collect streamed answer
            from main import retrieve, build_messages, format_sources
            import ollama

            results = retrieve(prompt, index, chunks, model)

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
            answer = ""
            placeholder = st.empty()
            for chunk in ollama.chat(
                model="llama3.2",
                messages=messages,
                stream=True
            ):
                piece = chunk["message"]["content"]
                answer += piece
                placeholder.markdown(answer + "▌")

            placeholder.markdown(answer)

            # Show sources
            sources_text = format_sources(results)
            with st.expander("📎 Sources"):
                st.text(sources_text)

    # Save assistant message and history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
    st.session_state.history.append({
        "question": prompt,
        "answer": answer
    })