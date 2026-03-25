import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone_store import upsert_documents, search_pinecone
from loader import load_documents, split_documents, load_pdf_file
from main import build_messages, format_sources
from database import sign_in, sign_up, create_conversation, save_message, get_conversations, get_messages, save_feedback, get_feedback_stats

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile",

st.set_page_config(
    page_title="LicenseBot",
    page_icon="🤖",
    layout="centered"
)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

if "user"              not in st.session_state:
    st.session_state.user = None
if "jwt"               not in st.session_state:
    st.session_state.jwt = None
if "messages"          not in st.session_state:
    st.session_state.messages = []
if "history"           not in st.session_state:
    st.session_state.history = []
if "conversation_id"   not in st.session_state:
    st.session_state.conversation_id = None
if "uploaded_files"    not in st.session_state:
    st.session_state.uploaded_files = []

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_sync():
    model  = SentenceTransformer(MODEL_NAME)
    docs   = load_documents()
    chunks = split_documents(docs)
    upsert_documents(chunks, model)
    return model

# ─────────────────────────────────────────────
# Login / Signup Page
# ─────────────────────────────────────────────
def show_auth_page():
    st.title("🤖 LicenseBot")
    st.caption("AI-powered assistant for software licensing policies")
    st.divider()

    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        st.subheader("Welcome back")
        email    = st.text_input("Email",    key="signin_email")
        password = st.text_input("Password", type="password", key="signin_password")

        if st.button("Sign In", use_container_width=True):
            if email and password:
                with st.spinner("Signing in..."):
                    result = sign_in(email, password)
                if result["success"]:
                    st.session_state.user = result["user"]
                    st.session_state.jwt  = result["session"].access_token
                    st.success("✅ Signed in successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")
            else:
                st.warning("Please enter email and password")

    with tab2:
        st.subheader("Create account")
        new_email    = st.text_input("Email",           key="signup_email")
        new_password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")

        if st.button("Sign Up", use_container_width=True):
            if new_email and new_password:
                with st.spinner("Creating account..."):
                    result = sign_up(new_email, new_password)
                if result["success"]:
                    st.success("✅ Account created! Please sign in.")
                else:
                    st.error(f"❌ {result['error']}")
            else:
                st.warning("Please enter email and password")

# ─────────────────────────────────────────────
# Main Chat Page
# ─────────────────────────────────────────────
def show_chat_page():
    model = load_model_and_sync()

    # Sidebar
    with st.sidebar:
        st.header("🤖 LicenseBot")
        st.caption(f"Signed in as: {st.session_state.user.email}")
        st.divider()

        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.messages       = []
            st.session_state.history        = []
            st.session_state.conversation_id = None
            st.rerun()

        # Past conversations
        st.subheader("💬 Past Conversations")
        conversations = get_conversations(
            st.session_state.user.id,
            st.session_state.jwt
        )
        for conv in conversations:
            if st.button(
                f"📄 {conv['title'][:30]}...",
                key=conv["id"],
                use_container_width=True
            ):
                st.session_state.conversation_id = conv["id"]
                msgs = get_messages(conv["id"], st.session_state.jwt)
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in msgs
                ]
                st.session_state.history = [
                    {"question": msgs[i]["content"], "answer": msgs[i+1]["content"]}
                    for i in range(0, len(msgs)-1, 2)
                    if msgs[i]["role"] == "user"
                ]
                st.rerun()

        st.divider()

        # PDF Upload
        st.subheader("📂 Upload Documents")
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
                with st.spinner("Processing PDFs..."):
                    for pdf_file in new_files:
                        pdf_docs   = load_pdf_file(pdf_file)
                        pdf_chunks = split_documents(pdf_docs)
                        upsert_documents(pdf_chunks, model)
                        st.session_state.uploaded_files.append(pdf_file.name)
                        st.success(f"✅ {pdf_file.name}")

        st.divider()

        # Sign out
        if st.button("🚪 Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ─────────────────────────────────────────
    # Chat UI
    # ─────────────────────────────────────────
    st.title("🤖 LicenseBot")
    st.caption("AI-powered assistant for software licensing policies")
    st.divider()

    st.success("✅ Knowledge base ready — ask me anything!")

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about licensing policies..."):

        # Create conversation on first message
        if st.session_state.conversation_id is None:
            conv = create_conversation(
                st.session_state.user.id,
                prompt[:50],
                st.session_state.jwt
            )
            if conv:
                st.session_state.conversation_id = conv["id"]

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({
            "role": "user", "content": prompt
        })

        # Save user message to Supabase
        if st.session_state.conversation_id:
            save_message(
                st.session_state.conversation_id,
                "user", prompt, "",
                st.session_state.jwt
            )

        # Get answer
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
                    context, prompt
                )

                answer      = ""
                placeholder = st.empty()
                from groq import Groq
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                stream=True
                )
                for chunk in stream:
                    piece    = chunk.choices[0].delta.content or ""
                    answer  += piece
                    placeholder.markdown(answer + "▌")
                placeholder.markdown(answer)

                sources_text = format_sources(results)
                with st.expander("📎 Sources"):
                    st.text(sources_text)

        # Save assistant message to Supabase
        message_id = None
        if st.session_state.conversation_id:
            saved = save_message(
                st.session_state.conversation_id,
                "assistant", answer, sources_text,
                st.session_state.jwt
            )
            if saved:
                message_id = saved["id"]

        # Feedback buttons
        if message_id:
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"up_{message_id}"):
                    from database import save_feedback
                    save_feedback(
                        st.session_state.user.id,
                        st.session_state.conversation_id,
                        message_id,
                        "up", "",
                        st.session_state.jwt
                    )
                    st.toast("Thanks for the feedback! 👍")
            with col2:
                if st.button("👎", key=f"down_{message_id}"):
                    from database import save_feedback
                    save_feedback(
                        st.session_state.user.id,
                        st.session_state.conversation_id,
                        message_id,
                        "down", "",
                        st.session_state.jwt
                    )
                    st.toast("Thanks for the feedback! 👎")

        st.session_state.messages.append({
            "role": "assistant", "content": answer
        })
        st.session_state.history.append({
            "question": prompt, "answer": answer
        })

# ─────────────────────────────────────────────
# Router — show login or chat
# ─────────────────────────────────────────────
if st.session_state.user is None:
    show_auth_page()
else:
    show_chat_page()