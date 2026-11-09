"""SmartBank AI - Pro BFSI Digital Banking & RAG Assistant
Streamlit Application for local development and Streamlit Community Cloud hosting.
"""

import os
from pathlib import Path
import sys

# Ensure root directory is on python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import streamlit as st

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="SmartBank AI | Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit-specific / environment secret resolution
def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from Streamlit secrets, environment, or user input."""
    # 1. Check Streamlit cloud secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
        if "geminiAPIkey" in st.secrets:
            return str(st.secrets["geminiAPIkey"]).strip()
    except Exception:
        pass

    # 2. Check local environment variable / .env
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("geminiAPIkey") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    # 3. Check session state (if entered in sidebar)
    return st.session_state.get("custom_api_key", "").strip()


from backend.rag_service import (
    clear_user_documents,
    format_rag_context,
    list_user_documents,
    retrieve,
    save_user_document,
)
from backend.chatbot import PRO_BFSI_SYSTEM_PROMPT, gemini_answer, local_rag_answer

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I am your **SmartBank Pro BFSI Advisory Assistant**.\n\n"
                "I can assist you with accounts, fixed deposits, retail loan guidelines, credit scores, "
                "insurance options, and fraud safety. You can also upload any financial document (PDF, TXT, MD, CSV) "
                "in the sidebar to query it with RAG. How may I assist you today?"
            ),
        }
    ]

if "session_user_id" not in st.session_state:
    import uuid
    st.session_state.session_user_id = f"st_user_{uuid.uuid4().hex[:8]}"

user_id = st.session_state.session_user_id
api_key = get_gemini_api_key()

# Custom CSS for clean, modern BFSI styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 950px;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    .doc-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 8px;
        background-color: #e6f7f2;
        color: #087d76;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Controls, Document Upload for RAG, and Status
with st.sidebar:
    st.markdown("### 🏦 SmartBank AI")
    st.caption("Pro BFSI Digital Banking & RAG Assistant")
    st.markdown("---")

    # API Key Configuration (if not configured in secrets or .env)
    if not api_key:
        st.warning("⚠️ No Gemini API key detected.")
        custom_key = st.text_input("Enter Gemini API Key:", type="password", key="custom_api_key_input")
        if custom_key:
            st.session_state["custom_api_key"] = custom_key
            st.rerun()
    else:
        st.success("✅ Gemini LLM Connected", icon="🤖")

    st.markdown("#### 📄 Document Upload (RAG)")
    st.caption("Upload statements, policies, or agreements for grounded analysis.")

    uploaded_file = st.file_uploader(
        "Upload file to RAG",
        type=["pdf", "txt", "md", "csv", "json"],
        label_visibility="collapsed",
        key="rag_file_uploader",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        try:
            safe_name, chunks_count = save_user_document(user_id, uploaded_file.name, file_bytes)
            st.success(f"Indexed **{safe_name}** ({chunks_count} sections) in RAG!")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Display active uploaded documents
    active_docs = list_user_documents(user_id)
    if active_docs:
        st.markdown("**Active Uploaded Documents:**")
        for doc in active_docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"📄 `{doc['filename']}`")
            with col2:
                if st.button("✕", key=f"del_{doc['filename']}", help="Remove from RAG index"):
                    clear_user_documents(user_id, doc["filename"])
                    st.rerun()

    st.markdown("---")

    # Clear Chat button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 Conversation cleared. I am your **SmartBank Pro BFSI Advisory Assistant**. "
                    "How can I assist you today?"
                ),
            }
        ]
        st.rerun()

    st.markdown("---")
    st.caption("Built with Google Gemini & Streamlit.")

# Main Chat Display - Clean & Spacious
st.title("SmartBank AI Banking Assistant")

# Display any active document badge
if active_docs:
    latest_doc = active_docs[-1]["filename"]
    st.markdown(f'<div class="doc-badge">📄 Active Document: <strong>{latest_doc}</strong> (RAG Enabled)</div>', unsafe_allow_html=True)

# Render Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input at the bottom of the chat
user_query = st.chat_input("Type your banking question or ask about your uploaded document...")

if user_query:
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Perform RAG Retrieval
    with st.spinner("Searching BFSI knowledge base and documents..."):
        rag_chunks = retrieve(user_query, limit=4, user_id=user_id)
        knowledge_context = format_rag_context(rag_chunks)

        # 3. Generate response with Gemini LLM
        # Set os.environ if using custom key from sidebar
        if api_key and not os.getenv("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = api_key

        bot_reply = gemini_answer(
            message=user_query,
            knowledge_context=knowledge_context,
            customer_context="",
        )

        # Fallback if Gemini request fails or offline
        if not bot_reply:
            bot_reply = local_rag_answer(user_query, rag_chunks, "")

    # 4. Display assistant message
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

        # Optional expandable sources info
        if rag_chunks:
            with st.expander("🔍 View Retrieved RAG Sources", expanded=False):
                for i, chunk in enumerate(rag_chunks, 1):
                    source_label = chunk.get("source", "Knowledge Base")
                    section_label = chunk.get("section", "General")
                    score = chunk.get("score", "N/A")
                    st.markdown(f"**{i}. {source_label}** — *{section_label}* (Score: {score})")
                    st.caption(chunk.get("text", "")[:300] + "...")

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
