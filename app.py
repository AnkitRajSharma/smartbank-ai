"""SmartBank AI - Pro BFSI Digital Banking & RAG Assistant
Streamlit application matching the FastAPI frontend design, colors, typography, and layout.
"""

from datetime import datetime
import html
import os
from pathlib import Path
import sys
import urllib.parse

# Ensure root directory is on python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import markdown
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="SmartBank AI | Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve Gemini API Key from Streamlit Secrets or Environment
def get_gemini_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
        if "geminiAPIkey" in st.secrets:
            return str(st.secrets["geminiAPIkey"]).strip()
    except Exception:
        pass

    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("geminiAPIkey") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    return st.session_state.get("custom_api_key", "").strip()


api_key = get_gemini_api_key()
if api_key and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = api_key

from backend.rag_service import (
    clear_user_documents,
    format_rag_context,
    list_user_documents,
    retrieve,
    save_user_document,
)
from backend.chatbot import gemini_answer, local_rag_answer

# Initialize Session State
WELCOME_MESSAGE = (
    "Hello! 👋 I am your **SmartBank Pro BFSI Advisory Assistant**.\n\n"
    "I can assist you with accounts, deposits, loan estimates, credit scores, "
    "insurance, wealth planning, and fraud protection. Use the **Upload Doc** button in the search bar "
    "below to analyze any statement or policy with RAG. How may I help you today?"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "time": "Just now",
        }
    ]

if "session_user_id" not in st.session_state:
    import uuid
    st.session_state.session_user_id = f"st_user_{uuid.uuid4().hex[:8]}"

user_id = st.session_state.session_user_id

# Handle URL action queries (e.g., ?clear=true, ?del_doc=...)
if st.query_params.get("clear") == "true":
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "time": "Just now",
        }
    ]
    st.query_params.clear()
    st.rerun()

del_param = st.query_params.get("del_doc")
if del_param:
    clear_user_documents(user_id, del_param)
    st.query_params.clear()
    st.rerun()

# Load base CSS from frontend/style.css
css_path = ROOT_DIR / "frontend" / "style.css"
base_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

# Inject styles to match FastAPI UI exactly
st.markdown(
    f"""
    <style>
    {base_css}

    /* Streamlit Chrome Overrides */
    #MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"] {{
        display: none !important;
        visibility: hidden !important;
    }}
    .stApp {{
        background:
            radial-gradient(ellipse at 12% 12%, rgba(55, 182, 165, .13), transparent 33%),
            radial-gradient(ellipse at 92% 90%, rgba(23, 107, 135, .11), transparent 36%),
            #f6f8fa !important;
        font-family: 'Avenir Next', Avenir, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }}

    /* Streamlit Sidebar matches FastAPI aside.sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0d2941, #123c55) !important;
        color: #fff !important;
        min-width: 250px !important;
        max-width: 250px !important;
        border-right: none !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #fff;
    }}
    [data-testid="stSidebarContent"] {{
        padding: 25px 16px 18px !important;
        background: transparent !important;
    }}
    [data-testid="stSidebarContent"] .stButton button {{
        width: 100%;
        margin-top: 14px;
        padding: 11px 12px;
        border: 1px solid rgba(255, 255, 255, .18) !important;
        border-radius: 9px !important;
        color: #dce8ee !important;
        background: transparent !important;
        font-size: 11px !important;
        text-align: left !important;
        font-weight: 600 !important;
        transition: background .18s ease;
    }}
    [data-testid="stSidebarContent"] .stButton button:hover {{
        background: rgba(255, 255, 255, .09) !important;
        color: #fff !important;
        border-color: rgba(255, 255, 255, .3) !important;
    }}

    /* Main Container layout */
    .main .block-container {{
        max-width: 920px !important;
        padding: 24px clamp(12px, 3vw, 36px) 130px !important;
        margin: 0 auto !important;
    }}

    /* Chat Card Frame Enclosing entire assistant experience */
    .chat-card {{
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        max-width: 920px !important;
        height: calc(100vh - 170px) !important;
        min-height: 520px !important;
        overflow: hidden !important;
        border: 1px solid rgba(16, 42, 67, .08) !important;
        border-radius: 22px !important;
        background: #ffffff !important;
        box-shadow: 0 24px 70px rgba(16, 42, 67, .12), 0 3px 12px rgba(16, 42, 67, .04) !important;
        margin: 0 auto !important;
    }}
    .chat-content {{
        display: flex !important;
        flex: 1 !important;
        flex-direction: column !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }}
    .chat-messages {{
        flex: 1 !important;
        overflow-y: auto !important;
        padding: 28px 30px 18px !important;
        scroll-behavior: smooth !important;
    }}

    /* Streamlit Chat Input matched to .composer */
    [data-testid="stBottom"] {{
        background: transparent !important;
        padding-bottom: 12px !important;
    }}
    [data-testid="stBottom"] > div {{
        max-width: 920px !important;
        margin: 0 auto !important;
        padding: 0 !important;
    }}
    [data-testid="stChatInput"] {{
        border: 1px solid #dfe7ec !important;
        border-radius: 14px !important;
        background: #ffffff !important;
        box-shadow: 0 8px 30px rgba(16, 42, 67, .08) !important;
        padding: 6px 10px !important;
        transition: border-color .18s, box-shadow .18s !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: #59bbaa !important;
        box-shadow: 0 0 0 3px rgba(55, 182, 165, .16), 0 8px 30px rgba(16, 42, 67, .08) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        font-size: 13px !important;
        color: #102a43 !important;
        font-family: inherit !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: #9ca9b4 !important;
    }}
    [data-testid="stChatInput"] button {{
        background: #102a43 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        transition: background .18s, transform .18s !important;
    }}
    [data-testid="stChatInput"] button:hover {{
        background: #174765 !important;
    }}
    [data-testid="stChatInput"] button svg {{
        fill: #ffffff !important;
    }}
    [data-testid="stChatInput"] [data-testid="stChatInputFileButton"],
    [data-testid="stChatInput"] button[aria-label="Upload files"] {{
        background: #f8fafc !important;
        border: 1px solid #dfe7ec !important;
        color: #334e68 !important;
        border-radius: 9px !important;
    }}
    [data-testid="stChatInput"] [data-testid="stChatInputFileButton"]:hover,
    [data-testid="stChatInput"] button[aria-label="Upload files"]:hover {{
        background: #edf8f5 !important;
        border-color: #59bbaa !important;
        color: #087d76 !important;
    }}

    /* Bot bubble table and typography overrides */
    .bot-bubble p {{
        margin: 0 0 8px;
        white-space: normal;
    }}
    .bot-bubble p:last-child {{
        margin-bottom: 0;
    }}
    .bot-bubble ul, .bot-bubble ol {{
        margin: 6px 0 8px 18px;
        padding: 0;
    }}
    .bot-bubble li {{
        margin-bottom: 4px;
    }}
    .bot-bubble table {{
        border-collapse: collapse;
        width: 100%;
        margin: 10px 0;
        font-size: 12px;
    }}
    .bot-bubble th, .bot-bubble td {{
        border: 1px solid #dce5ea;
        padding: 6px 10px;
        text-align: left;
    }}
    .bot-bubble th {{
        background: #e4edf1;
        font-weight: 600;
    }}

    /* Footer Banner */
    .st-chat-footer {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 7px;
        margin-top: 10px;
        color: #95a2ad;
        font-size: 10px;
        text-align: center;
    }}
    .st-chat-footer .status-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #45b79d;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar UI - Matches FastAPI aside.sidebar exactly
with st.sidebar:
    st.markdown(
        """
        <aside class="sidebar" style="position: static; height: auto; padding: 0; background: transparent; width: 100%;">
            <div class="brand-lockup" style="padding: 0 7px 22px;">
                <span class="brand-mark">S</span>
                <span><strong>SmartBank AI</strong><small>Personal banking</small></span>
            </div>
            <nav class="side-nav">
                <a href="javascript:void(0);">Overview</a>
                <a class="active" href="javascript:void(0);">Banking assistant</a>
                <a href="javascript:void(0);">Transactions</a>
                <a href="javascript:void(0);">Credit profile</a>
                <a href="javascript:void(0);">Spending analytics</a>
                <a href="javascript:void(0);">Apply for a loan</a>
                <a href="javascript:void(0);">My applications</a>
                <a href="javascript:void(0);">Profile</a>
            </nav>
        </aside>
        """,
        unsafe_allow_html=True,
    )

    # Document management section in sidebar
    st.markdown("<p style='font-size: 10px; font-weight: 700; color: #8faec2; letter-spacing: 1px; margin: 20px 0 8px 4px;'>RAG DOCUMENT STORE</p>", unsafe_allow_html=True)

    active_docs = list_user_documents(user_id)
    if active_docs:
        for doc in active_docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"<span style='font-size: 11px; color: #bdf1e4;'>📄 {html.escape(doc['filename'])}</span>", unsafe_allow_html=True)
            with col2:
                if st.button("✕", key=f"del_{doc['filename']}", help="Remove document"):
                    clear_user_documents(user_id, doc["filename"])
                    st.rerun()
    else:
        st.markdown("<p style='font-size: 11px; color: #7f99ab; margin-left: 4px;'>No custom document attached.</p>", unsafe_allow_html=True)

    if not api_key:
        st.markdown("---")
        st.warning("⚠️ Enter Gemini API Key:")
        custom_key = st.text_input("Gemini Key", type="password", key="custom_key_input", label_visibility="collapsed")
        if custom_key:
            st.session_state["custom_api_key"] = custom_key
            st.rerun()

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    if st.button("Sign out", key="btn_signout"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
                "time": "Just now",
            }
        ]
        clear_user_documents(user_id)
        st.rerun()

# Build Messages HTML
def render_formatted_message(role: str, content: str, time_str: str = "Just now") -> str:
    if role == "assistant":
        body_html = markdown.markdown(content, extensions=["tables", "fenced_code", "nl2br"])
        return f"""
        <div class="message-row bot-row">
            <div class="avatar bot-avatar" aria-hidden="true">S</div>
            <div class="message-group">
                <span class="sender-label">SmartBank Assistant</span>
                <div class="message-bubble bot-bubble">
                    {body_html}
                </div>
                <time class="message-time">{html.escape(time_str)}</time>
            </div>
        </div>
        """
    else:
        escaped_text = html.escape(content).replace("\n", "<br>")
        return f"""
        <div class="message-row user-row">
            <div class="avatar user-avatar" aria-hidden="true">Y</div>
            <div class="message-group">
                <span class="sender-label">You</span>
                <div class="message-bubble user-bubble">
                    <p>{escaped_text}</p>
                </div>
                <time class="message-time">{html.escape(time_str)}</time>
            </div>
        </div>
        """

messages_html = "\n".join(
    render_formatted_message(m["role"], m["content"], m.get("time", "Just now"))
    for m in st.session_state.messages
)

# Active document bar matching FastAPI's #uploaded-doc-bar
doc_bar_html = ""
if active_docs:
    latest_doc = active_docs[-1]["filename"]
    encoded_doc = urllib.parse.quote(latest_doc)
    doc_bar_html = f"""
    <div class="uploaded-doc-bar">
        <span class="uploaded-doc-icon">📄</span>
        <span class="uploaded-doc-label">{html.escape(latest_doc)}</span>
        <span class="uploaded-doc-tag">RAG Indexed</span>
        <a href="?del_doc={encoded_doc}" class="clear-doc-btn" title="Remove document" style="text-decoration: none;">×</a>
    </div>
    """

# Render complete .chat-card matching FastAPI section.chat-card
full_chat_card_html = f"""
<section class="chat-card" aria-label="SmartBank AI chat">
    <header class="chat-header">
        <div class="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" role="img"><path d="M4 12h24L16 5 4 12Zm3 3v9m6-9v9m6-9v9M4 27h24" /></svg>
        </div>
        <div class="brand-copy">
            <h1>SmartBank AI</h1>
            <p>Your Personal Banking Assistant</p>
        </div>
        <a class="clear-button" href="?clear=true" style="text-decoration: none;" title="Clear chat">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6m4-6v6M6 7l1 14h10l1-14M9 7V4h6v3" /></svg>
            <span>Clear chat</span>
        </a>
    </header>
    <div class="chat-content">
        <div class="chat-messages" id="chat-messages" role="log" aria-live="polite" aria-relevant="additions">
            {messages_html}
        </div>
    </div>
    {doc_bar_html}
</section>
"""

st.markdown(full_chat_card_html, unsafe_allow_html=True)

# Render footer security notice
st.markdown(
    """
    <div class="st-chat-footer">
        <span class="status-dot"></span> SmartBank AI can make mistakes. Never share your PIN or password.
    </div>
    """,
    unsafe_allow_html=True,
)

# Search & Composer Input (with native file attachment support matching the FastAPI search area)
prompt_input = st.chat_input(
    placeholder="Type your banking question or attach a document for RAG analysis...",
    accept_file=True,
    file_type=["pdf", "txt", "md", "csv", "json"],
)

if prompt_input:
    user_text = ""
    uploaded_files = []

    if isinstance(prompt_input, dict):
        user_text = prompt_input.get("text", "") or ""
        uploaded_files = prompt_input.get("files", []) or []
    elif hasattr(prompt_input, "text"):
        user_text = getattr(prompt_input, "text", "") or ""
        uploaded_files = getattr(prompt_input, "files", []) or []
    else:
        user_text = str(prompt_input)

    current_time = datetime.now().strftime("%I:%M %p").lstrip("0")

    # 1. Handle file attachment if user uploaded a file in the input
    indexed_doc_msg = ""
    if uploaded_files:
        for f in uploaded_files:
            file_bytes = f.read()
            try:
                safe_name, chunks_count = save_user_document(user_id, f.name, file_bytes)
                indexed_doc_msg = f"📄 Indexed **{safe_name}** ({chunks_count} sections) into your active RAG knowledge base.\n\n"
            except Exception as e:
                indexed_doc_msg = f"⚠️ Could not index **{f.name}**: {e}\n\n"

    # 2. If user provided a question or file
    if user_text.strip() or uploaded_files:
        display_query = user_text.strip() if user_text.strip() else f"Uploaded document: {uploaded_files[0].name}"
        st.session_state.messages.append({"role": "user", "content": display_query, "time": current_time})

        # 3. Perform RAG Retrieval
        rag_query = user_text.strip() if user_text.strip() else "summarize key points of uploaded document"
        rag_chunks = retrieve(rag_query, limit=4, user_id=user_id)
        knowledge_context = format_rag_context(rag_chunks)

        # 4. Generate LLM response
        bot_reply = gemini_answer(
            message=rag_query,
            knowledge_context=knowledge_context,
            customer_context="",
        )

        if not bot_reply:
            bot_reply = local_rag_answer(rag_query, rag_chunks, "")

        final_response = f"{indexed_doc_msg}{bot_reply}" if indexed_doc_msg else bot_reply
        st.session_state.messages.append({"role": "assistant", "content": final_response, "time": current_time})
        st.rerun()
