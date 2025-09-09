import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # must come first
import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import lancedb
import markdown
import time
import torch
import re

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
# -----------------------------
#  DATABASE: USERS & CHAT
# -----------------------------
def init_db():
    conn = sqlite3.connect("SQLITE.db")
    c = conn.cursor()
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT
        )
    """)
    # Chat history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            question TEXT,
            answer TEXT,
            category TEXT,
            intent TEXT,  
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(conn, username, password):
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
              (username, hash_password(password)))
    conn.commit()

def verify_user(conn, username, password):
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    return row and row[0] == hash_password(password)

def save_chat(conn, username, question, answer, category, intent):
    c = conn.cursor()
    c.execute("""
        INSERT INTO chat_history (username, question, answer, category, intent)
        VALUES (?, ?, ?, ?, ?)
    """, (username, question, answer, category, intent))
    conn.commit()

def get_user_history(conn, username):
    return pd.read_sql_query(
        "SELECT question, answer, category, timestamp FROM chat_history WHERE username=? ORDER BY timestamp DESC",
        conn, params=(username,)
    )

def clear_user_history(conn, username):
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE username=?", (username,))
    conn.commit()

# -----------------------------
#  LANCEDB: FAQ TABLE
# -----------------------------
@st.cache_resource
def load_faq_table():
    """
    Load the existing LanceDB FAQ table
    """
    db = lancedb.connect("/home/zorin17/Desktop/LLM/")
    table = db.open_table("LANCEDB_FAQ")
    return table

def get_categories(table):
    df = table.to_pandas()
    return sorted(df["category"].dropna().unique().tolist())

def get_intents(table, category_filter=None):
    df = table.to_pandas()
    if category_filter:
        df = df[df["category"] == category_filter]
    return sorted(df["intent"].dropna().unique().tolist())

# -----------------------------
#  RAG (placeholder for now)
# -----------------------------
# Search reranker
def search(query, table, top_k=5, category=None, intent=None):
    s = table.search(query).limit(top_k)
    if category:
        s = s.where(f"category = '{category}'")
    if intent:
        s = s.where(f"intent = '{intent}'")

    result = s.to_list()
    if not result:
        return "[Context 1]:\n(no relevant context found)\n"

    blocks = []
    for i, r in enumerate(result, 1):
        text = (r.get("text") or "").strip()
        # optional: guard against empty rows
        if not text:
            text = "(empty context)"
        blocks.append(f"[Context {i}]:\n{text}\n")
    return "\n".join(blocks)

# -----------------------------
#  Post Processing
# -----------------------------
def enforce_sentence_structure_multi_p(html_text: str) -> str:
    """
    Capitalize sentences inside multiple <p>...</p> blocks.
    Also:
    - Break inline numbered items like "1) ..." or "1. ..." onto new lines
      (works for "1.Please" and even "4click" -> "4. click")
    Handles:
    - First letter of each paragraph
    - After punctuation (.!?)
    - After "Step X:"
    - After numbered lists like "1)" or "2."
    - After colons (:)
    - Title Case for all words inside quotes ("..." or '...')
    """

    def title_case_inside_quotes(match):
        content = match.group(2)
        content_tc = " ".join(w.capitalize() for w in content.split())
        return match.group(1) + content_tc + match.group(3)

    def fix_text(text: str) -> str:
        text = text.strip()

        # Capitalize very first letter of the paragraph
        text = re.sub(r'^[a-z]', lambda m: m.group(0).upper(), text)

        # Capitalize after punctuation (.!?)
        text = re.sub(r'([.!?]\s+)([a-z])',
                      lambda m: m.group(1) + m.group(2).upper(),
                      text)

        # Capitalize after "Step X:"
        text = re.sub(r'(Step\s*\d+:)(\s*)([a-z])',
                      lambda m: m.group(1) + m.group(2) + m.group(3).upper(),
                      text,
                      flags=re.IGNORECASE)

        # Capitalize after colon ":"
        text = re.sub(r'(:\s*)([a-z])',
                      lambda m: m.group(1) + m.group(2).upper(),
                      text)

        # Title Case inside double quotes
        text = re.sub(r'(")([^"]+)(")', title_case_inside_quotes, text)

        # --- Normalize list markers & line breaks ---

        # (A) Fix "bare" numerals like "4click" -> "4. click"
        #     Only for 1-2 digit numbers, and only when followed by a letter,
        #     at start, after <br>, after whitespace, or after sentence end.
        text = re.sub(
            r'(?:(?<=^)|(?<=<br>)|(?<=\s)|(?<=[.!?]\s))([1-9]\d?)(?=[A-Za-z])',
            r'\1. ',
            text
        )

        # (B) Ensure a space after "n)" or "n." if the next char is a letter (handles "1.Please")
        text = re.sub(r'(\d+[\)\.])(?=[A-Za-z])', r'\1 ', text)

        # (C) Break inline numbered items onto new lines (avoid decimals like "2.5")
        #     Not at start and not already after a <br>.
        text = re.sub(
            r'(?<!^)(?<!<br>)\s*((?:[1-9]\d?)[\)\.])(?=(?:\s|[A-Za-z]))',
            r'<br>\1 ',
            text
        )

        # (D) Capitalize the first letter after a numbered marker "n)" or "n."
        text = re.sub(
            r'(([1-9]\d?)[\)\.]\s*)([a-z])',
            lambda m: m.group(1) + m.group(3).upper(),
            text
        )

        return text

    def fix_paragraph(match):
        inner = match.group(1).strip()
        return f"<p>{fix_text(inner)}</p>"

    return re.sub(r"<p>(.*?)</p>", fix_paragraph, html_text,
                  flags=re.DOTALL | re.IGNORECASE)


# placeholder mapping
placeholders = {
#  'account details': '',
#  'cancel purchase': '',
#  'cancellation policy': '',
 'carrier name': 'LALAMOVE',
#  'case number': '',
 'city 1': 'Kuala Lumpur',
 'city 2': 'Penang',
 'city 3': 'Johor',
 'company': 'ShopGrocEazy',
 'company account': 'ShopGrocEazy',
 'company name': 'ShopGrocEazy',
 'companyname': 'ShopGrocEazy',
#  'compensation identifier': '',
#  'confirm cancellation': '',
 'country': 'Malaysia',
 'currency symbol': 'RM',
#  'customer id': '',
 'customer support email': 'support@shopgrocazy.com',
 'customer support hours': '8am - 6pm',
 'customer support phone number': '+60 3-1234 9999',
#  'customer support team': '',
 'cut off time': '12pm',
#  'date range': '',
#  'delivery city': '',
#  'delivery country': '',
#  'delivery date': '',
 'delivery time': '3-5',
#  'destination': '',
 'e-commerce platform 1': 'Shopee',
 'e-commerce platform 2': 'Lazada',
 'e-commerce platform names': 'Shopee',
#  'estimated delivery time': '',
#  'eta': '',
 'expedited delivery time': '1-2',
 'expedited shipping time': '1-2',
 'express delivery time': '1-2',
 'express shipping days': '1-2',
 'express shipping time': '1-2',
#  'live chat support': '',
 'max delivery time': '5',
 'min delivery time': '1',
#  'money amount': '',
#  'my purchases': '',
 'number of days': '3-5',
 'online company portal info': 'ShopGrocEazy',
#  'online customer support channel': '',
#  'online order interaction': '',
#  'online store': '',
#  'order number': '',
#  'order status': '',
#  'order tracker': '',
#  'order tracking': '',
#  'order/claim number': '',
#  'order/claim/compensation': '',
#  'order/invoice number': '',
#  'order/refund/case number': '',
#  'order/refund/transaction': '',
#  'order/transaction/reference number': '',
#  'order/transaction/reimbursement': '',
#  'product/service name': '',
#  'purchase details': '',
#  'purchase history': '',
#  'purchase status': '',
#  'rebate id': '',
#  'rebate identifier': '',
#  'rebate tracking number': '',
#  'reference number': '',
#  'refund amount': '',
 'refund helpline number': '+60 3-1234 9999',
 'refund hotline number': '+60 3-1234 9999',
#  'refund policy': '',
 'refund processing time': '5-7 business days',
#  'reimbursement id': '',
#  'restitution id': '',
#  'retail stores': '',
#  'return policy': '',
 'same-day order time': '12pm',
#  'shipment tracking number': '',
#  'shipping address': '',
 'shipping cut-off time': '12pm',
#  'shipping method': '',
#  'shipping status': '',
 'standard delivery time': '3-5',
 'standard shipping days': '3-5',
 'standard shipping time': '3-5',
#  'store location': '',
 'store pickup time': '1-2',
#  'track order': '',
#  'track reimbursement': '',
#  'tracking number': '',
 'website url': 'https://www.shopgroceazy.com',
 'x-day/money back guarantee period': '7 days (money-back guarantee period)',
#  'zip code': ''

# Additional
 'website': 'https://www.shopgroceazy.com',
 'website chat': 'https://www.shopgroceazy.com',
 'website_url': 'https://www.shopgroceazy.com',
 'online store': 'ShopGrocEazy',
 'business days': '3-5 business days',
 'customer_support_hours': '8am - 6pm',
 'customer service phone number': '+60 3-1234 9999',
}


def replace_braces(val: str) -> str:
    if val is None:
        return val

    # allow spaces inside {{   key   }} and do a case-insensitive lookup
    def repl(m):
        raw = m.group(1)
        key = raw.strip()
        # case-insensitive match against your placeholders keys
        # build a small lookup map once for speed if you want (outside function)
        for k, v in placeholders.items():
            if k.strip().lower() == key.lower():
                return v
        return m.group(0)  # keep original {{key}} if not found

    return re.sub(r"\{\{\s*(.*?)\s*\}\}", repl, val)


# ----------------------------
# Generate function
# ----------------------------
# ---------- your models ----------
MODELS = {
    "llama": {
        "base":    "meta-llama/Llama-3.2-1B",
        "adapter": "qlora-outputs/Llama-3.2-1B-faq",
    },
    "qwen": {
        "base":    "Qwen/Qwen3-0.6B-Base",
        "adapter": "qlora-outputs/Qwen3-0.6B-Base-faq",
    },
    "olmo": {
        "base":    "allenai/OLMo-2-0425-1B",
        "adapter": "qlora-outputs/OLMo-2-0425-1B-faq",
    },
}
# choose model key
MODEL_KEY = "qwen"  # "llama" | "qwen" | "olmo"
BASE_MODEL  = MODELS[MODEL_KEY]["base"]
ADAPTER_DIR = MODELS[MODEL_KEY]["adapter"]

USE_CUDA = torch.cuda.is_available()

# load tokenizer + model once
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- load base in 4-bit + attach LoRA adapter ---
supports_bf16 = USE_CUDA and torch.cuda.get_device_capability(0)[0] >= 8
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if supports_bf16 else torch.float16,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16 if supports_bf16 else torch.float16,
    trust_remote_code=True,
)

hf_model = PeftModel.from_pretrained(base, ADAPTER_DIR)  # <-- attach your QLoRA adapter
# re-enable cache for inference
if getattr(hf_model.config, "use_cache", None) is not True:
    hf_model.config.use_cache = True

hf_model.eval()

USE_CUDA = torch.cuda.is_available()

custom_template = """{% if messages | selectattr('role','equalto','system') | list %}
System: {{ (messages | selectattr('role','equalto','system') | map(attribute='content') | list) | join('\\n') }}
{% endif %}
{% for m in messages %}
{% if m['role'] == 'user' -%}
User: {{ m['content'] }}
{% elif m['role'] == 'assistant' -%}
Assistant: {{ m['content'] }}
{% elif m['role'] == 'tool' -%}
Tool: {{ m['content'] }}
{% elif m['role'] == 'developer' -%}
System: {{ m['content'] }}
{% else -%}
{{ m['role']|capitalize }}: {{ m['content'] }}
{% endif -%}
{% endfor %}
Assistant:"""

def generate(base_prompt, question, context, temperature=0.1, max_new_tokens=512):
    system_content = f"{base_prompt.format(question, context)}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Question: {question}\n\nContext: {context}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=custom_template
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
    input_len = inputs["input_ids"].shape[-1]

    # 🔹 Run in eval + inference mode

    outputs = hf_model.generate(
            **inputs,
            do_sample=(temperature > 0),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=0.95,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
    )

    new_tokens = outputs.sequences[0, input_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return markdown.markdown(response_text)

# --- Full RAG ---
def rag(question, table, base_prompt, category=None, intent=None):
    context = search(question, table, top_k=5, category=category, intent=intent)

    # Apply placeholders to CONTEXT so the model (and the expander) see final values
    context = replace_braces(context or "")

    raw_html = generate(base_prompt, question, context)

    # Structure/tidy the model's HTML
    answer_html = enforce_sentence_structure_multi_p(raw_html or "")

    # Apply placeholders to FINAL ANSWER HTML as well
    answer_html = replace_braces(answer_html)

    return answer_html, context

# -----------------------------
#  STREAMLIT APP
# -----------------------------
st.set_page_config(page_title="ShopGrocEazy Retail Assistant", layout="wide")
conn = init_db()

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# --- LOGIN / REGISTER ---
if not st.session_state.logged_in:
    st.title("🔐 Login to ShopGrocEazy Retail Assistant")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if verify_user(conn, login_username, login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success(f"Welcome, {login_username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        reg_username = st.text_input("New Username", key="reg_user")
        reg_password = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            try:
                add_user(conn, reg_username, reg_password)
                st.success("User registered successfully! Please log in.")
            except sqlite3.IntegrityError:
                st.error("Username already exists.")

# --- MAIN APP ---
else:
    st.sidebar.success(f"✅ Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    # --------- Navigation ---------
    page = st.sidebar.radio("📂 Pages", ["💬 Chatbot", "📊 Data Analysis"])

        # --------- Chatbot Page ---------
    if page == "💬 Chatbot":
        st.title("💬 ShopGrocEazy Retail Assistant")

        # ---------------------------
        # Initialize persistent chat
        # ---------------------------
        if "chat_history" not in st.session_state:
            df_hist = get_user_history(conn, st.session_state.username)
            st.session_state.chat_history = (
                df_hist[["question", "answer"]].iloc[::-1].to_dict("records")
                if not df_hist.empty else []
            )

        # ---------------------------
        # Toggle chat conversation view
        # ---------------------------
        show_chat = st.sidebar.checkbox("💬 Show My Chat History", value=True)

        if show_chat:
            st.markdown("### 💬 Chat Conversation")

            st.markdown("""
                <style>
                .chat-box {
                    max-height: 300px;   /* fixed height */
                    min-height: 300px;
                    overflow-y: auto;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 10px;
                    background-color: #f9f9f9;
                }
                .chat-message-user {
                    background-color: #DCF8C6;
                    padding: 6px 10px;
                    border-radius: 10px;
                    margin: 5px 0;
                }
                .chat-message-assistant {
                    background-color: #FFFFFF;
                    padding: 6px 10px;
                    border-radius: 10px;
                    margin: 5px 0;
                    border: 1px solid #eee;
                }
                </style>
            """, unsafe_allow_html=True)

            # Render chat messages
            chat_html = ""
            if st.session_state.chat_history:
                for chat in st.session_state.chat_history:
                    chat_html += f"<div class='chat-message-user'>👤 {chat['question']}</div>"
                    chat_html += f"<div class='chat-message-assistant'>🤖 {chat['answer']}</div>"
            else:
                chat_html = "<em>ℹ️ No chat history yet. Ask your first question!</em>"

            st.markdown(f"<div class='chat-box'>{chat_html}</div>", unsafe_allow_html=True)

        # ---------------------------
        # Category filter BELOW chat
        # ---------------------------
        table = load_faq_table()
        # Pull once to pandas for filtering
        df_all = table.to_pandas()

        # Category select
        category_options = [None] + sorted(
            df_all["category"].dropna().unique().tolist()
        ) if "category" in df_all.columns else [None]

        category = st.selectbox(
            "Category",
            options=category_options,
            index=0,
            key="category"
        )

        # Build intents list based on (optional) category filter
        if "intent" in df_all.columns:
            if category:
                df_filtered = df_all[df_all["category"] == category]
            else:
                df_filtered = df_all

            intent_options = [None] + sorted(
                df_filtered["intent"].dropna().unique().tolist()
            )
        else:
            intent_options = [None]

        intent = st.selectbox(
            "Intent",
            options=intent_options,
            index=0,
            key="intent_select"  # different key to avoid collisions
        )

        # ---------------------------
        # Question input
        # ---------------------------
        question = st.text_input("Ask a question:", placeholder="e.g. I want to track my order")

        base_prompt = """You are a helpful retail assistant. Your task is to answer the user question using provided contexts as the answer. 
        You must make your response organized and structured.

        User question: {}
        Contexts:
        {}
        """

        # base_prompt = """You are an AI assistant. Answer the user's question using ONLY the information provided in the FAQ contexts.  

        # Rules:  
        # - If the context contains the answer, respond with the answer in a clear and concise way.  
        # - If the context does NOT contain the answer, reply exactly: "I am sorry, I cannot answer your question."  
        # - Do not add information that is not in the contexts.  

        # User Question:  
        # {}  

        # FAQ Contexts:  
        # {}  
        # """

        # ---------------------------
        # Handle Get Answer button
        # ---------------------------
        col1, col2 = st.columns([15, 1])  # 2 equal-width columns

        with col1:
            if st.button("Get Answer") and question:
                with st.spinner("Generating..."):
                    answer, context = rag(question, table, base_prompt, category=category, intent=intent)

                    # Append to persistent chat view
                    st.session_state.chat_history.append({"question": question, "answer": answer})

                    # Save to DB
                    save_chat(conn, st.session_state.username, question, answer, category, intent)

                    # Show inline result (answer + context) just below button
                    st.markdown("### ✅ Answer")
                    st.markdown(answer, unsafe_allow_html=True)

                    with st.expander("📖 Retrieved Contexts"):
                        st.text(context)

        with col2:
            if st.button("Next"):
                st.rerun()
        # ---------------------------
        # Sidebar management
        # ---------------------------
        # Download history (always show)
        df_hist = get_user_history(conn, st.session_state.username)

        if df_hist.empty:
            if st.sidebar.button("⬇️ Download My History (CSV)"):
                st.sidebar.warning("⚠️ No chat history available")
        else:
            st.sidebar.download_button(
                "⬇️ Download My History (CSV)",
                df_hist.to_csv(index=False).encode("utf-8"),
                "chat_history.csv",
                "text/csv",
                key="download_history"
            )

        # Clear chat history
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.confirm_clear = True

        if st.session_state.get("confirm_clear", False):
            st.sidebar.warning("⚠️ Are you sure you want to clear your chat history?")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("✅ Yes, clear", key="yes_clear"):
                    clear_user_history(conn, st.session_state.username)
                    st.session_state.chat_history = []  # clear session view too
                    st.sidebar.success("✅ Chat history cleared")
                    time.sleep(1)  # wait 1 second
                    st.session_state.confirm_clear = False
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", key="cancel_clear"):
                    st.session_state.confirm_clear = False
                    st.rerun()


    # --------- Data Analysis Page ---------
    elif page == "📊 Data Analysis":
        st.title("📊 Data Analysis")
        table = load_faq_table()
        df = table.to_pandas()

        # filters
        categories = [None] + sorted(df["category"].dropna().unique().tolist())
        category = st.selectbox("Category", categories, index=0)
        intents_pool = df if not category else df[df["category"] == category]
        intents = [None] + sorted(intents_pool["intent"].dropna().unique().tolist())
        intent = st.selectbox("Intent", intents, index=0)

        # === Word Cloud ===
        st.subheader("☁️ Word Cloud of User Instruction")

        q_series = df["question"].astype(str)
        if category:
            q_series = q_series[df["category"] == category]
        if intent:
            q_series = q_series[df["intent"] == intent]

        if not q_series.empty:
            def clean_text(s: str) -> str:
                s = re.sub(r"\{\{.*?\}\}", " ", s)
                s = re.sub(r"http\S+|www\.\S+", " ", s)
                s = re.sub(r"[^A-Za-z0-9' ]+", " ", s)
                return s

            text_blob = " ".join(q_series.map(clean_text))
            wc = WordCloud(width=1200, height=500, background_color="white",
                           stopwords=set(STOPWORDS)).generate(text_blob)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # === Top-5 Intents ===
        st.subheader("🔝 Top 5 Intents")
        counts = df["intent"].value_counts().head(5)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        counts.iloc[::-1].plot(kind="barh", ax=ax2)
        ax2.set_xlabel("Count")
        ax2.set_ylabel("Intent")
        st.pyplot(fig2)

