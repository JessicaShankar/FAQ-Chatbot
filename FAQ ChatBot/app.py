"""
ShopAssist – E-Commerce FAQ Chatbot
NLP-based FAQ matching using TF-IDF and cosine similarity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK data on first run (needed on Streamlit Community Cloud)
@st.cache_resource(show_spinner=False)
def _download_nltk_data() -> None:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

_download_nltk_data()



# ── Constants ──────────────────────────────────────────────────────────────────
FAQ_PATH = Path(__file__).with_name("faq_data.json")
SIMILARITY_THRESHOLD = 0.18
FALLBACK_ANSWER = (
    "I'm sorry, I couldn't find a reliable answer to that question. "
    "Please try asking about **orders**, **delivery**, **returns**, "
    "**refunds**, **payments**, or contact our support team directly."
)
TOKENIZER = RegexpTokenizer(r"[a-z0-9]+")
FALLBACK_STOP_WORDS: set[str] = {
    "a", "about", "after", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "but", "by", "can",
    "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "ours", "please", "she", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "to", "up", "us", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours",
}


def preprocess_text(text: str) -> list[str]:
    """Lowercase and tokenize text, removing English stopwords when available."""
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = FALLBACK_STOP_WORDS

    tokens = TOKENIZER.tokenize(text.lower())
    return [token for token in tokens if token not in stop_words]


def load_faqs(path: Path = FAQ_PATH) -> tuple[list[dict[str, str]], int]:
    """Load valid FAQ records and report how many malformed records were skipped."""
    with path.open(encoding="utf-8") as faq_file:
        raw_faqs: Any = json.load(faq_file)

    if not isinstance(raw_faqs, list):
        raise ValueError("FAQ data must be a JSON array of FAQ entries.")

    faqs = []
    skipped = 0
    for item in raw_faqs:
        if not isinstance(item, dict):
            skipped += 1
            continue
        record = {
            field: item.get(field, "").strip()
            if isinstance(item.get(field), str)
            else ""
            for field in ("question", "answer", "category")
        }
        if not all(record.values()):
            skipped += 1
            continue
        faqs.append(record)

    if not faqs:
        raise ValueError("No valid FAQ entries were found in faq_data.json.")
    return faqs, skipped


def build_matcher(faqs: list[dict[str, str]]) -> tuple[TfidfVectorizer, Any]:
    """Fit TF-IDF vectors for the preprocessed FAQ questions."""
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
    )
    faq_vectors = vectorizer.fit_transform([faq["question"] for faq in faqs])
    return vectorizer, faq_vectors


def find_answer(
    question: str,
    faqs: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    faq_vectors: Any,
) -> dict[str, Any]:
    """Return the closest FAQ answer only when cosine similarity clears the threshold."""
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, faq_vectors).flatten()
    best_index = int(scores.argmax())
    similarity = float(scores[best_index])
    matched = similarity >= SIMILARITY_THRESHOLD

    return {
        "answer": faqs[best_index]["answer"] if matched else FALLBACK_ANSWER,
        "similarity": similarity,
        "category": faqs[best_index]["category"] if matched else None,
        "matched": matched,
    }


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopAssist | Customer Support",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --ink:         #1a2421;
    --ink-light:   #3d4e48;
    --muted:       #6b7c75;
    --line:        #e2e9e5;
    --paper:       #ffffff;
    --wash:        #f4f7f5;
    --green:       #1d6b4e;
    --green-mid:   #22855f;
    --green-light: #e8f4ef;
    --green-ring:  #b6deca;
    --tag-bg:      #ddf0e7;
    --tag-ink:     #145c3a;
    --danger:      #b54535;
}
html, body, [data-testid="stApp"] {
    background: var(--paper) !important;
    color: var(--ink);
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--wash) !important;
    border-right: 1px solid var(--line) !important;
}
.brand-wrap { display: flex; align-items: center; gap: 10px; padding: 0 4px; }
.brand-icon {
    width: 40px; height: 40px; border-radius: 11px;
    background: var(--green); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 800; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(29,107,78,.28);
}
.brand-text  { font-size: 18px; font-weight: 700; color: var(--ink); line-height: 1; }
.brand-sub   { font-size: 12px; color: var(--muted); margin-top: 2px; }
.sidebar-eyebrow {
    font-size: 10.5px; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: var(--muted); margin: 20px 0 8px;
}
div.stButton > button {
    width: 100%; border: 1px solid var(--line);
    background: var(--paper); color: var(--ink-light);
    border-radius: 8px; min-height: 38px; text-align: left;
    font-size: 13px; padding: 7px 12px;
    transition: border-color .15s, color .15s, background .15s;
}
div.stButton > button:hover {
    border-color: var(--green-ring); background: var(--green-light); color: var(--green);
}
div.stButton > button[kind="primary"] {
    background: var(--green); border-color: var(--green); color: #fff; font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--green-mid); border-color: var(--green-mid);
}
.page-header { padding: 20px 0 12px; border-bottom: 1px solid var(--line); margin-bottom: 4px; }
.page-title  { font-size: 25px; font-weight: 700; color: var(--ink); margin: 0 0 3px; }
.page-sub    { font-size: 14px; color: var(--muted); margin: 0; }
[data-testid="stChatMessage"] { border-bottom: 1px solid var(--line); padding: 14px 2px; }
[data-testid="stChatMessage"] p { line-height: 1.65; font-size: 14.5px; }
.cat-pill {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    background: var(--tag-bg); color: var(--tag-ink);
    font-size: 11px; font-weight: 600; margin-right: 6px;
}
.sim-label { font-size: 11px; color: var(--muted); }
.sim-low   { font-size: 11px; color: var(--danger); }
.stat-strip {
    display: flex; gap: 20px; font-size: 12px; color: var(--muted);
    padding: 10px 0; flex-wrap: wrap;
}
.stat-item strong { color: var(--ink); font-size: 15px; display: block; }
@media (max-width: 640px) { .page-title { font-size: 21px; } }
</style>
""", unsafe_allow_html=True)

# ── Load data & build matcher ──────────────────────────────────────────────────
try:
    faqs, skipped_faqs = load_faqs()
    vectorizer, faq_vectors = build_matcher(faqs)
except FileNotFoundError:
    st.error(
        f"**FAQ knowledge base not found.** Expected `{FAQ_PATH.name}` in the same "
        "folder as `app.py`. Add the file and restart the app."
    )
    st.stop()
except (json.JSONDecodeError, OSError, ValueError) as exc:
    st.error(f"**Could not load FAQ knowledge base:** {exc}")
    st.stop()

if skipped_faqs:
    st.warning(
        f"⚠️ Skipped {skipped_faqs} invalid FAQ "
        f"{'entry' if skipped_faqs == 1 else 'entries'} while loading."
    )

categories = sorted({faq["category"] for faq in faqs})

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-icon">SA</div>
        <div>
            <div class="brand-text">ShopAssist</div>
            <div class="brand-sub">Customer care, made simple</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="sidebar-eyebrow">Try asking about</p>', unsafe_allow_html=True)

    suggestions = [
        "How can I track my order?",
        "Can I cancel my order?",
        "How do I return a product?",
        "When will I receive my refund?",
        "What payment methods do you accept?",
        "What if I received a damaged product?",
    ]
    for idx, sug in enumerate(suggestions):
        if st.button(sug, key=f"sug_{idx}", use_container_width=True):
            st.session_state.pending_question = sug

    st.divider()
    st.markdown(
        f'<div class="stat-strip">'
        f'<span class="stat-item"><strong>{len(faqs)}</strong>answers</span>'
        f'<span class="stat-item"><strong>{len(categories)}</strong>categories</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Supported topics"):
        for cat in categories:
            st.markdown(f"• {cat}")

# ── Page header ────────────────────────────────────────────────────────────────
hdr_col, btn_col = st.columns([5, 1])
with hdr_col:
    st.markdown("""
    <div class="page-header">
        <p class="page-title">How can we help? 🛍️</p>
        <p class="page-sub">Ask anything about your order, delivery, returns, payments, and more.</p>
    </div>
    """, unsafe_allow_html=True)
with btn_col:
    st.write("")
    st.write("")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

# ── Welcome message (shown only when chat is empty) ────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Welcome to **ShopAssist**! I'm here to help with your shopping questions.\n\n"
            "Ask me about **order tracking**, **cancellations**, **returns**, "
            "**refunds**, **payments**, and more — or pick a suggestion from the sidebar."
        )

# ── Render conversation history ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "similarity" in msg:
            sim = msg["similarity"]
            cat = msg.get("category")
            matched = msg.get("matched", False)
            if matched and cat:
                st.markdown(
                    f'<span class="cat-pill">{cat}</span>'
                    f'<span class="sim-label">cosine similarity: {sim:.2f}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="sim-low">'
                    f'Best cosine similarity: {sim:.2f} — below answer threshold ({SIMILARITY_THRESHOLD})'
                    f'</span>',
                    unsafe_allow_html=True,
                )

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question, e.g. Where is my package?")

pending = st.session_state.pending_question
question_to_answer: str = ""

if pending:
    question_to_answer = pending
    st.session_state.pending_question = None
elif user_input:
    stripped = user_input.strip()
    if not stripped:
        st.warning("⚠️ Please type a question before sending.")
    else:
        question_to_answer = stripped

# ── Match & respond ────────────────────────────────────────────────────────────
if question_to_answer:
    result = find_answer(question_to_answer, faqs, vectorizer, faq_vectors)
    st.session_state.messages.extend([
        {
            "role":    "user",
            "content": question_to_answer,
        },
        {
            "role":       "assistant",
            "content":    result["answer"],
            "similarity": result["similarity"],
            "category":   result["category"],
            "matched":    result["matched"],
        },
    ])
    st.rerun()