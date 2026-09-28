"""
Offline NLP test for ShopAssist FAQ matching.
Tests the 8 required questions from the project spec.
"""
import json
import sys
from pathlib import Path

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Load FAQ ──────────────────────────────────────────────────────────────────
faqs = json.loads(Path("faq_data.json").read_text(encoding="utf-8"))
print(f"Loaded {len(faqs)} FAQs across {len({f['category'] for f in faqs})} categories\n")

# ── Preprocessing ─────────────────────────────────────────────────────────────
TOKENIZER = RegexpTokenizer(r"[a-z0-9]+")
FALLBACK_SW = {
    "a", "an", "and", "the", "is", "it", "i", "to", "of", "in", "for",
    "do", "my", "what", "how", "can", "will", "be", "that",
}

def preprocess(text: str) -> list[str]:
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        sw = FALLBACK_SW
    return [t for t in TOKENIZER.tokenize(text.lower()) if t not in sw]

# ── Build TF-IDF ──────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    tokenizer=preprocess, preprocessor=None, token_pattern=None,
    lowercase=False, ngram_range=(1, 2),
)
matrix = vectorizer.fit_transform([f["question"] for f in faqs])

# ── Test cases (question, should_match: bool) ─────────────────────────────────
THRESHOLD = 0.18
tests = [
    ("Where is my package?",             True),   # → Delivery
    ("I want to cancel my order",        True),   # → Order cancellation
    ("Can I get my money back?",         True),   # → Refunds
    ("How do I return something?",       True),   # → Returns
    ("What cards can I use?",            True),   # → Payment
    ("My product arrived broken",        True),   # → Damaged or wrong products
    ("Can I change my delivery address?",True),   # → Shipping address
    ("Who won the cricket match?",       False),  # → Fallback (no match)
]

passed = 0
for q, should_match in tests:
    vec = vectorizer.transform([q])
    scores = cosine_similarity(vec, matrix).flatten()
    best = int(scores.argmax())
    sim = float(scores[best])
    matched = sim >= THRESHOLD
    cat = faqs[best]["category"] if matched else None
    preview = (faqs[best]["answer"][:70] + "…") if matched else "(fallback response)"

    ok = matched == should_match
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1

    print(f"[{status}] \"{q}\"")
    print(f"       category={cat!r}  sim={sim:.3f}")
    print(f"       answer: {preview}")
    print()

print("=" * 60)
print(f"Result: {passed}/{len(tests)} tests passed")
sys.exit(0 if passed == len(tests) else 1)
