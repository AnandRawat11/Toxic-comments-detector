"""
Toxic Comment Shield API — v4
FastAPI backend for the browser extension.
Loads the advanced ML pipeline (word TF-IDF + char TF-IDF + hard rules).
"""

import re
import numpy as np
import scipy.sparse as sp
import joblib

from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load pipeline payload ──────────────────────────────────────────────────────
payload    = joblib.load("model/toxic_model.pkl")
word_vec   = payload["word_vec"]
char_vec   = payload["char_vec"]
model      = payload["model"]
THRESHOLD  = payload["threshold"]
HARD_RULES = payload["hard_rules"]

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Toxic Comment Shield API",
    description="Advanced toxicity detection: word + char TF-IDF, hard rules, LinearSVC",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class Comment(BaseModel):
    text: str

class BatchComments(BaseModel):
    texts: List[str]

# ── Text helpers ───────────────────────────────────────────────────────────────
PROFANITY_LEXICON = [
    "idiot","stupid","moron","trash","garbage","loser","dumb",
    "retard","scum","hate","kill","die","worthless","disgusting",
    "imbecile","jerk","pathetic","ugly","freak","shut up",
    "go away","nobody likes you","kill yourself","you suck",
]

LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "!": "i",
    "$": "s", "8": "b",
})

def clean_text(text: str) -> str:
    text = str(text).lower()
    # Leet-speak → plain letters (before ASCII strip eats @ ! etc.)
    text = text.translate(LEET_MAP)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", lambda m: m.group()[1:], text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def hard_rule_check(text: str) -> bool:
    lower = text.lower()
    return any(rule in lower for rule in HARD_RULES)

def profanity_features(texts) -> sp.csr_matrix:
    rows = []
    for text in texts:
        lower = str(text).lower()
        count = sum(1 for w in PROFANITY_LEXICON if w in lower)
        words = lower.split()
        rows.append([count / max(len(words), 1), 1 if count > 0 else 0])
    return sp.csr_matrix(np.array(rows, dtype=np.float32))

def build_features(texts) -> sp.csr_matrix:
    cleaned = [clean_text(t) for t in texts]
    X_word  = word_vec.transform(cleaned)
    X_char  = char_vec.transform(cleaned)
    X_prof  = profanity_features(texts)
    return sp.hstack([X_word, X_char, X_prof], format="csr")

def score_texts(texts: List[str]) -> List[float]:
    X = build_features(texts)
    return model.predict_proba(X)[:, 1].tolist()

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Toxic Comment Shield API Running",
        "version": "4.0",
        "model": "LinearSVC (word + char TF-IDF + profanity features)",
        "threshold": THRESHOLD,
        "hard_rules": len(HARD_RULES)
    }


@app.post("/predict")
def predict(comment: Comment):
    text = comment.text.strip()
    if not text:
        return {"toxicity_score": 0.0, "toxic": False, "source": "empty"}

    if hard_rule_check(text):
        return {"toxicity_score": 1.0, "toxic": True, "source": "hard_rule"}

    score = float(score_texts([text])[0])
    return {
        "toxicity_score": round(score, 4),
        "toxic": score >= THRESHOLD,
        "source": "model"
    }


@app.post("/predict_batch")
def predict_batch(data: BatchComments):
    texts = [t.strip() for t in data.texts if t.strip()]
    if not texts:
        return {"scores": [], "threshold": THRESHOLD}

    scores = []
    for text in texts:
        if hard_rule_check(text):
            scores.append(1.0)
        else:
            scores.append(None)  # placeholder

    # Batch model call for non-hard-rule texts
    non_rule_idx   = [i for i, s in enumerate(scores) if s is None]
    non_rule_texts = [texts[i] for i in non_rule_idx]

    if non_rule_texts:
        model_scores = score_texts(non_rule_texts)
        for i, score in zip(non_rule_idx, model_scores):
            scores[i] = round(float(score), 4)

    return {"scores": scores, "threshold": THRESHOLD}


@app.get("/threshold")
def get_threshold():
    return {"threshold": THRESHOLD}
