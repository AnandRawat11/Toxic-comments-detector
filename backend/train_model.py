"""
Toxic Comment Shield — Dataset Merge & Retraining (v5)
=======================================================
Data sources:
  1. Jigsaw Toxic Comment Dataset  (model/Gigsaw Dataset/train.csv)
     159,571 rows — multi-label → binary toxicity
  2. HateXplain Dataset            (model/HateXplain_classes.npy)
     NOTE: the .npy file available contains only the 3 class-name
     strings ['hatespeech','normal','offensive']. If the full
     HateXplain JSON (dataset.json) is added to model/, this script
     will merge it automatically.
"""

import re, os, time, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score
)

print("=" * 64)
print("  Toxic Comment Shield — Dataset Merge & Retraining (v5)")
print("=" * 64)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

PROFANITY_LEXICON = [
    # Strong / explicit
    "idiot","stupid","moron","trash","garbage","loser","dumb",
    "retard","scum","hate","kill","die","worthless","disgusting",
    "imbecile","jerk","pathetic","ugly","freak","shut up",
    "go away","nobody likes you","kill yourself","you suck",
    "clown","piece of crap","get lost",
    # Indirect / soft insults
    "clueless","braindead","brain dead","ignorant","ignorance",
    "embarrassing","embarrass","cringe","cringeworthy",
    "no idea what","have no clue","stop talking","just stop",
    "nobody asked","no one asked","delete this","log off",
    "confidently wrong","spectacularly wrong","laughably wrong",
    "bad take","terrible take","worst take","worst comment",
    "out of your depth","out of your league","touch grass",
    "please stop","do everyone a favor","sit this out",
    "uninformed","misinformed","lack of awareness",
    "not as smart","not that smart","not smart",
    "absolutely wrong","completely wrong","deeply wrong",
    "missed the point","missing the point",
]

HARD_RULES = [
    "go kill yourself","kill yourself","you should die",
    "kys","go die","i will kill","i want to kill",
    "hope you die","end your life",
]

TOXIC_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# ══════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════

LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "!": "i",
    "$": "s", "8": "b",
})

def clean_text(text: str) -> str:
    text = str(text).lower()
    # ① Leet-speak → plain letters (before ASCII strip eats @ ! etc.)
    text = text.translate(LEET_MAP)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", lambda m: m.group()[1:], text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ══════════════════════════════════════════════════════════════════════
# 2. LOAD JIGSAW DATASET
# ══════════════════════════════════════════════════════════════════════

print("\n── Loading Jigsaw Dataset ───────────────────────────────────")
jigsaw_path = "model/Gigsaw Dataset/train.csv"
jig = pd.read_csv(jigsaw_path)
print(f"  Raw rows: {len(jig):,}")

# Binary label: 1 if ANY toxicity column > 0
jig["Toxicity"] = (jig[TOXIC_COLS].sum(axis=1) > 0).astype(int)
jig_clean = jig[["comment_text", "Toxicity"]].rename(
    columns={"comment_text": "tweet"}
)
print(f"  Toxic rows  : {jig_clean['Toxicity'].sum():,}")
print(f"  Clean rows  : {(jig_clean['Toxicity']==0).sum():,}")

# ══════════════════════════════════════════════════════════════════════
# 3. LOAD HATEXPLAIN DATASET (optional — requires dataset.json)
# ══════════════════════════════════════════════════════════════════════

print("\n── Loading HateXplain Dataset ───────────────────────────────")

# Check class names from .npy
npy_path = "model/HateXplain_classes.npy"
if os.path.exists(npy_path):
    classes = np.load(npy_path, allow_pickle=True)
    print(f"  Class labels found: {list(classes)}")

HATE_LABEL_MAP = {"hatespeech": 1, "offensive": 1, "normal": 0}
hatexplain_df  = None

# Try to load the actual text data (dataset.json — standard HateXplain format)
json_candidates = [
    "model/HateXplain_dataset.json",   # user-provided filename
    "model/dataset.json",
    "model/HateXplain/dataset.json",
    "model/hatexplain.json",
]
for json_path in json_candidates:
    if os.path.exists(json_path):
        print(f"  Found HateXplain JSON: {json_path}")
        with open(json_path) as f:
            raw = json.load(f)
        rows = []
        for post_id, entry in raw.items():
            try:
                # Each post has 3 annotator labels — take majority vote
                labels    = [a["label"] for a in entry["annotators"]]
                majority  = max(set(labels), key=labels.count)
                toxicity  = HATE_LABEL_MAP.get(majority, 0)
                tokens    = entry.get("post_tokens", [])
                text      = " ".join(tokens)
                if text.strip():
                    rows.append({"tweet": text, "Toxicity": toxicity})
            except (KeyError, TypeError):
                continue
        hatexplain_df = pd.DataFrame(rows)
        print(f"  HateXplain rows loaded: {len(hatexplain_df):,}")
        print(f"  Toxic: {hatexplain_df['Toxicity'].sum():,}")
        break

if hatexplain_df is None:
    print("  ⚠️  HateXplain text data (dataset.json) not found.")
    print("  ℹ️  Proceeding with Jigsaw-only dataset.")
    print("  ℹ️  To add HateXplain: place dataset.json in model/ folder.")

# ── Load synthetic soft-insult data ───────────────────────────────────────────
print("\n── Loading Synthetic Soft-Insult Data ───────────────────────")
synth_path = "model/synthetic_soft_insults.csv"
synth_df   = None
if os.path.exists(synth_path):
    synth_df = pd.read_csv(synth_path)
    # Oversample synthetic data to increase its influence
    synth_df = pd.concat([synth_df] * 10, ignore_index=True)
    print(f"  Synthetic rows (×10 oversample): {len(synth_df):,}")
    print(f"  Toxic: {synth_df['Toxicity'].sum():,}")
else:
    print("  ⚠️  synthetic_soft_insults.csv not found — skipping")

# ══════════════════════════════════════════════════════════════════════
# 4. MERGE DATASETS
# ══════════════════════════════════════════════════════════════════════

print("\n── Merging & Cleaning Dataset ───────────────────────────────")
frames = [jig_clean]
if hatexplain_df is not None:
    frames.append(hatexplain_df)
if synth_df is not None:
    frames.append(synth_df)

df = pd.concat(frames, ignore_index=True)
df = df.dropna(subset=["tweet","Toxicity"])
df["Toxicity"] = df["Toxicity"].astype(int)

# Deduplicate on tweet text
before = len(df)
df = df.drop_duplicates(subset=["tweet"]).reset_index(drop=True)
print(f"  Duplicates removed: {before - len(df):,}")

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text
df["clean_text"] = df["tweet"].apply(clean_text)
df = df[df["clean_text"].str.len() > 2].reset_index(drop=True)

# Print stats
n_total  = len(df)
n_toxic  = df["Toxicity"].sum()
n_clean  = n_total - n_toxic
print(f"\n  Total samples   : {n_total:,}")
print(f"  Toxic     (1)   : {n_toxic:,}  ({n_toxic/n_total*100:.1f}%)")
print(f"  Non-toxic (0)   : {n_clean:,}  ({n_clean/n_total*100:.1f}%)")
print(f"  Imbalance ratio : {n_clean/n_toxic:.2f}:1")

# Save merged dataset
os.makedirs("model", exist_ok=True)
df[["Toxicity","tweet"]].to_csv("model/merged_dataset.csv", index=False)
print(f"\n  ✅ Saved → model/merged_dataset.csv")

# ══════════════════════════════════════════════════════════════════════
# 5. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════

X_raw = df["tweet"].values
y     = df["Toxicity"].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train_raw):,} | Test: {len(X_test_raw):,}")

# ══════════════════════════════════════════════════════════════════════
# 6. VECTORISERS
# ══════════════════════════════════════════════════════════════════════

word_vec = TfidfVectorizer(
    analyzer="word", ngram_range=(1,2),
    max_features=20_000, stop_words="english",
    sublinear_tf=True, min_df=2,
)
char_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3,5),
    max_features=20_000, sublinear_tf=True, min_df=3,
)

def profanity_features(texts) -> sp.csr_matrix:
    rows = []
    for text in texts:
        lower = str(text).lower()
        count = sum(1 for w in PROFANITY_LEXICON if w in lower)
        words = lower.split()
        rows.append([count / max(len(words),1), 1 if count > 0 else 0])
    return sp.csr_matrix(np.array(rows, dtype=np.float32))

def build_features(texts, fit=False) -> sp.csr_matrix:
    cleaned = [clean_text(t) for t in texts]
    if fit:
        Xw = word_vec.fit_transform(cleaned)
        Xc = char_vec.fit_transform(cleaned)
    else:
        Xw = word_vec.transform(cleaned)
        Xc = char_vec.transform(cleaned)
    Xp = profanity_features(texts)
    return sp.hstack([Xw, Xc, Xp], format="csr")

print("\n── Building Feature Matrices ────────────────────────────────")
print("  Fitting vectorisers…")
X_train = build_features(X_train_raw, fit=True)
X_test  = build_features(X_test_raw)
print(f"  Feature shape: {X_train.shape}")

# ══════════════════════════════════════════════════════════════════════
# 7. TRAIN & COMPARE MODELS
# ══════════════════════════════════════════════════════════════════════

print("\n── Model Training ───────────────────────────────────────────")

lr = LogisticRegression(
    C=1.0, max_iter=1000, solver="lbfgs",
    class_weight="balanced", random_state=42, n_jobs=-1
)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_f1    = f1_score(y_test, lr_preds, pos_label=1)

print(f"\n[Model 1] Logistic Regression")
print(classification_report(y_test, lr_preds, target_names=["Non-Toxic","Toxic"]))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, lr_preds)}")

svc = CalibratedClassifierCV(
    LinearSVC(C=0.5, max_iter=2000, class_weight="balanced", random_state=42), cv=3
)
svc.fit(X_train, y_train)
svc_preds = svc.predict(X_test)
svc_f1    = f1_score(y_test, svc_preds, pos_label=1)

print(f"\n[Model 2] LinearSVC (Calibrated)")
print(classification_report(y_test, svc_preds, target_names=["Non-Toxic","Toxic"]))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, svc_preds)}")

# ══════════════════════════════════════════════════════════════════════
# 8. THRESHOLD OPTIMISATION
# ══════════════════════════════════════════════════════════════════════

print(f"\n── Threshold Optimisation (LR) ──────────────────────────────")
lr_probs    = lr.predict_proba(X_test)[:, 1]
best_thresh = 0.5
best_f1_val = 0.0

for t in [0.30, 0.35, 0.40, 0.45, 0.50]:
    preds = (lr_probs >= t).astype(int)
    f1    = f1_score(y_test, preds, pos_label=1)
    rec   = recall_score(y_test, preds, pos_label=1)
    prec  = precision_score(y_test, preds, pos_label=1, zero_division=0)
    mark  = " ← best" if f1 > best_f1_val else ""
    print(f"  t={t:.2f} | F1={f1:.4f} | Recall={rec:.4f} | Prec={prec:.4f}{mark}")
    if f1 > best_f1_val:
        best_f1_val = f1
        best_thresh = t

print(f"\n  Best LR threshold : {best_thresh}  (F1={best_f1_val:.4f})")

# ══════════════════════════════════════════════════════════════════════
# 9. MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════

print(f"\n── Model Selection ──────────────────────────────────────────")
lr_rec  = recall_score(y_test, lr_preds, pos_label=1)
svc_rec = recall_score(y_test, svc_preds, pos_label=1)
print(f"  LR  F1={lr_f1:.4f}  Recall={lr_rec:.4f}")
print(f"  SVC F1={svc_f1:.4f}  Recall={svc_rec:.4f}")

if svc_f1 >= lr_f1:
    chosen_name, chosen_model, final_thresh = "LinearSVC (Calibrated)", svc, 0.5
    print(f"  ✅ LinearSVC selected  (threshold={final_thresh})")
else:
    chosen_name, chosen_model, final_thresh = "Logistic Regression", lr, best_thresh
    print(f"  ✅ Logistic Regression selected  (threshold={final_thresh})")

# ══════════════════════════════════════════════════════════════════════
# 10. HARD RULE FILTER + PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def hard_rule_check(text: str) -> bool:
    lower = text.lower()
    return any(rule in lower for rule in HARD_RULES)

def predict_toxicity(text: str) -> dict:
    if hard_rule_check(text):
        return {"toxicity_score": 1.0, "is_toxic": True, "source": "hard_rule"}
    X = build_features([text])
    score = float(chosen_model.predict_proba(X)[0][1])
    return {"toxicity_score": round(score,4), "is_toxic": score >= final_thresh, "source": "model"}

# ══════════════════════════════════════════════════════════════════════
# 11. SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════

print(f"\n── Sanity Checks ────────────────────────────────────────────")
checks = [
    ("go kill yourself",            True),
    ("you are pathetic trash",      True),
    ("great tutorial thanks!",      False),
    ("this video is stupid",        True),
    ("I love this content!",        False),
    ("you are a complete moron",    True),
]
for text, expected in checks:
    res = predict_toxicity(text)
    ok  = "✓" if res["is_toxic"] == expected else "✗"
    src = f"[{res['source']}]"
    lbl = "🚫 TOXIC" if res["is_toxic"] else "✅ CLEAN"
    print(f"  {ok} {lbl} ({res['toxicity_score']:.2f}) {src}  →  {text}")

# ══════════════════════════════════════════════════════════════════════
# 12. LATENCY BENCHMARK
# ══════════════════════════════════════════════════════════════════════

print(f"\n── Latency Benchmark (n=1000) ───────────────────────────────")
samples = df["tweet"].sample(1000, random_state=99).tolist()
_ = predict_toxicity(samples[0])   # warm-up

t0 = time.perf_counter()
for s in samples:
    predict_toxicity(s)
elapsed = time.perf_counter() - t0

avg_ms = elapsed / 1000 * 1000
print(f"  Total   : {elapsed*1000:.1f} ms for 1000 comments")
print(f"  Average : {avg_ms:.3f} ms/comment")
print(f"  Target  : < 6 ms  {'✅ PASS' if avg_ms < 6 else '⚠️  SLOW'}")

# ══════════════════════════════════════════════════════════════════════
# 13. SAVE ARTEFACTS
# ══════════════════════════════════════════════════════════════════════

payload = {
    "word_vec":   word_vec,
    "char_vec":   char_vec,
    "model":      chosen_model,
    "threshold":  final_thresh,
    "hard_rules": HARD_RULES,
}
joblib.dump(payload, "model/toxic_model.pkl")
joblib.dump(word_vec, "model/vectorizer.pkl")
with open("model/threshold.txt", "w") as f:
    f.write(str(final_thresh))

print(f"\n  ✅ model/toxic_model.pkl   saved")
print(f"  ✅ model/vectorizer.pkl    saved")
print(f"  ✅ model/threshold.txt     saved  ({final_thresh})")

print(f"\n{'='*64}")
print(f"  Model      : {chosen_name}")
print(f"  Dataset    : {n_total:,} samples")
print(f"  Toxic F1   : {max(lr_f1, svc_f1):.4f}")
print(f"  Latency    : {avg_ms:.3f} ms/comment")
print(f"{'='*64}\n")
