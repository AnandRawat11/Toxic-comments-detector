# 🛡️ Toxic Comment Shield

A real-time toxic comment detection system consisting of a **Chrome browser extension** and a **FastAPI machine learning backend**. The extension automatically detects and hides toxic comments on YouTube, Reddit, and Instagram as you browse, using a custom-trained classical ML model capable of processing comments in under 1 millisecond each.

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Development Phases](#-development-phases)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Datasets Used](#-datasets-used)
- [Model Performance](#-model-performance)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [Running the Chrome Extension](#-running-the-chrome-extension)
- [Example Usage](#-example-usage)
- [Future Improvements](#-future-improvements)

---

## 📖 Project Description

Online comment sections on platforms like YouTube, Reddit, and Instagram are frequently polluted with toxic language — insults, hate speech, and harassment — that degrades the experience for users. Existing platform-level moderation is reactive and inconsistent.

**Toxic Comment Shield** gives users control by running toxicity detection locally in their browser. Comments are sent in batches to a lightweight FastAPI backend, scored by a trained ML model, and hidden automatically — with a click-to-reveal option if the user wants to read a flagged comment.

The system is designed around three principles:
- **Speed** — under 1 ms per comment using classical ML (no transformers)
- **Accuracy** — trained on 180,000+ real-world and synthetic samples from multiple datasets
- **Privacy** — no data is stored; comments are scored and discarded immediately

---

## ✨ Key Features

- **Real-time comment detection** using a `MutationObserver` that reacts instantly to newly loaded comments
- **Multi-platform support** — YouTube, Reddit, and Instagram via site-specific adapter modules
- **Batch prediction API** — processes up to 50 comments per request to minimize network overhead
- **Hard rule pre-filter** — zero-latency detection of extreme toxic phrases (e.g. `"go kill yourself"`) before the ML model runs
- **Leet-speak normalization** — deobfuscates character substitutions like `id1ot → idiot`, `m0ron → moron`
- **Profanity feature engineering** — numeric features for toxic word count and presence, fed alongside TF-IDF
- **Character n-gram TF-IDF** — detects disguised insults via subword patterns
- **Client-side caching** — previously scored comments are never sent to the API again
- **Sensitivity slider** — users can control the toxicity threshold via popup UI
- **Click-to-reveal** — hidden comments can be revealed on demand
- **Persistent preferences** — toggle state and threshold saved via `chrome.storage.sync`

---

## 🏗️ System Architecture

```
User browses YouTube / Reddit / Instagram
         │
         ▼
┌─────────────────────────────┐
│   Chrome Extension           │
│                             │
│  content.js                 │  ← MutationObserver watches DOM
│  sites/youtube.js           │  ← Site-specific selectors
│  sites/reddit.js            │
│  sites/instagram.js         │
│                             │
│  toxicityCache (Map)        │  ← Client-side cache
│  batchQueue []              │  ← 1.5s batch interval
└────────────┬────────────────┘
             │  POST /predict_batch
             │  { "texts": ["comment1", "comment2", ...] }
             ▼
┌─────────────────────────────┐
│   FastAPI Backend            │
│   http://127.0.0.1:8000     │
│                             │
│  Hard Rule Filter           │  ← Instant match on known phrases
│  clean_text()               │  ← Leet-speak + regex normalization
│  build_features()           │  ← Word TF-IDF + Char TF-IDF + Profanity
│  model.predict_proba()      │  ← Logistic Regression
│                             │
└────────────┬────────────────┘
             │  { "scores": [0.02, 0.97, ...], "threshold": 0.5 }
             ▼
┌─────────────────────────────┐
│   Extension applies result  │
│                             │
│  score ≥ threshold → hide   │
│  score < threshold → show   │
│  (click-to-reveal on hover) │
└─────────────────────────────┘
```

---

## 🗺️ Development Phases

### Phase 1 — Rule-Based Filtering
Built the initial content script with a keyword-based toxic word list. Comments containing any listed word were hidden immediately. Established the extension skeleton: `manifest.json`, `content.js`, `popup.html`, `popup.js`, `styles.css`.

**Outcome:** Working prototype that filtered obvious profanity.

---

### Phase 2 — ML Backend Integration
Trained the first machine learning model using a balanced CSV dataset (~56k samples, TF-IDF + Logistic Regression). Integrated a FastAPI backend (`main.py`) that the extension calls for predictions. Replaced the keyword filter with real ML scores.

**Outcome:** Extension now uses probability scores instead of keyword matching.

---

### Phase 3 — Real-Time Comment Monitoring
Replaced the polling `setInterval` scanner with a `MutationObserver` that fires only when the DOM changes. The extension detects newly injected comments instantly without burning CPU cycles.

**Outcome:** Zero-delay detection of dynamically loaded comments.

---

### Phase 4 — Extension User Interface
Built a dark-themed popup UI with a toggle switch (enable/disable filtering) and a sensitivity slider (toxicity threshold 0–100). State is persisted via `chrome.storage.sync` so preferences survive browser restarts.

**Outcome:** User-controllable filtering with no configuration files.

---

### Phase 5 — Probability Scores & Threshold Control
Updated the model from `predict()` to `predict_proba()` so the API returns a continuous toxicity score (0.0–1.0) rather than a binary label. The extension compares the score to the user's slider threshold.

**Outcome:** Tunable sensitivity — users choose their own tolerance level.

---

### Phase 6 — Multi-Platform Support
Introduced a site-adapter system (`sites/youtube.js`, `sites/reddit.js`, `sites/instagram.js`) so each platform's unique DOM structure is handled independently. The extension auto-detects the current site and loads the correct adapter.

**Outcome:** Single codebase, three platforms.

---

### Phase 7 — Performance Optimization
Added a **client-side `toxicityCache`** (Map) to avoid re-scoring identical comments. Introduced a **batch queue** that collects comments for 1.5 seconds and sends them together in a single `/predict_batch` request (max 50 per call). Both the backend and extension were tuned for throughput.

**Outcome:** 200 comments → 4 API requests instead of 200.

---

### Phase 8 — Advanced Feature Engineering
Upgraded from a single word TF-IDF to a combined feature matrix:
- **Word TF-IDF** (1–2 grams, 20k features)
- **Character TF-IDF** (`char_wb`, 3–5 grams, 20k features)
- **Profanity numeric features** (toxic word count + binary flag)

Added a **hard rule pre-filter** for extreme phrases and switched to **CalibratedLinearSVC** for better calibration.

**Outcome:** Char n-grams catch disguised insults. Combined feature space: 40,002 dimensions.

---

### Phase 9 — Dataset Expansion & Merging
Replaced the single CSV with a merged dataset from two public sources: **Jigsaw Toxic Comment Dataset** (159k rows) and **HateXplain** (20k rows). Labels were unified to binary toxicity. Added a synthetic soft-insult dataset (420 hand-curated examples, 10× oversampled) to improve detection of indirect insults.

**Outcome:** 180,000+ training samples; imbalance improved from 9:1 to 5.25:1.

---

### Phase 10 — Leet-Speak & Soft Insult Handling
Added `LEET_MAP` character normalization (`0→o`, `1→i`, `@→a`, `!→i`, etc.) as the first step in `clean_text()` to deobfuscate disguised toxic text before vectorization. Expanded the profanity lexicon with indirect-insult phrases. Added a second round of targeted synthetic examples for patterns like "clown take", "nobody listens to you", and "laughably bad".

**Outcome:** Detection of character-substituted insults improved from 6/10 to 8/10.

---

## 🤖 Machine Learning Pipeline

```
Raw Comment Text
      │
      ▼ clean_text()
  • lowercase
  • leet-speak normalization  (0→o, 1→i, @→a, !→i, $→s, 8→b, ...)
  • remove URLs
  • remove @usernames
  • strip non-ASCII / emojis
  • remove punctuation
      │
      ├─── Word TF-IDF ─────────────────────── 20,000 features
      │    ngram_range=(1,2), stop_words="english", sublinear_tf=True
      │
      ├─── Character TF-IDF (char_wb) ─────── 20,000 features
      │    ngram_range=(3,5), min_df=3, sublinear_tf=True
      │
      └─── Profanity Features ──────────────── 2 features
           • normalised toxic word count
           • binary toxic word flag
                        │
                        ▼ scipy.sparse.hstack
               Feature Matrix (40,002 dimensions)
                        │
                        ▼
         Hard Rule Check → score = 1.0 (instant)
         OR
         LogisticRegression(class_weight="balanced")
                        │
                        ▼
            predict_proba() → toxicity score [0.0 – 1.0]
                        │
                        ▼
         score ≥ threshold → TOXIC
         score <  threshold → CLEAN
```

**Model selection:** Logistic Regression and CalibratedLinearSVC are both trained and evaluated. The winner is selected by toxic-class F1 score. LR is currently selected (F1 = 0.784, Recall = 0.886).

---

## 📦 Datasets Used

### 1. Jigsaw Toxic Comment Classification Challenge
- **Source:** Kaggle / Google Jigsaw
- **File:** `backend/model/Gigsaw Dataset/train.csv`
- **Size:** 159,571 rows
- **Label logic:** `Toxicity = 1` if any of `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` > 0
- **Distribution:** 16,225 toxic (10.2%) / 143,346 non-toxic (89.8%)

### 2. HateXplain
- **Source:** [HateXplain GitHub](https://github.com/hate-speech-and-offensive-language/HateXplain)
- **File:** `backend/model/HateXplain_dataset.json`
- **Size:** 20,148 entries
- **Label logic:** Majority vote across 3 annotators — `hatespeech` or `offensive` → 1, `normal` → 0
- **Distribution:** 12,334 toxic (61.2%) / 7,814 non-toxic (38.8%)

### 3. Synthetic Soft-Insult Dataset
- **File:** `backend/model/synthetic_soft_insults.csv`
- **Size:** 420 base examples (×10 oversampled = 4,200 effective rows)
- **Content:** Hand-curated indirect insults — condescending phrasing, clown-take variants, sarcastic dismissals, leet-speak patterns, brain/intelligence insults
- **Distribution:** 345 toxic / 75 non-toxic

### Merged Dataset
| Metric | Value |
|---|---|
| Total samples | 180,081 |
| Toxic samples | ~28,771 (16.0%) |
| Non-toxic samples | ~151,310 (84.0%) |
| Imbalance ratio | 5.25:1 |
| File | `backend/model/merged_dataset.csv` |

---

## 📊 Model Performance

Evaluated on a held-out 20% test split (35,986 samples):

| Metric | Non-Toxic | Toxic |
|---|---|---|
| Precision | 0.98 | 0.71 |
| Recall | 0.93 | **0.886** |
| F1 Score | 0.95 | **0.784** |
| Accuracy | | **92%** |

**Inference latency:** `0.83 ms` per comment (1,000-comment benchmark) — 7× under the 6 ms target.

**Hard rule pre-filter** catches extreme phrases in `O(k)` time before the model runs.

---

## 🔌 API Endpoints

Base URL: `http://127.0.0.1:8000`

---

### `GET /`
Health check and model info.

**Response:**
```json
{
  "message": "Toxic Comment Shield API Running",
  "version": "4.0",
  "model": "LinearSVC (word + char TF-IDF + profanity features)",
  "threshold": 0.5,
  "hard_rules": 9
}
```

---

### `POST /predict`
Score a single comment.

**Request:**
```json
{
  "text": "you are such an idiot"
}
```

**Response:**
```json
{
  "toxicity_score": 0.9963,
  "toxic": true,
  "source": "model"
}
```

> `source` is `"hard_rule"` when the comment matched a hard-rule phrase, `"model"` otherwise.

---

### `POST /predict_batch`
Score multiple comments in one call (used by the extension).

**Request:**
```json
{
  "texts": [
    "great tutorial, really helpful!",
    "go kill yourself",
    "what a clown take"
  ]
}
```

**Response:**
```json
{
  "scores": [0.0025, 1.0, 0.6574],
  "threshold": 0.5
}
```

---

### `GET /threshold`
Returns the trained model's best classification threshold.

**Response:**
```json
{
  "threshold": 0.5
}
```

---

## 📁 Project Structure

```
toxic-comment-shield/
│
├── README.md
│
├── backend/
│   ├── main.py                    # FastAPI application & prediction endpoints
│   ├── train_model.py             # Full ML training pipeline
│   ├── generate_synthetic.py      # Synthetic soft-insult dataset generator
│   ├── requirements.txt           # Python dependencies
│   │
│   └── model/
│       ├── Gigsaw Dataset/
│       │   └── train.csv          # Jigsaw toxic comment dataset
│       ├── HateXplain_dataset.json # HateXplain dataset (text + annotations)
│       ├── HateXplain_classes.npy  # HateXplain class label names
│       ├── synthetic_soft_insults.csv # Hand-curated indirect insult examples
│       ├── merged_dataset.csv     # Final merged training dataset (180k rows)
│       ├── toxic_model.pkl        # Saved model payload (word_vec + char_vec + model)
│       ├── vectorizer.pkl         # Standalone word TF-IDF vectorizer
│       └── threshold.txt          # Best classification threshold
│
└── extension/
    ├── manifest.json              # Chrome extension manifest (MV3)
    ├── content.js                 # Main content script (observer, cache, batch queue)
    ├── popup.html                 # Extension popup UI
    ├── popup.js                   # Popup logic (toggle, slider, storage)
    ├── styles.css                 # Extension styles
    │
    └── sites/
        ├── youtube.js             # YouTube-specific comment selectors
        ├── reddit.js              # Reddit comment selectors (new + legacy)
        └── instagram.js           # Instagram comment selectors
```

---

## ⚙️ Installation Guide

### Prerequisites
- Python 3.9+
- Google Chrome (or any Chromium-based browser)
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/toxic-comment-shield.git
cd toxic-comment-shield
```

### 2. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
pip install joblib scipy  # additional runtime dependencies
```

### 3. Add Datasets

Place the following files in `backend/model/`:

| File | Source |
|---|---|
| `Gigsaw Dataset/train.csv` | [Kaggle — Jigsaw Toxic Comment Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) |
| `HateXplain_dataset.json` | [HateXplain GitHub](https://github.com/hate-speech-and-offensive-language/HateXplain) |

### 4. Generate Synthetic Data

```bash
python generate_synthetic.py
```

### 5. Train the Model

```bash
python train_model.py
```

Training takes approximately 3–5 minutes. Outputs:
- `model/toxic_model.pkl`
- `model/vectorizer.pkl`
- `model/merged_dataset.csv`
- `model/threshold.txt`

### 6. Start the API Server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Verify the server is running:

```bash
curl http://127.0.0.1:8000/
```

---

## 🔧 Running the Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `extension/` folder from this repository
5. The Toxic Comment Shield icon will appear in your Chrome toolbar
6. Ensure the backend API is running at `http://127.0.0.1:8000`
7. Navigate to YouTube, Reddit, or Instagram — the extension filters comments automatically

> **Tip:** Click the extension icon to open the popup and adjust sensitivity or toggle filtering on/off.

---

## 💡 Example Usage

### Testing via curl

```bash
# Single comment
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "you are an idiot"}'

# Response
{
  "toxicity_score": 1.0,
  "toxic": true,
  "source": "model"
}
```

```bash
# Batch comments
curl -X POST http://127.0.0.1:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "great tutorial, very helpful!",
      "shut up you moron",
      "what a clown take",
      "go kill yourself"
    ]
  }'

# Response
{
  "scores": [0.0025, 1.0, 0.6574, 1.0],
  "threshold": 0.5
}
```

### Interactive API Docs

Open `http://127.0.0.1:8000/docs` for the auto-generated Swagger UI.

---

## 🚀 Future Improvements

| Area | Improvement |
|---|---|
| **Model accuracy** | Fine-tune a small transformer (e.g. DistilBERT) for a cloud endpoint while keeping the fast local model for real-time use |
| **Sarcasm detection** | Add sentiment-inversion heuristics or a dedicated sarcasm classifier |
| **Multilingual support** | Extend the pipeline to detect toxicity in Hindi, Spanish, French, and other languages |
| **Cloud deployment** | Deploy the FastAPI backend to AWS / GCP / Railway so the extension works without a local server |
| **More platforms** | Add support for Twitter/X, TikTok comments, Twitch chat |
| **Whitelist / allowlist** | Let users trust specific channels or creators to skip filtering |
| **Context-aware scoring** | Use the parent comment as context when scoring replies |
| **Model updates** | Periodic automated retraining as new toxic language patterns emerge |
| **Chrome Web Store** | Package the extension for public distribution |

---

## 🧪 Running Tests

Quick sanity check against the live API:

```bash
curl -X POST http://127.0.0.1:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "go kill yourself",
      "you are pathetic trash",
      "great tutorial thanks!",
      "what a clown take",
      "this video is stupid",
      "I love this content!"
    ]
  }'
```

Expected output — scores above `0.5` are toxic:
```json
{
  "scores": [1.0, 1.0, 0.0025, 0.66, 1.0, 0.03],
  "threshold": 0.5
}
```

---

## 📄 License

This project is intended for educational and research purposes. Dataset usage is subject to the terms of the original dataset providers (Jigsaw / Google and HateXplain).
