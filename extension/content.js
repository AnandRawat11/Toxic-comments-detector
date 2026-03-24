// ──────────────────────────────────────────
//  Toxic Comment Shield — Phase 10
//  Batch Processing + Client-Side Cache
//  Backend: FastAPI @ http://127.0.0.1:8000
// ──────────────────────────────────────────
//  Site adapter globals injected before this
//  file by manifest content_scripts order:
//    youtubeConfig / redditConfig / instagramConfig
// ──────────────────────────────────────────

const API_URL_BATCH = "http://127.0.0.1:8000/predict_batch";
const API_URL_SINGLE = "http://127.0.0.1:8000/predict";
const DEFAULT_THRESHOLD = 0.70;
const BATCH_SIZE        = 20;   // max comments per API call
const BATCH_INTERVAL_MS = 1500; // process queue every 1.5s

// ── Site detection ────────────────────────
function detectSite() {
    const host = window.location.hostname;
    if (host.includes("youtube.com"))   return "youtube";
    if (host.includes("reddit.com"))    return "reddit";
    if (host.includes("instagram.com")) return "instagram";
    return null;
}

const siteConfigs = {
    youtube:   typeof youtubeConfig   !== "undefined" ? youtubeConfig   : null,
    reddit:    typeof redditConfig    !== "undefined" ? redditConfig    : null,
    instagram: typeof instagramConfig !== "undefined" ? instagramConfig : null,
};

const siteName = detectSite();
const config   = siteName ? siteConfigs[siteName] : null;

if (!config) {
    console.log("[Toxic Shield] Unsupported site — extension inactive.");
    throw new Error("Toxic Shield: unsupported site");
}

console.log(`[Toxic Shield] Platform: ${siteName}`);

// ── Client-side cache ─────────────────────
// Key: comment text → Value: toxicity score (0.0–1.0)
// Prevents identical comments from hitting the API more than once.
const toxicityCache = new Map();

// ── Batch queue ───────────────────────────
// Accumulates { element, text } objects until the interval fires.
let commentQueue = [];

// ── User preferences ──────────────────────
function isFilterEnabled() {
    return new Promise(resolve =>
        chrome.storage.sync.get(["filterEnabled"], r =>
            resolve(r.filterEnabled ?? true)));
}

function getThreshold() {
    return new Promise(resolve =>
        chrome.storage.sync.get(["toxicityThreshold"], r =>
            resolve(r.toxicityThreshold ?? DEFAULT_THRESHOLD)));
}

// ── Loading indicator ─────────────────────
function markPending(commentEl) {
    commentEl.style.opacity = "0.5";
    commentEl.title = "Analyzing…";
}

function clearPending(commentEl) {
    commentEl.style.opacity = "";
    commentEl.title = "";
}

// ── Hide toxic comment ────────────────────
function hideToxicComment(commentEl, score) {
    clearPending(commentEl);

    commentEl.dataset.originalText = commentEl.innerText;
    commentEl.style.filter     = "blur(6px)";
    commentEl.style.userSelect = "none";
    commentEl.style.cursor     = "pointer";
    commentEl.title            = "Click to reveal";

    commentEl.addEventListener("click", () => {
        commentEl.style.filter     = "";
        commentEl.style.userSelect = "";
        commentEl.style.cursor     = "";
        commentEl.title            = "";
        commentEl.innerText        = commentEl.dataset.originalText;
        commentEl.parentElement?.querySelector(".toxic-shield-warning")?.remove();
    }, { once: true });

    const parent = commentEl.closest(config.commentContainer) || commentEl.parentElement;
    if (parent && !parent.querySelector(".toxic-shield-warning")) {
        const warning = document.createElement("div");
        warning.className = "toxic-shield-warning";
        warning.style.cssText = `
            background:#7f1d1d; color:#fca5a5; font-size:12px;
            font-family:'Segoe UI',Arial,sans-serif; padding:4px 10px;
            border-radius:4px; margin-bottom:4px;
            display:inline-block; cursor:pointer;
        `;
        warning.textContent =
            `⚠️ Toxic Shield: Comment hidden (score: ${score.toFixed(2)}) — click to reveal`;
        parent.insertBefore(warning, commentEl);
        warning.addEventListener("click", () => commentEl.click(), { once: true });
    }
}

// ── Apply filter result ───────────────────
async function applyFilter(commentEl, score) {
    clearPending(commentEl);
    const threshold = await getThreshold();
    if (score >= threshold) {
        hideToxicComment(commentEl, score);
        console.log(
            `[Toxic Shield][${siteName}] 🚫 ${score.toFixed(2)} ≥ ${threshold.toFixed(2)}: "${commentEl.dataset.originalText?.slice(0, 60) ?? ""}"`
        );
    }
}

// ── Send a batch to /predict_batch ────────
async function sendBatchRequest(batch) {
    // Split into cached vs needs API call
    const toFetch = [];
    const cached  = [];

    batch.forEach(item => {
        if (toxicityCache.has(item.text)) {
            cached.push({ ...item, score: toxicityCache.get(item.text) });
        } else {
            toFetch.push(item);
        }
    });

    // Apply cached results immediately
    cached.forEach(item => applyFilter(item.element, item.score));

    if (toFetch.length === 0) return;

    try {
        const response = await fetch(API_URL_BATCH, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ texts: toFetch.map(i => i.text) })
        });

        if (!response.ok) return;
        const result = await response.json();

        result.scores.forEach((score, idx) => {
            const item = toFetch[idx];
            toxicityCache.set(item.text, score); // cache for future
            applyFilter(item.element, score);
        });

        console.log(
            `[Toxic Shield] ✅ Batch processed: ${toFetch.length} new + ${cached.length} cached`
        );
    } catch {
        // Backend offline — clear pending state gracefully
        toFetch.forEach(item => clearPending(item.element));
    }
}

// ── Process the queue in BATCH_SIZE chunks ─
async function processQueue() {
    if (commentQueue.length === 0) return;

    const enabled = await isFilterEnabled();
    if (!enabled) {
        // Clear pending indicators and discard queue
        commentQueue.forEach(item => clearPending(item.element));
        commentQueue = [];
        return;
    }

    // Take up to BATCH_SIZE items from the front
    const batch = commentQueue.splice(0, BATCH_SIZE);
    sendBatchRequest(batch); // async, don't await — let next interval fire freely
}

// ── Enqueue a comment for processing ──────
function processComment(commentEl) {
    if (commentEl.dataset.checked) return;
    commentEl.dataset.checked = "true";

    const text = commentEl.innerText.trim();
    if (!text) return;

    // If score is cached, skip the queue entirely
    if (toxicityCache.has(text)) {
        applyFilter(commentEl, toxicityCache.get(text));
        return;
    }

    markPending(commentEl);
    commentQueue.push({ element: commentEl, text });
}

// ── Scan all visible comments ─────────────
function scanExistingComments() {
    const comments = document.querySelectorAll(config.commentSelector);
    if (comments.length > 0)
        console.log(`[Toxic Shield][${siteName}] Queuing ${comments.length} comment(s)…`);
    comments.forEach(c => processComment(c));
}

// ── MutationObserver ──────────────────────
function startObserver() {
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType !== 1) return;
                if (node.matches?.(config.commentSelector)) { processComment(node); return; }
                node.querySelectorAll?.(config.commentSelector).forEach(c => processComment(c));
            });
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
    console.log(`[Toxic Shield][${siteName}] 👀 Observer active.`);
}

// ── Entry point ───────────────────────────
window.addEventListener("load", () => {
    console.log(`[Toxic Shield] Phase 10 — Batch + Cache active on ${siteName}.`);
    const startDelay = config.delayMs ?? 4000;
    setTimeout(() => {
        scanExistingComments();
        startObserver();
        // Flush queue every BATCH_INTERVAL_MS
        setInterval(processQueue, BATCH_INTERVAL_MS);
    }, startDelay);
});
