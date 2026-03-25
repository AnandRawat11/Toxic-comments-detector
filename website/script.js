/* ═══════════════════════════════════════════════
   TOXIC COMMENT SHIELD — Shared JavaScript
   API integration + shared UI components
═══════════════════════════════════════════════ */

const API_BASE = 'http://localhost:8000';

// ── Navbar active state ───────────────────────
function setActiveNavLink() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href');
    if ((path.endsWith('index.html') || path === '/' || path.endsWith('/website/')) && href && (href.includes('index') || href === '../index.html' || href === 'index.html')) {
      a.classList.add('active');
    } else if (href && path.includes(href.replace('../', '').replace('.html', ''))) {
      a.classList.add('active');
    }
  });
}

// ── Hamburger menu ────────────────────────────
function initHamburger() {
  const burger = document.querySelector('.hamburger');
  const links  = document.querySelector('.nav-links');
  if (!burger || !links) return;
  burger.addEventListener('click', () => {
    links.classList.toggle('open');
    const spans = burger.querySelectorAll('span');
    spans.forEach(s => s.style.opacity = links.classList.contains('open') ? '0.6' : '1');
  });
  document.addEventListener('click', e => {
    if (!burger.contains(e.target) && !links.contains(e.target)) links.classList.remove('open');
  });
}

// ── Core API call ─────────────────────────────
// Backend expects: { "texts": ["..."] }
// Backend returns: { "scores": [0.92], "threshold": 0.5 }
async function analyzeComments(texts) {
  const res = await fetch(`${API_BASE}/predict_batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts })
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// ── Score classification ──────────────────────
function classifyScore(score) {
  if (score >= 0.7)      return { label: 'Toxic',    cls: 'toxic',  fillCls: 'fill-toxic' };
  if (score >= 0.4)      return { label: 'Mild',     cls: 'mild',   fillCls: 'fill-mild'  };
  return                        { label: 'Safe',     cls: 'safe',   fillCls: 'fill-safe'  };
}

// ── Render score bar ──────────────────────────
function scoreBarHTML(score, fillCls) {
  const pct = Math.round(score * 100);
  return `
    <div class="score-bar-wrap">
      <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;font-size:0.8rem;">
        <span style="color:var(--on-surface-var)">Toxicity Score</span>
        <span style="font-weight:600;font-family:'Manrope',sans-serif">${pct}%</span>
      </div>
      <div class="score-bar-track">
        <div class="score-bar-fill ${fillCls}" style="width:${pct}%"></div>
      </div>
    </div>`;
}

// ── Toast notification ────────────────────────
function showToast(msg, type = 'error') {
  const t = document.createElement('div');
  t.style.cssText = `
    position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;
    padding:0.85rem 1.5rem;border-radius:0.75rem;
    font-size:0.875rem;font-weight:500;font-family:'Inter',sans-serif;
    background:${type === 'error' ? 'rgba(147,0,10,0.9)' : 'rgba(0,167,75,0.9)'};
    color:${type === 'error' ? 'var(--error)' : 'var(--tertiary)'};
    backdrop-filter:blur(12px);border:1px solid ${type === 'error' ? 'rgba(255,180,171,0.2)' : 'rgba(74,225,118,0.2)'};
    box-shadow:0 8px 24px rgba(0,0,0,0.3);animation:fadeUp 0.3s ease;
  `;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ── Single comment analyzer (used on index.html) ──
function initSingleAnalyzer({ inputId, btnId, resultId, loadingId }) {
  const input   = document.getElementById(inputId);
  const btn     = document.getElementById(btnId);
  const result  = document.getElementById(resultId);
  const loading = document.getElementById(loadingId);
  if (!input || !btn || !result) return;

  async function runAnalysis(text) {
    const trimmed = (text || input.value).trim();
    if (!trimmed) { showToast('Please enter a comment to analyze.'); return; }

    btn.disabled = true;
    if (loading) loading.style.display = 'flex';
    result.classList.remove('show');

    const t0 = performance.now();
    try {
      const data  = await analyzeComments([trimmed]);
      const score = data.scores[0];
      const ms    = (performance.now() - t0).toFixed(1);
      const { label, cls, fillCls } = classifyScore(score);

      result.innerHTML = `
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.25rem;flex-wrap:wrap;">
          <div class="result-verdict verdict-${cls}">${label}</div>
          <span class="badge badge-${cls}">${label === 'Safe' ? '✅' : label === 'Mild' ? '⚠️' : '🚫'} ${label}</span>
          <span class="badge badge-info">⚡ ${ms}ms</span>
        </div>
        ${scoreBarHTML(score, fillCls)}
        <div style="margin-top:1.25rem;background:var(--surface-low);border-radius:0.75rem;padding:1rem 1.25rem;">
          <div style="font-size:0.75rem;color:var(--on-surface-var);margin-bottom:0.5rem;font-weight:500;text-transform:uppercase;letter-spacing:.08em;">Analyzed comment</div>
          <div style="font-size:0.9rem;color:var(--on-surface);font-style:italic;">"${trimmed.slice(0, 200)}"</div>
        </div>`;
      result.classList.add('show');
    } catch (e) {
      showToast('⚠️ Cannot reach the API. Make sure the backend is running on port 8000.');
      console.error(e);
    } finally {
      btn.disabled = false;
      if (loading) loading.style.display = 'none';
    }
  }

  btn.addEventListener('click', () => runAnalysis());
  input.addEventListener('keydown', e => { if (e.ctrlKey && e.key === 'Enter') runAnalysis(); });
  return runAnalysis;
}

// ── Example chips ─────────────────────────────
function initExampleChips(chipClass, runFn) {
  document.querySelectorAll(chipClass).forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.dataset.text;
      const input = document.getElementById('comment-input') || document.getElementById('single-input');
      if (input) { input.value = text; input.focus(); }
      if (runFn) runFn(text);
    });
  });
}

// ── Init on DOM ready ─────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setActiveNavLink();
  initHamburger();
});
