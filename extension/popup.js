const toggle        = document.getElementById("toggleFilter");
const status        = document.getElementById("status");
const slider        = document.getElementById("thresholdSlider");
const thresholdBadge = document.getElementById("thresholdValue");
const sensitivityHint = document.getElementById("sensitivityHint");

// ── Load saved preferences ────────────────
chrome.storage.sync.get(["filterEnabled", "toxicityThreshold"], (result) => {
    const enabled   = result.filterEnabled    ?? true;
    const threshold = result.toxicityThreshold ?? 0.70;

    toggle.checked = enabled;
    slider.value   = threshold;

    updateStatus(enabled);
    updateSlider(threshold);
});

// ── Toggle ────────────────────────────────
toggle.addEventListener("change", () => {
    const enabled = toggle.checked;
    chrome.storage.sync.set({ filterEnabled: enabled });
    updateStatus(enabled);
});

// ── Slider ────────────────────────────────
slider.addEventListener("input", () => {
    const value = parseFloat(slider.value);
    chrome.storage.sync.set({ toxicityThreshold: value });
    updateSlider(value);
});

// ── UI helpers ────────────────────────────
function updateStatus(enabled) {
    status.textContent = enabled ? "🟢 Filter is ON" : "🔴 Filter is OFF";
    status.className   = enabled ? "status on" : "status off";
}

function updateSlider(value) {
    thresholdBadge.textContent = value.toFixed(2);

    let hint = "";
    if (value <= 0.40)      hint = "Very strict — many comments filtered";
    else if (value <= 0.60) hint = "Strict — most toxic comments hidden";
    else if (value <= 0.75) hint = "Balanced — clearly toxic comments hidden";
    else if (value <= 0.85) hint = "Lenient — only very toxic comments hidden";
    else                    hint = "Very lenient — only extreme content hidden";

    sensitivityHint.textContent = hint;
}
