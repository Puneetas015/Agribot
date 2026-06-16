/**
 * static/js/crop.js
 * ==================
 * Handles all AJAX interactions for the Crop Recommendation page.
 * Uses the Fetch API — no page reload during inference.
 */

"use strict";

// ── Input field definitions ──────────────────────────────────
const CROP_FIELDS = [
  { id: "N",           label: "Nitrogen",     unit: "mg/kg", min: 0,   max: 140,  step: 1   },
  { id: "P",           label: "Phosphorus",   unit: "mg/kg", min: 0,   max: 145,  step: 1   },
  { id: "K",           label: "Potassium",    unit: "mg/kg", min: 0,   max: 205,  step: 1   },
  { id: "temperature", label: "Temperature",  unit: "°C",    min: -10, max: 55,   step: 0.1 },
  { id: "humidity",    label: "Humidity",     unit: "%",     min: 0,   max: 100,  step: 0.1 },
  { id: "ph",          label: "Soil pH",      unit: "pH",    min: 0,   max: 14,   step: 0.01},
  { id: "rainfall",    label: "Rainfall",     unit: "mm",    min: 0,   max: 300,  step: 1   },
];

// Crop emoji mapping for display flair
const CROP_EMOJI = {
  rice: "🌾", maize: "🌽", chickpea: "🫘", kidneybeans: "🫘",
  pigeonpeas: "🫛", mothbeans: "🫘", mungbean: "🫘", blackgram: "🫘",
  lentil: "🫛", pomegranate: "🍎", banana: "🍌", mango: "🥭",
  grapes: "🍇", watermelon: "🍉", muskmelon: "🍈", apple: "🍎",
  orange: "🍊", papaya: "🍈", coconut: "🥥", cotton: "🌿",
  jute: "🌿", coffee: "☕",
};

// ── DOM references ───────────────────────────────────────────
let predictBtn, loader, errorMsg, resultBox;

document.addEventListener("DOMContentLoaded", () => {
  predictBtn = document.getElementById("predict-btn");
  loader     = document.getElementById("crop-loader");
  errorMsg   = document.getElementById("error-msg");
  resultBox  = document.getElementById("result-box");

  buildForm();
  attachEnterKey();
});

// ── Build input form dynamically ─────────────────────────────
function buildForm() {
  const container = document.getElementById("crop-form");
  if (!container) return;

  container.innerHTML = CROP_FIELDS.map(f => `
    <div class="form-group">
      <label for="${f.id}">
        ${f.label}
        <span style="color:var(--text-3);font-weight:400;">(${f.unit})</span>
      </label>
      <input
        type="number" id="${f.id}" name="${f.id}"
        step="${f.step}" min="${f.min}" max="${f.max}"
        placeholder="e.g. ${f.min + Math.round((f.max - f.min) * 0.4)}"
      />
    </div>
  `).join("");
}

function attachEnterKey() {
  document.querySelectorAll("#crop-form input").forEach(inp => {
    inp.addEventListener("keydown", e => {
      if (e.key === "Enter") predictCrop();
    });
  });
}

// ── Main inference call ──────────────────────────────────────
async function predictCrop() {
  clearError();
  const body = collectInputs();
  if (!body) return;  // validation failed

  setLoading(true);

  try {
    const resp = await fetch(CROP_PREDICT_URL, {   // URL injected via template tag
      method:  "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken":  getCookie("csrftoken"),
      },
      body: JSON.stringify(body),
    });

    const data = await resp.json();

    if (!resp.ok) {
      showError(data.error || "Prediction failed. Check server logs.");
      return;
    }

    renderResult(data);

  } catch (err) {
    showError("Network error — is the Django server running?");
    console.error("Crop prediction fetch error:", err);
  } finally {
    setLoading(false);
  }
}

// ── Collect & validate inputs ────────────────────────────────
function collectInputs() {
  const body = {};

  for (const f of CROP_FIELDS) {
    const el  = document.getElementById(f.id);
    const raw = el ? el.value.trim() : "";

    if (raw === "" || isNaN(raw)) {
      showError(`"${f.label}" is required and must be a number.`);
      el && el.focus();
      return null;
    }

    const val = parseFloat(raw);
    if (val < f.min || val > f.max) {
      showError(`"${f.label}" must be between ${f.min} and ${f.max} ${f.unit}.`);
      el && el.focus();
      return null;
    }

    body[f.id] = val;
  }

  return body;
}

// ── Render prediction result ─────────────────────────────────
function renderResult(data) {
  const cropKey   = (data.crop || "").toLowerCase().replace(/\s+/g, "");
  const emoji     = CROP_EMOJI[cropKey] || "🌱";
  const cropLabel = data.crop.charAt(0).toUpperCase() + data.crop.slice(1);

  // Main result
  document.getElementById("result-crop").innerHTML =
    `${emoji} ${cropLabel}`;
  document.getElementById("result-conf").textContent =
    `Confidence: ${parseFloat(data.confidence).toFixed(1)}%`;

  // Top-3 pills
  const pillContainer = document.getElementById("top3-pills");
  if (pillContainer && Array.isArray(data.top_3)) {
    pillContainer.innerHTML = data.top_3.map((item, i) => {
      const key   = (item.crop || "").toLowerCase().replace(/\s+/g, "");
      const em    = CROP_EMOJI[key] || "🌿";
      const color = i === 0 ? "var(--green)" : "var(--text-2)";
      return `<span class="pill" style="color:${color};">
        ${em} ${item.crop}
        <span style="font-family:var(--mono);font-size:11px;opacity:0.7;">
          ${parseFloat(item.confidence).toFixed(1)}%
        </span>
      </span>`;
    }).join("");
  }

  // Show result box with animation
  resultBox.classList.add("show");
  resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Clear form ───────────────────────────────────────────────
function clearForm() {
  CROP_FIELDS.forEach(f => {
    const el = document.getElementById(f.id);
    if (el) el.value = "";
  });
  resultBox && resultBox.classList.remove("show");
  clearError();

  // Focus first input
  const first = document.getElementById(CROP_FIELDS[0].id);
  first && first.focus();
}

// ── Load example / demo values ───────────────────────────────
function loadExample(cropName) {
  const EXAMPLES = {
    rice:   { N: 20, P: 28, K: 50, temperature: 23.5, humidity: 82, ph: 6.5, rainfall: 200 },
    wheat:  { N: 85, P: 58, K: 41, temperature: 17.0, humidity: 65, ph: 6.2, rainfall: 75  },
    maize:  { N: 77, P: 48, K: 22, temperature: 22.6, humidity: 65, ph: 6.3, rainfall: 85  },
    mango:  { N: 20, P: 27, K: 30, temperature: 31.0, humidity: 50, ph: 6.0, rainfall: 95  },
    banana: { N: 100,P: 82, K: 50, temperature: 27.4, humidity: 80, ph: 5.8, rainfall: 105 },
  };

  const vals = EXAMPLES[cropName.toLowerCase()];
  if (!vals) return;

  Object.entries(vals).forEach(([key, val]) => {
    const el = document.getElementById(key);
    if (el) el.value = val;
  });

  clearError();
  resultBox && resultBox.classList.remove("show");
}

// ── UI helpers ───────────────────────────────────────────────
function setLoading(on) {
  if (predictBtn) predictBtn.disabled = on;
  if (loader)     loader.classList.toggle("show", on);
}

function showError(msg) {
  if (errorMsg) errorMsg.textContent = msg;
}

function clearError() {
  if (errorMsg) errorMsg.textContent = "";
}

function getCookie(name) {
  const val   = `; ${document.cookie}`;
  const parts = val.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(";").shift();
  return "";
}
