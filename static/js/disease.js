/**
 * static/js/disease.js
 * =====================
 * Handles image upload, drag-and-drop, preview,
 * and CNN disease detection inference for the Disease Scan page.
 * Uses Fetch API with FormData — no page reload.
 */

"use strict";

// ── State ────────────────────────────────────────────────────
let selectedFile     = null;   // File object currently staged
let currentDiseaseId = null;   // DB PK returned after prediction

// ── DOM refs (populated on DOMContentLoaded) ─────────────────
let uploadZone, fileInput, previewImg, previewContainer;
let scanBtn, loader, errorEl;
let placeholderEl, resultPanel;

document.addEventListener("DOMContentLoaded", () => {
  uploadZone       = document.getElementById("upload-zone");
  fileInput        = document.getElementById("img-input");
  previewImg       = document.getElementById("preview-img");
  previewContainer = document.getElementById("preview-container");
  scanBtn          = document.getElementById("scan-btn");
  loader           = document.getElementById("disease-loader");
  errorEl          = document.getElementById("disease-error");
  placeholderEl    = document.getElementById("disease-placeholder");
  resultPanel      = document.getElementById("disease-result");

  bindEvents();
});

// ── Event bindings ───────────────────────────────────────────
function bindEvents() {
  // Click-to-select
  uploadZone?.addEventListener("click", () => fileInput?.click());

  // File input change
  fileInput?.addEventListener("change", e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });

  // Drag-and-drop
  uploadZone?.addEventListener("dragover", e => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });
  uploadZone?.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
  });
  uploadZone?.addEventListener("drop", e => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const file = e.dataTransfer?.files[0];
    if (file) handleFile(file);
  });

  // Paste image from clipboard
  document.addEventListener("paste", e => {
    const item = Array.from(e.clipboardData?.items || [])
      .find(i => i.type.startsWith("image/"));
    if (item) handleFile(item.getAsFile());
  });
}

// ── Handle selected/dropped/pasted file ──────────────────────
function handleFile(file) {
  const ALLOWED = ["image/jpeg", "image/png", "image/webp", "image/gif"];
  const MAX_MB  = 10;

  if (!ALLOWED.includes(file.type)) {
    showError("Unsupported format. Please use JPG, PNG, or WebP.");
    return;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    showError(`File too large. Maximum size is ${MAX_MB} MB.`);
    return;
  }

  clearError();
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    if (previewImg) {
      previewImg.src   = e.target.result;
      previewImg.style.display = "block";
    }
    if (previewContainer) previewContainer.style.display = "block";
  };
  reader.readAsDataURL(file);

  // Reset result panel
  hideResult();
  currentDiseaseId = null;
}

// ── Main inference call ───────────────────────────────────────
async function detectDisease() {
  clearError();

  if (!selectedFile) {
    showError("Please select or drop a leaf image first.");
    return;
  }

  setLoading(true);
  setPlaceholder("🔬 Analysing leaf…");

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const resp = await fetch(DISEASE_PREDICT_URL, {  // URL injected via template
      method:  "POST",
      headers: { "X-CSRFToken": getCookie("csrftoken") },
      body:    formData,
    });

    const data = await resp.json();

    if (!resp.ok) {
      showError(data.error || "Disease detection failed. Check server logs.");
      setPlaceholder("Upload an image and run the scan to see results here.");
      return;
    }

    currentDiseaseId = data.id;
    // Persist for chat page linkage
    sessionStorage.setItem("agribot_disease_id", currentDiseaseId);

    renderResult(data);

  } catch (err) {
    showError("Network error — is the Django server running?");
    setPlaceholder("Upload an image and run the scan to see results here.");
    console.error("Disease detection fetch error:", err);
  } finally {
    setLoading(false);
  }
}

// ── Render prediction result ──────────────────────────────────
function renderResult(data) {
  // Disease name
  const diseaseEl = document.getElementById("disease-name");
  if (diseaseEl) {
    diseaseEl.textContent = data.disease || data.predicted_disease || "Unknown";
  }

  // Plant badge
  const plantBadge = document.getElementById("plant-name-badge");
  if (plantBadge) plantBadge.textContent = "🌿 " + (data.plant || "");

  // Health status badge
  const healthBadge = document.getElementById("healthy-badge");
  if (healthBadge) {
    if (data.is_healthy) {
      healthBadge.textContent = "✅ Healthy";
      healthBadge.className   = "stat-badge badge-green";
    } else {
      healthBadge.textContent = "⚠️ Diseased";
      healthBadge.className   = "stat-badge badge-amber";
    }
  }

  // Confidence
  const confEl = document.getElementById("disease-conf");
  if (confEl) {
    confEl.textContent = `Model confidence: ${parseFloat(data.confidence).toFixed(1)}%`;
  }

  // Top-3 table
  const tbody = document.getElementById("top3-body");
  if (tbody && Array.isArray(data.top_3)) {
    tbody.innerHTML = data.top_3.map((item, i) => {
      const clean = (item.disease || "")
        .replace(/__/g, " — ")
        .replace(/_/g, " ");
      const barColor = i === 0
        ? "var(--green)"
        : (i === 1 ? "var(--amber)" : "var(--text-3)");
      return `
        <tr>
          <td style="padding:8px 0;border-bottom:1px solid var(--border);
                     color:${i === 0 ? "var(--text-1)" : "var(--text-2)"};">
            ${clean}
          </td>
          <td style="padding:8px 0;border-bottom:1px solid var(--border);min-width:120px;">
            <span style="font-family:var(--mono);font-size:12px;
                         color:${barColor};">${parseFloat(item.confidence).toFixed(1)}%</span>
            <div class="conf-bar">
              <div class="conf-fill"
                   style="width:${item.confidence}%;background:${barColor};"></div>
            </div>
          </td>
        </tr>`;
    }).join("");
  }

  // Show panel, hide placeholder
  if (placeholderEl) placeholderEl.style.display = "none";
  if (resultPanel)   resultPanel.style.display   = "block";
}

// ── Remove selected image ─────────────────────────────────────
function removeImage() {
  selectedFile = null;
  if (fileInput)        fileInput.value        = "";
  if (previewImg)       { previewImg.src = ""; previewImg.style.display = "none"; }
  if (previewContainer) previewContainer.style.display = "none";
  hideResult();
  clearError();
  currentDiseaseId = null;
}

// ── Navigate to chat with disease context ─────────────────────
function openChatWithDisease() {
  if (currentDiseaseId) {
    sessionStorage.setItem("agribot_disease_id", currentDiseaseId);
  }
  window.location.href = AGRIBOT_URL;   // injected via template
}

// ── UI helpers ────────────────────────────────────────────────
function setLoading(on) {
  if (scanBtn) scanBtn.disabled = on;
  if (loader)  loader.classList.toggle("show", on);
}

function showError(msg) {
  if (errorEl) errorEl.textContent = msg;
}

function clearError() {
  if (errorEl) errorEl.textContent = "";
}

function setPlaceholder(msg) {
  if (placeholderEl) {
    placeholderEl.textContent  = msg;
    placeholderEl.style.display = "block";
  }
}

function hideResult() {
  if (resultPanel)   resultPanel.style.display   = "none";
  if (placeholderEl) {
    placeholderEl.textContent  = "Upload an image and run the scan to see results here.";
    placeholderEl.style.display = "block";
  }
}

function getCookie(name) {
  const val   = `; ${document.cookie}`;
  const parts = val.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(";").shift();
  return "";
}
