/**
 * static/js/agribot.js
 * =====================
 * AgriBot Chat — Fetch API integration with the Django /api/chat/ endpoint,
 * which proxies to a local Ollama server running Llama 3.2-Vision.
 *
 * Features:
 *  • Session persistence via localStorage
 *  • Disease scan context (disease_id from sessionStorage)
 *  • Typing indicator with animated dots
 *  • Auto-scroll chat window
 *  • Quick-prompt pills
 *  • Optional: streaming response rendering (toggle ENABLE_STREAMING)
 */

"use strict";

// ── Config ───────────────────────────────────────────────────
const ENABLE_STREAMING = false;   // Set true only if your Django view supports SSE/streaming
const MAX_HISTORY_DISPLAY = 50;   // Max bubbles to keep in DOM

// ── State ────────────────────────────────────────────────────
let sessionId    = localStorage.getItem("agribot_session") || null;
let diseaseId    = sessionStorage.getItem("agribot_disease_id") || null;
let messageCount = 0;
let isWaiting    = false;

// ── DOM refs ─────────────────────────────────────────────────
let chatWindow, chatInput, sendBtn, modelLabel, tokenCounter, contextBanner;

document.addEventListener("DOMContentLoaded", () => {
  chatWindow     = document.getElementById("chat-window");
  chatInput      = document.getElementById("chat-input");
  sendBtn        = document.getElementById("send-btn");
  modelLabel     = document.getElementById("model-label");
  tokenCounter   = document.getElementById("token-counter");
  contextBanner  = document.getElementById("disease-context-banner");

  initChat();
  bindEvents();
});

// ── Initialise chat state ────────────────────────────────────
function initChat() {
  // Show disease context banner if coming from Disease Scan
  if (diseaseId && contextBanner) {
    contextBanner.textContent = ` 🔬 Disease scan #${diseaseId} linked for context.`;
    contextBanner.style.display = "inline";
  }
}

function bindEvents() {
  // Send on Enter (Shift+Enter = newline in textarea; plain input sends)
  chatInput?.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// ── Send a message ────────────────────────────────────────────
async function sendMessage() {
  if (isWaiting) return;

  const text = chatInput?.value.trim();
  if (!text) return;

  chatInput.value = "";
  appendBubble("user", text);
  const typingBubble = appendTypingIndicator();

  isWaiting = true;
  if (sendBtn) sendBtn.disabled = true;

  try {
    const payload = { message: text };
    if (sessionId) payload.session_id = sessionId;
    if (diseaseId) payload.disease_id = parseInt(diseaseId, 10);

    const resp = await fetch(CHAT_URL, {     // URL injected via template
      method:  "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken":  getCookie("csrftoken"),
      },
      body: JSON.stringify(payload),
    });

    const data = await resp.json();
    typingBubble.remove();

    if (!resp.ok) {
      appendBubble("bot",
        `⚠️ Error: ${data.error || "Request failed. Check server logs."}`,
        "error");
      return;
    }

    appendBubble("bot", data.reply);

    // Update state
    sessionId = data.session_id;
    localStorage.setItem("agribot_session", sessionId);

    messageCount++;
    updateMeta(data.model);

    // Trim DOM if too many bubbles
    trimChatHistory();

  } catch (err) {
    typingBubble.remove();
    appendBubble("bot",
      "⚠️ Cannot reach AgriBot. Make sure `ollama serve` is running on port 11434.",
      "error");
    console.error("AgriBot chat error:", err);
  } finally {
    isWaiting = false;
    if (sendBtn) sendBtn.disabled = false;
    chatInput?.focus();
  }
}

// ── Quick-prompt shortcut ─────────────────────────────────────
function askQuick(question) {
  if (!chatInput) return;
  chatInput.value = question;
  sendMessage();
}

// ── Clear entire conversation ─────────────────────────────────
function clearChat() {
  if (!chatWindow) return;

  chatWindow.innerHTML = `
    <div class="chat-bubble bot">
      Chat cleared. How can I help with your farm today?
    </div>`;

  sessionId    = null;
  diseaseId    = null;
  messageCount = 0;

  localStorage.removeItem("agribot_session");
  sessionStorage.removeItem("agribot_disease_id");

  if (contextBanner) contextBanner.style.display = "none";
  updateMeta("");
}

// ── DOM helpers ───────────────────────────────────────────────

/**
 * Append a chat bubble.
 * @param {"user"|"bot"} role
 * @param {string} text
 * @param {string} [variant]  "error" for red-tinted bot bubble
 */
function appendBubble(role, text, variant = "") {
  const div = document.createElement("div");
  div.className = `chat-bubble ${role}`;
  if (variant === "error") div.style.borderColor = "var(--red)";

  // Simple markdown-ish rendering: bold, newlines
  div.innerHTML = formatMessage(text);

  chatWindow?.appendChild(div);
  scrollToBottom();
  return div;
}

/** Append an animated typing indicator; returns the element so caller can remove it. */
function appendTypingIndicator() {
  const div = document.createElement("div");
  div.className = "chat-bubble bot typing-indicator";
  div.innerHTML = `
    <span class="typing-dots">
      <span></span><span></span><span></span>
    </span>`;
  chatWindow?.appendChild(div);
  scrollToBottom();
  return div;
}

/** Convert plain text to safe HTML with basic markdown support. */
function formatMessage(text) {
  // Escape HTML
  let safe = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Bold: **text**
  safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // Italic: *text*
  safe = safe.replace(/\*(.*?)\*/g, "<em>$1</em>");

  // Code: `code`
  safe = safe.replace(/`([^`]+)`/g,
    '<code style="font-family:var(--mono);font-size:12px;background:var(--bg-0);padding:1px 5px;border-radius:4px;">$1</code>');

  // Numbered lists: lines starting with "1. "
  safe = safe.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");
  safe = safe.replace(/(<li>.*<\/li>)/s, "<ol>$1</ol>");

  // Bullet lists: lines starting with "- "
  safe = safe.replace(/^[-•]\s+(.+)$/gm, "<li>$1</li>");

  // Newlines → <br>
  safe = safe.replace(/\n/g, "<br>");

  return safe;
}

function scrollToBottom() {
  if (chatWindow) {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
}

function trimChatHistory() {
  if (!chatWindow) return;
  const bubbles = chatWindow.querySelectorAll(".chat-bubble:not(.typing-indicator)");
  if (bubbles.length > MAX_HISTORY_DISPLAY) {
    const toRemove = bubbles.length - MAX_HISTORY_DISPLAY;
    for (let i = 0; i < toRemove; i++) {
      bubbles[i].remove();
    }
  }
}

function updateMeta(model) {
  if (modelLabel) {
    modelLabel.textContent = model ? `Model: ${model}` : "";
  }
  if (tokenCounter) {
    tokenCounter.textContent =
      messageCount > 0
        ? `${messageCount} message${messageCount !== 1 ? "s" : ""} this session`
        : "";
  }
}

// ── Utility ───────────────────────────────────────────────────
function getCookie(name) {
  const val   = `; ${document.cookie}`;
  const parts = val.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(";").shift();
  return "";
}
