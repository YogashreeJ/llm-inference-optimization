/**
 * KuralMind — Frontend Logic
 * Handles screen transitions, language selection, and chat communication.
 */

// ─── State ───────────────────────────────────────────
let selectedLanguage = null;
let isSending = false;

// ─── Screen Navigation ──────────────────────────────
function switchScreen(fromId, toId) {
  const from = document.getElementById(fromId);
  const to = document.getElementById(toId);
  from.classList.add("exit");
  from.classList.remove("active");
  setTimeout(() => {
    from.classList.remove("exit");
    to.classList.add("active");
  }, 400);
}

function goToLanguage() {
  switchScreen("screen-welcome", "screen-language");
}

function goToChat() {
  if (!selectedLanguage) return;
  document.getElementById("chat-lang-badge").textContent =
    selectedLanguage.charAt(0).toUpperCase() + selectedLanguage.slice(1);
  switchScreen("screen-language", "screen-chat");
  setTimeout(() => document.getElementById("chat-input").focus(), 600);
}

// ─── Language Selection ─────────────────────────────
function selectLanguage(card) {
  document.querySelectorAll(".lang-card").forEach((c) => c.classList.remove("selected"));
  card.classList.add("selected");
  selectedLanguage = card.dataset.lang;
  document.getElementById("btn-continue").classList.add("enabled");
}

// ─── Daily Thought ──────────────────────────────────
async function loadDailyThought() {
  try {
    const res = await fetch("/api/thought");
    const data = await res.json();
    const kuralEl = document.getElementById("thought-kural");
    const meaningEl = document.getElementById("thought-meaning");
    const metaEl = document.getElementById("thought-meta");

    kuralEl.classList.remove("thought-loading");
    kuralEl.textContent = data.kural;
    meaningEl.textContent = `"${data.meaning}"`;
    meaningEl.style.display = "";
    metaEl.textContent = `— Kural ${data.kural_number}, ${data.adhigaaram}`;
    metaEl.style.display = "";
  } catch (err) {
    document.getElementById("thought-kural").textContent =
      "Every sunrise is a reminder that you can begin again.";
    document.getElementById("thought-kural").classList.remove("thought-loading");
  }
}

// ─── Chat Functions ─────────────────────────────────
function getTimeStr() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addMessage(text, sender) {
  const container = document.getElementById("chat-messages");
  // Remove welcome message on first interaction
  const welcome = container.querySelector(".chat-welcome");
  if (welcome) welcome.remove();

  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${sender}`;
  msgDiv.innerHTML = `
    <div class="message-bubble">${escapeHtml(text)}</div>
    <div class="message-time">${getTimeStr()}</div>
  `;
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
  return msgDiv;
}

function addTypingIndicator() {
  const container = document.getElementById("chat-messages");
  const msgDiv = document.createElement("div");
  msgDiv.className = "message bot";
  msgDiv.id = "typing-indicator";
  msgDiv.innerHTML = `
    <div class="message-bubble typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function sendMessage() {
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text || isSending) return;

  isSending = true;
  input.value = "";
  autoResizeInput();
  updateSendButton();

  addMessage(text, "user");
  addTypingIndicator();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language: selectedLanguage, message: text }),
    });

    removeTypingIndicator();

    if (!res.ok) throw new Error("Server error");

    // Stream the response
    const container = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = "message bot";
    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "message-bubble";
    const timeDiv = document.createElement("div");
    timeDiv.className = "message-time";
    timeDiv.textContent = getTimeStr();
    msgDiv.appendChild(bubbleDiv);
    msgDiv.appendChild(timeDiv);
    container.appendChild(msgDiv);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      fullText += chunk;
      bubbleDiv.textContent = fullText;
      container.scrollTop = container.scrollHeight;
    }

    if (!fullText.trim()) {
      bubbleDiv.textContent = "I'm sorry, I couldn't generate a response. Please try again.";
    }
  } catch (err) {
    removeTypingIndicator();
    addMessage("⚠️ Something went wrong. Please try again.", "bot");
  }

  isSending = false;
  updateSendButton();
}

// ─── Input Handling ─────────────────────────────────
function handleInputKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResizeInput() {
  const input = document.getElementById("chat-input");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
  updateSendButton();
}

function updateSendButton() {
  const input = document.getElementById("chat-input");
  const btn = document.getElementById("btn-send");
  btn.disabled = !input.value.trim() || isSending;
}

// ─── Init ───────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadDailyThought();
  const input = document.getElementById("chat-input");
  input.addEventListener("input", updateSendButton);
});
