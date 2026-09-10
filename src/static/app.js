const state = {
  // Start each app launch at sign-in; users authenticate explicitly.
  token: "",
  user: null,
  signup: false,
  busy: false,
  sessions: [],
  cursor: null,
  active: null,
  messages: [],
  prefs: {},
  status: "",
  abort: null,
};

localStorage.removeItem("thread_token");

function applyTheme(theme) {
  const light = theme === "light";
  document.body.classList.toggle("light-theme", light);
  const button = el("theme-toggle");
  if (button) {
    button.textContent = light ? "Dark mode" : "Light mode";
    button.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
  }
}

const labels = {
  clothing_size: "Clothing size",
  budget: "Budget",
  color_preference: "Favorite colors",
  favorite_brands: "Favorite brands",
  style_notes: "Personal style",
};
const prompts = ["Find black dresses under $40", "Comfortable shoes for long days", "I like simple, understated styles"];
const el = (id) => document.getElementById(id);

function resizeQuestion() {
  const input = el("question");
  if (!input) return;
  input.style.height = "auto";
  const lineHeight = parseFloat(getComputedStyle(input).lineHeight) || 24;
  const maxHeight = lineHeight * 4 + 16;
  input.style.height = `${Math.min(input.scrollHeight, maxHeight)}px`;
  input.style.overflowY = input.scrollHeight > maxHeight ? "auto" : "hidden";
}

async function api(path, method = "GET", data) {
  const headers = {"Content-Type": "application/json"};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(path, {method, headers, body: data === undefined ? undefined : JSON.stringify(data)});
  const text = await response.text();
  const result = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const detail = Array.isArray(result.detail)
      ? result.detail.map((item) => item.msg || item.message || JSON.stringify(item)).join(" ")
      : result.detail;
    throw new Error(typeof detail === "string" ? detail : "Please check your input and try again.");
  }
  return result;
}

async function readEvents(response, onEvent) {
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === "string" ? data.detail : "The request failed.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", complete = false;
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done}).replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const lines = block.split("\n");
        const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
        const data = lines.filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
        if (!event || !data) continue;
        const parsed = JSON.parse(data);
        if (event === "error") throw new Error(parsed.message);
        if (event === "done") complete = true;
        onEvent(event, parsed);
      }
      if (done) break;
    }
    if (!complete) throw new Error("Connection ended before the answer was complete. Please retry.");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

function setError(id, message) {
  const node = el(id);
  node.textContent = message || "";
  node.hidden = !message;
}

function setBusy(value) {
  state.busy = value;
  document.body.classList.toggle("body-busy", value);
  el("conversation")?.setAttribute("aria-busy", String(value));
  document.querySelectorAll("button,input,select,textarea").forEach((node) => {
    // Keep navigation and theme controls usable while a chat response streams.
    // Keep the composer editable while a response streams so the user can prepare
    // the next question. `send()` still rejects duplicate submissions while busy.
    if (node.id !== "stop" && node.id !== "question" && !node.closest(".sidebar") && node.id !== "theme-toggle") node.disabled = value;
  });
  el("stop").hidden = !value;
  el("send").hidden = value;
  const authSubmit = el("auth-submit");
  if (authSubmit) {
    authSubmit.setAttribute("aria-busy", String(value));
    authSubmit.innerHTML = value
      ? '<span class="button-spinner" aria-hidden="true"></span><span>Signing in…</span>'
      : `<span>${state.signup ? "Create account" : "Sign in"}</span>`;
  }
  if (!value) document.querySelectorAll("button,input,select,textarea").forEach((node) => node.disabled = false);
}

function showApp() {
  el("auth-shell").hidden = !!state.user;
  el("app-shell").hidden = !state.user;
  if (!state.user) document.body.classList.remove("prefs-open");
  document.body.classList.toggle("auth-view", !state.user);
  el("boot-loading")?.setAttribute("hidden", "");
  updateChatLayoutState();
}

function updateChatLayoutState() {
  document.body.classList.toggle("chat-empty", Boolean(state.user) && state.messages.length === 0);
  document.body.classList.toggle("chat-active", Boolean(state.user) && state.messages.length > 0);
}

function renderAuth() {
  el("auth-title").textContent = state.signup ? "Make yourself at home." : "Welcome back.";
  el("auth-subtitle").textContent = state.signup ? "Create your account to start finding your style." : "Your conversations and preferences are waiting.";
  const nameInput = el("name-label").querySelector("input");
  el("name-label").hidden = !state.signup;
  nameInput.required = state.signup;
  nameInput.disabled = !state.signup;
  if (!state.signup) nameInput.value = "";
  if (!state.busy) el("auth-submit").innerHTML = `<span>${state.signup ? "Create account" : "Sign in"}</span>`;
  el("toggle-auth").textContent = state.signup ? "Already have an account? Sign in" : "New here? Create an account";
  el("auth-form").querySelector("[name=password]").autocomplete = state.signup ? "new-password" : "current-password";
}

function renderAccount() {
  const name = state.user?.name || state.user?.username || "";
  el("account-name").textContent = name;
  el("avatar").textContent = name ? name[0].toUpperCase() : "";
}

function renderSessions() {
  const nav = el("sessions");
  nav.replaceChildren();
  state.sessions.forEach((session) => {
    const row = document.createElement("div");
    row.className = "session-row";
    const open = document.createElement("button");
    open.type = "button";
    open.textContent = session.title;
    open.className = state.active === session.id ? "selected" : "";
    open.addEventListener("click", () => openSession(session.id));
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "x";
    del.setAttribute("aria-label", `Delete ${session.title}`);
    del.addEventListener("click", () => deleteSession(session.id));
    row.append(open, del);
    nav.append(row);
  });
  el("load-more").hidden = !state.cursor;
}

function renderPrefs() {
  const select = el("preference-field");
  if (!select.children.length) {
    Object.entries(labels).forEach(([key, label]) => select.add(new Option(label, key)));
  }
  const list = el("prefs-list");
  list.replaceChildren();
  Object.entries(state.prefs).forEach(([key, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = labels[key] || key;
    const definition = document.createElement("dd");
    definition.textContent = typeof value === "string" ? value : Array.isArray(value) ? value.join(", ") : JSON.stringify(value);
    const clear = document.createElement("button");
    clear.type = "button";
    clear.textContent = "Clear";
    clear.setAttribute("aria-label", `Clear ${labels[key] || key}`);
    clear.addEventListener("click", async () => {
      try {
        state.prefs = await api("/api/preferences", "PATCH", {field: key, action: "clear"});
        renderPrefs();
      } catch (error) { setError("app-error", error.message); }
    });
    row.append(term, definition, clear);
    list.append(row);
  });
}

function messageNode(message, index) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = message.role === "user" ? "You" : "FitFinder AI";
  const text = document.createElement("p");
  text.className = "message-text";
  const content = message.content || (state.busy && index === state.messages.length - 1 ? (state.status === "generating" ? "Writing the answer from retrieved evidence..." : "Retrieving catalog and review evidence...") : "");
  renderMessageText(text, content, message.role === "assistant");
  article.append(label, text);
  if (message.products?.length) {
    const grid = document.createElement("div");
    grid.className = "product-grid";
    message.products.forEach((product) => grid.append(productCard(product)));
    article.append(grid);
  }
  if (message.citations?.length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `Read the review evidence (${message.citations.length})`;
    details.append(summary);
    message.citations.forEach((citation) => {
      const quote = document.createElement("blockquote");
      const small = document.createElement("small");
      small.textContent = citation.id;
      const p = document.createElement("p");
      p.textContent = citation.excerpt;
      quote.append(small, p);
      details.append(quote);
    });
    article.append(details);
  }
  if (message.suggested_relaxations?.length) {
    const row = document.createElement("div");
    row.className = "relaxations";
    message.suggested_relaxations.forEach((relaxation) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = relaxation;
      button.addEventListener("click", () => send(undefined, relaxation));
      row.append(button);
    });
    article.append(row);
  }
  return article;
}

// Render the small Markdown subset emitted by the answer finalizer without
// inserting raw model output into innerHTML. This keeps formatting readable
// while avoiding an XSS risk from model-generated content.
function renderMessageText(node, value, markdown = false) {
  node.replaceChildren();
  const content = String(value || "");
  if (!markdown || !content) {
    node.textContent = content;
    return;
  }

  const lines = content.replace(/\r\n?/g, "\n").split("\n");
  let list = null;
  const closeList = () => {
    if (list) {
      node.append(list);
      list = null;
    }
  };

  lines.forEach((line, index) => {
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bullet) {
      if (!list) list = document.createElement("ul");
      const item = document.createElement("li");
      appendInlineMarkdown(item, bullet[1]);
      list.append(item);
      return;
    }
    closeList();
    if (!line.trim()) return;
    const paragraph = document.createElement("span");
    paragraph.className = "message-paragraph";
    appendInlineMarkdown(paragraph, line);
    node.append(paragraph);
    if (index < lines.length - 1) node.append(document.createElement("br"));
  });
  closeList();
}

function appendInlineMarkdown(node, value) {
  const source = String(value || "");
  const pattern = /\*\*(.+?)\*\*/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(source))) {
    if (match.index > cursor) node.append(document.createTextNode(source.slice(cursor, match.index)));
    const strong = document.createElement("strong");
    strong.textContent = match[1];
    node.append(strong);
    cursor = match.index + match[0].length;
  }
  if (cursor < source.length) node.append(document.createTextNode(source.slice(cursor)));
}

function productCard(product) {
  const card = document.createElement("article");
  card.className = "product-card";
  const image = document.createElement("div");
  image.className = "product-image";
  if (product.image_url) {
    const img = document.createElement("img");
    img.src = product.image_url;
    img.alt = product.title;
    img.loading = "lazy";
    img.referrerPolicy = "no-referrer";
    img.addEventListener("error", () => { image.replaceChildren(Object.assign(document.createElement("span"), {textContent: "Image unavailable"})); });
    image.append(img);
  } else {
    image.append(Object.assign(document.createElement("span"), {textContent: "Image unavailable"}));
  }
  const body = document.createElement("div");
  body.className = "product-body";
  body.innerHTML = `<small></small><h3></h3><div class="price-row"><strong></strong><span></span></div>`;
  body.querySelector("small").textContent = product.brand || "Amazon Fashion";
  body.querySelector("h3").textContent = product.title;
  body.querySelector("strong").textContent = product.price == null ? "Price unavailable" : `$${Number(product.price).toFixed(2)}`;
  body.querySelector(".price-row span").textContent = product.average_rating ? `${Number(product.average_rating).toFixed(1)} / 5` : "Not rated";
  if (product.review_excerpt) {
    const quote = document.createElement("blockquote");
    quote.textContent = product.review_excerpt;
    body.append(quote);
  }
  const link = document.createElement("a");
  link.href = `https://www.amazon.com/dp/${encodeURIComponent(product.parent_asin)}`;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "View on Amazon";
  body.append(link);
  card.append(image, body);
  return card;
}

function renderMessages() {
  const conversation = el("conversation");
  conversation.querySelectorAll(".message").forEach((node) => node.remove());
  el("welcome").hidden = state.messages.length > 0;
  updateChatLayoutState();
  state.messages.forEach((message, index) => conversation.append(messageNode(message, index)));
  conversation.scrollTop = conversation.scrollHeight;
}

function renderLiveReply(reply) {
  const conversation = el("conversation");
  const assistantNodes = conversation.querySelectorAll(".message.assistant");
  const article = assistantNodes[assistantNodes.length - 1];
  if (!article) {
    renderMessages();
    return;
  }
  const text = article.querySelector(".message-text");
  if (!text) return;
  renderMessageText(text, reply.content || (state.status === "generating"
    ? "Writing the answer from retrieved evidence..."
    : "Retrieving catalog and review evidence..."), true);
  conversation.scrollTop = conversation.scrollHeight;
}

async function refresh() {
  const [data, prefs] = await Promise.all([
    api("/api/sessions"),
    api("/api/preferences"),
  ]);
  state.sessions = data.sessions;
  state.cursor = data.next_cursor;
  state.prefs = prefs;
  renderSessions();
  renderPrefs();
}

async function authenticate(event) {
  event.preventDefault();
  setError("auth-error", "");
  const raw = Object.fromEntries(new FormData(event.currentTarget));
  const data = state.signup
    ? {name: raw.name, username: raw.username, password: raw.password}
    : {username: raw.username, password: raw.password};
  setBusy(true);
  try {
    const result = await api(state.signup ? "/api/signup" : "/api/login", "POST", data);
    state.token = result.access_token;
    localStorage.setItem("thread_token", state.token);
    state.user = {username: result.username, name: result.name};
    const [_, info] = await Promise.all([refresh(), api("/info")]);
    el("catalog-tag").textContent = info.mongo_products_count ? `${Number(info.mongo_products_count).toLocaleString()} catalog products` : "Amazon Fashion";
    showApp();
    renderAccount();
  } catch (error) { setError("auth-error", error.message); }
  finally { setBusy(false); }
}

async function send(event, text = el("question").value) {
  event?.preventDefault();
  if (!text.trim() || state.busy) return;
  setBusy(true);
  setError("app-error", "");
  state.status = "retrieving";
  el("question").value = "";
  const controller = new AbortController();
  state.abort = controller;
  const reply = {role: "assistant", content: ""};
  state.messages.push({role: "user", content: text}, reply);
  renderMessages();
  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json", Authorization: `Bearer ${state.token}`},
      body: JSON.stringify({question: text, session_id: state.active}),
      signal: controller.signal,
    });
    await readEvents(response, (eventName, data) => {
      if (eventName === "session") state.active = data.session_id;
      if (eventName === "status") {
        state.status = data.state === "generating" ? "generating" : "retrieving";
        renderLiveReply(reply);
        return;
      }
      if (eventName === "token") {
        reply.content += data.text;
        renderLiveReply(reply);
        return;
      }
      if (eventName === "products") reply.products = data.products;
      if (eventName === "citations") reply.citations = data.citations;
      if (eventName === "done") {
        if (typeof data.answer === "string") reply.content = data.answer;
        reply.suggested_relaxations = data.suggested_relaxations || [];
      }
      renderMessages();
    });
    await refresh();
  } catch (error) {
    reply.content = "";
    reply.products = [];
    reply.citations = [];
    renderMessages();
    setError("app-error", error.name === "AbortError" ? "Response stopped. You can retry your question." : error.message);
    el("question").value = text;
  } finally {
    state.status = "";
    state.abort = null;
    setBusy(false);
  }
}

async function openSession(id) {
  if (state.busy) return;
  setError("app-error", "");
  try {
    const data = await api(`/api/sessions/${id}`);
    state.active = id;
    state.messages = data.messages || [];
    renderSessions();
    renderMessages();
  } catch (error) { setError("app-error", error.message); }
}

async function deleteSession(id) {
  if (!confirm("Delete this saved conversation?")) return;
  try {
    await api(`/api/sessions/${id}`, "DELETE");
    if (state.active === id) {
      state.active = null;
      state.messages = [];
      renderMessages();
    }
    await refresh();
  } catch (error) { setError("app-error", error.message); }
}

function bindEvents() {
  el("auth-form").addEventListener("submit", authenticate);
  el("toggle-auth").addEventListener("click", () => {
    state.signup = !state.signup;
    setError("auth-error", "");
    renderAuth();
  });
  el("new-chat").addEventListener("click", () => {
    state.active = null;
    state.messages = [];
    setError("app-error", "");
    renderSessions();
    renderMessages();
  });
  el("toggle-prefs").addEventListener("click", () => {
    const panel = el("preferences");
    panel.hidden = !panel.hidden;
    document.body.classList.toggle("prefs-open", !panel.hidden);
    el("toggle-prefs").setAttribute("aria-expanded", String(!panel.hidden));
  });
  el("theme-toggle").addEventListener("click", () => {
    const next = document.body.classList.contains("light-theme") ? "dark" : "light";
    localStorage.setItem("fitfinder_theme", next);
    applyTheme(next);
  });
  el("close-prefs").addEventListener("click", () => {
    el("preferences").hidden = true;
    document.body.classList.remove("prefs-open");
    el("toggle-prefs").setAttribute("aria-expanded", "false");
  });
  el("sign-out").addEventListener("click", () => {
    localStorage.removeItem("thread_token");
    Object.assign(state, {token: "", user: null, active: null, messages: [], prefs: {}, sessions: []});
    showApp();
    renderAuth();
  });
  el("composer").addEventListener("submit", send);
  el("stop").addEventListener("click", () => state.abort?.abort());
  el("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send(event);
    }
  });
  el("question").addEventListener("input", resizeQuestion);
  resizeQuestion();
  el("load-more").addEventListener("click", async () => {
    try {
      const data = await api(`/api/sessions?cursor=${encodeURIComponent(state.cursor)}`);
      state.sessions.push(...data.sessions);
      state.cursor = data.next_cursor;
      renderSessions();
    } catch (error) { setError("app-error", error.message); }
  });
  el("preference-field").addEventListener("change", () => {
    el("preference-value-label").firstChild.nodeValue = el("preference-field").value === "budget" ? "Maximum budget ($)" : "Value";
  });
  el("preference-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const field = data.get("field");
    let value = data.get("value");
    if (field === "budget") value = {max: Number(value)};
    try {
      state.prefs = await api("/api/preferences", "PATCH", {field, action: ["favorite_brands", "color_preference"].includes(field) ? "add" : "set", value});
      form.reset();
      renderPrefs();
    } catch (error) { setError("app-error", error.message); }
  });
  prompts.forEach((text, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<span class="suggestion-number">0${index + 1}</span><span></span>`;
    button.querySelector("span:nth-child(2)").textContent = text;
    button.addEventListener("click", () => send(undefined, text));
    el("suggestions").append(button);
  });
}

async function boot() {
  bindEvents();
  applyTheme(localStorage.getItem("fitfinder_theme") || "dark");
  renderAuth();
  renderPrefs();
  if (!state.token) {
    showApp();
    return;
  }
  try {
    state.user = await api("/api/me");
    await refresh();
    const info = await api("/info");
    el("catalog-tag").textContent = info.mongo_products_count ? `${Number(info.mongo_products_count).toLocaleString()} catalog products` : "Amazon Fashion";
    renderAccount();
  } catch {
    localStorage.removeItem("thread_token");
    state.token = "";
    state.user = null;
  }
  showApp();
}

boot();
