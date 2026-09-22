const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderMarkdown(src) {
  if (!src) return "";
  const escaped = escapeHtml(src);
  const lines = escaped.split("\n");
  const out = [];
  let list = false;
  const closeList = () => {
    if (list) {
      out.push("</ul>");
      list = false;
    }
  };
  for (const line of lines) {
    const heading = line.match(/^(#{1,3})\s+(.*)$/);
    const bullet = line.match(/^[-*]\s+(.*)$/);
    const check = line.match(/^[-*]\s+\[( |x|X)\]\s+(.*)$/);
    const inline = (t) =>
      t
        .replaceAll(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replaceAll(/`(.+?)`/g, "<code>$1</code>");
    if (heading) {
      closeList();
      const n = heading[1].length;
      out.push(`<h${n}>${inline(heading[2])}</h${n}>`);
    } else if (check) {
      if (!list) {
        out.push("<ul>");
        list = true;
      }
      const mark = check[1].toLowerCase() === "x" ? "☑" : "☐";
      out.push(`<li>${mark} ${inline(check[2])}</li>`);
    } else if (bullet) {
      if (!list) {
        out.push("<ul>");
        list = true;
      }
      out.push(`<li>${inline(bullet[1])}</li>`);
    } else if (!line.trim()) {
      closeList();
    } else {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return out.join("");
}

function formatWhen(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return iso;
  }
}

const SPK_COLORS = {
  A: "#c9190b",
  B: "#0066cc",
  C: "#3e8635",
  D: "#f0ab00",
  E: "#6753ac",
  F: "#009596",
  G: "#a30000",
  H: "#002f5d",
};

function speakerColor(id) {
  return SPK_COLORS[id] || "#6a6e73";
}

async function renameSpeaker(id, current) {
  const name = prompt(`Name for Speaker ${id}`, current || `Speaker ${id}`);
  if (!name) return;
  const r = await fetch("/api/speakers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, name: name.trim() }),
  });
  render(await r.json());
}

function setLamp(id, mode) {
  const el = $(id);
  el.classList.remove("on", "busy", "bad");
  const state = el.querySelector(".lamp-state");
  if (mode === "on") {
    el.classList.add("on");
    if (state) state.textContent = "ready";
  } else if (mode === "busy") {
    el.classList.add("busy");
    if (state) state.textContent = "busy";
  } else if (mode === "bad") {
    el.classList.add("bad");
    if (state) state.textContent = "error";
  } else if (state) {
    state.textContent = "idle";
  }
}

function render(state) {
  $("title").value = state.title || "";
  const speakers = state.speakers || [];
  $("counts").textContent = `${state.segment_count || 0} takes · ${speakers.length} speakers · ${state.pending_chars || 0} chars pending`;
  const sc = state.sidecar || {};
  $("device").textContent = sc.listening
    ? `Listening · ${sc.device || "audio"} · ${sc.model || "whisper"}`
    : "Sidecar idle — start with make sidecar";
  setLamp("lamp-listen", sc.listening ? "on" : "");
  setLamp("lamp-sum", state.summarizing ? "busy" : state.notes ? "on" : "");

  const roster = $("roster");
  roster.innerHTML = "";
  for (const sp of speakers) {
    const li = document.createElement("li");
    const pct = Math.round((sp.share || 0) * 100);
    li.innerHTML = `<button type="button" style="--spk:${speakerColor(sp.id)}"><span class="bar"><i style="width:${pct}%"></i></span>${escapeHtml(sp.name)} · ${escapeHtml(sp.duration)} · ${sp.turns} turns</button>`;
    li.querySelector("button").addEventListener("click", () => renameSpeaker(sp.id, sp.name));
    roster.appendChild(li);
  }

  const list = $("transcript");
  list.innerHTML = "";
  for (const block of state.blocks || []) {
    const li = document.createElement("li");
    li.className = "block";
    li.style.setProperty("--spk", speakerColor(block.speaker));
    const langs = (block.langs || []).join("/");
    li.innerHTML = `<div class="who"><button type="button">${escapeHtml(block.name)}</button><span>${escapeHtml(formatWhen(block.start))} · ${escapeHtml(block.duration)} · ${escapeHtml(langs)}</span></div><div class="body">${escapeHtml(block.text)}</div>`;
    li.querySelector("button").addEventListener("click", () => renameSpeaker(block.speaker, block.name));
    list.appendChild(li);
  }
  $("tx-empty").hidden = (state.blocks || []).length > 0;
  $("notes").innerHTML = renderMarkdown(state.notes || "");
  $("nt-empty").hidden = Boolean(state.notes) || Boolean(state.summary_error);
  const err = $("nt-err");
  if (state.summary_error) {
    err.hidden = false;
    err.textContent = state.summary_error;
  } else {
    err.hidden = true;
  }
}

async function fetchState() {
  const r = await fetch("/api/state");
  render(await r.json());
}

async function fetchHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    setLamp("lamp-llm", h.llm ? "on" : "bad");
  } catch {
    setLamp("lamp-llm", "bad");
  }
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.state) render(msg.state);
  };
  ws.onclose = () => setTimeout(connectWs, 1500);
}

$("btn-summarize").addEventListener("click", async () => {
  $("btn-summarize").disabled = true;
  try {
    const r = await fetch("/api/summarize", { method: "POST" });
    render(await r.json());
  } finally {
    $("btn-summarize").disabled = false;
  }
});

$("btn-new").addEventListener("click", async () => {
  if (!confirm("Start a new session? Current notes stay on disk.")) return;
  const r = await fetch("/api/session", { method: "POST" });
  render(await r.json());
});

let titleTimer;
$("title").addEventListener("input", () => {
  clearTimeout(titleTimer);
  titleTimer = setTimeout(() => {
    fetch("/api/title", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: $("title").value }),
    });
  }, 400);
});

fetchState();
fetchHealth();
setInterval(fetchHealth, 8000);
connectWs();
