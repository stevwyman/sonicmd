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
  startRename(id, current);
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

function formatClock(s) {
  const n = Math.max(0, Math.round(Number(s) || 0));
  const m = Math.floor(n / 60);
  const r = n % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

const ROUTES = {
  both: "Headphones: Output StenoPod Headphones, Input MacBook mic. Wav is stereo — left=you, right=them.",
  speakers: "Speakers: Output StenoPod Speakers, Input MacBook mic. Wav is stereo — left=you, right=them. Echo from the speakers is cancelled on the left.",
  system: "StenoPod Speakers · system audio only. Set Sound → Output to StenoPod Speakers.",
};

let lastState = null;
let renamingId = null;
let renameDraft = "";
let pendingJob = null;
let pollTimer = null;
let recordStartedAt = null;
let recordTick = null;

function formatDay(iso) {
  try {
    return new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

function activeJob(state, serverBusy) {
  if (serverBusy && state.job && state.job.label) return state.job;
  return pendingJob;
}

function ensureRecordTick(listening) {
  if (listening) {
    if (recordStartedAt == null) recordStartedAt = Date.now();
    if (!recordTick) {
      recordTick = setInterval(() => {
        if (lastState) render(lastState);
      }, 250);
    }
    return;
  }
  recordStartedAt = null;
  if (recordTick) {
    clearInterval(recordTick);
    recordTick = null;
  }
}

function recordingClock(state, listening) {
  const reported = Number(listening ? state.sidecar && state.sidecar.recorded_s : state.audio_duration_s) || 0;
  if (listening && recordStartedAt != null) {
    return Math.max(reported, (Date.now() - recordStartedAt) / 1000);
  }
  return reported;
}

function ensurePoll(busy) {
  if (busy && !pollTimer) {
    pollTimer = setInterval(() => {
      fetchState().catch(() => {});
    }, 1500);
  } else if (!busy && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function renderRecordings(rows, busy) {
  const ul = $("recordings");
  const empty = $("rec-empty");
  if (!ul || !empty) return;
  const items = rows || [];
  const count = $("rec-count");
  if (count) count.textContent = items.length ? `(${items.length})` : "";
  empty.hidden = items.length > 0;
  const key = `${busy ? 1 : 0}|${items
    .map((row) =>
      [
        row.path,
        row.title,
        row.current ? 1 : 0,
        row.transcribed ? 1 : 0,
        row.too_short ? 1 : 0,
        row.segments,
        Math.round(Number(row.duration_s) || 0),
      ].join("\t"),
    )
    .join("\n")}`;
  if (key === ul.dataset.key) return;
  ul.dataset.key = key;
  ul.innerHTML = "";
  items.forEach((row, index) => {
    const li = document.createElement("li");
    if (row.current) li.classList.add("current");
    let status = "Not transcribed";
    if (row.too_short) status = "Too short to transcribe";
    else if (row.transcribed) status = `Transcribed · ${row.segments} takes`;
    else if (row.segments > 0) status = `${row.segments} takes · not finished`;
    const when = formatDay(row.recorded_at);
    li.innerHTML = `<div class="rec-main"><b></b><span></span></div><div class="rec-actions"></div>`;
    li.querySelector("b").textContent = row.title || row.name;
    li.querySelector("span").textContent = `${when} · ${formatClock(row.duration_s)} · ${status}`;
    const actions = li.querySelector(".rec-actions");
    if (!row.current) {
      const open = document.createElement("button");
      open.type = "button";
      open.textContent = "Open";
      open.dataset.action = "open";
      open.dataset.index = String(index);
      open.disabled = busy;
      actions.appendChild(open);
    }
    if (!row.too_short) {
      const run = document.createElement("button");
      run.type = "button";
      run.className = "primary";
      run.dataset.action = row.transcribed ? "again" : "transcribe";
      run.dataset.index = String(index);
      run.textContent = row.transcribed ? "Transcribe again" : row.segments > 0 ? "Continue" : "Transcribe";
      run.disabled = busy;
      actions.appendChild(run);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "Delete";
    remove.dataset.action = "delete";
    remove.dataset.index = String(index);
    remove.disabled = busy;
    actions.appendChild(remove);
    ul.appendChild(li);
  });
}

async function commitRename(id, name) {
  const trimmed = name.trim();
  renamingId = null;
  renameDraft = "";
  if (!trimmed) {
    if (lastState) render(lastState);
    return;
  }
  const r = await fetch("/api/speakers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, name: trimmed }),
  });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(apiError(body, r.statusText));
    if (lastState) render(lastState);
    return;
  }
  render(body);
}

function startRename(id, current) {
  renamingId = id;
  renameDraft = current || "";
  if (lastState) render(lastState);
  const input = document.querySelector("#roster .rename input");
  if (input) {
    input.focus();
    input.select();
  }
}

function speakerChip(sp, editing) {
  const li = document.createElement("li");
  if (editing) {
    li.innerHTML = `<form class="rename"><input maxlength="40" aria-label="Name for speaker ${escapeHtml(sp.id)}" value="${escapeHtml(renameDraft)}"><button type="submit">Save</button></form>`;
    const form = li.querySelector("form");
    const input = form.querySelector("input");
    input.addEventListener("input", () => {
      renameDraft = input.value;
    });
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      commitRename(sp.id, input.value);
    });
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        ev.preventDefault();
        renamingId = null;
        renameDraft = "";
        if (lastState) render(lastState);
      }
    });
    return li;
  }
  const pct = Math.round((sp.share || 0) * 100);
  li.innerHTML = `<button type="button" title="Rename ${escapeHtml(sp.name)}" style="--spk:${speakerColor(sp.id)}"><span class="bar"><i style="width:${pct}%"></i></span>${escapeHtml(sp.name)} · ${escapeHtml(sp.duration)} · ${sp.turns} turns</button>`;
  li.querySelector("button").addEventListener("click", () => startRename(sp.id, sp.name));
  return li;
}

function renderJob(job) {
  const box = $("job");
  const bar = $("job-bar");
  const fill = $("job-fill");
  if (!job || !job.label) {
    box.hidden = true;
    bar.classList.remove("indeterminate");
    fill.style.width = "0";
    return;
  }
  box.hidden = false;
  $("job-label").textContent = job.label;
  const pct = job.progress == null ? null : Math.round(Math.max(0, Math.min(1, Number(job.progress))) * 100);
  if (pct == null) {
    bar.classList.add("indeterminate");
    bar.removeAttribute("aria-valuenow");
    fill.style.width = "";
  } else {
    bar.classList.remove("indeterminate");
    bar.setAttribute("aria-valuenow", String(pct));
    fill.style.width = `${pct}%`;
  }
}

function render(state) {
  if (lastState && state.id !== lastState.id) {
    renamingId = null;
    renameDraft = "";
  }
  lastState = state;
  const listening = Boolean(state.sidecar && state.sidecar.listening);
  ensureRecordTick(listening);
  const serverBusy = Boolean(state.transcribing || state.summarizing || (state.job && state.job.kind));
  const job = activeJob(state, serverBusy);
  const busy = serverBusy || Boolean(pendingJob);
  const duration = recordingClock(state, listening);
  const hasRecording = !listening && Number(duration) > 1;
  const transcribed =
    hasRecording && Number(state.audio_cursor_s) >= Number(duration) - 1 && (state.segment_count || 0) > 0;
  const sel = $("capture-source");
  if (document.activeElement !== sel && state.capture_source && sel.value !== state.capture_source && !listening) {
    sel.value = state.capture_source;
  }

  if (document.activeElement !== $("title")) {
    $("title").value = state.title || "";
  }
  applySavedBounds(state);
  $("counts").textContent = `${state.segment_count || 0} takes · ${(state.speakers || []).length} speakers`;

  const route = ROUTES[sel.value] || ROUTES.both;
  const showJob = Boolean(busy && job && job.label);
  $("device").hidden = showJob;
  if (showJob) {
    setLamp("lamp-listen", "busy");
  } else if (listening) {
    $("device").textContent = `Recording ${formatClock(duration)}. Transcription starts when you stop. ${route}`;
    setLamp("lamp-listen", "on");
  } else if (state.transcribe_error) {
    const lines = String(state.transcribe_error).split("\n").map((line) => line.trim()).filter(Boolean);
    $("device").textContent = lines[lines.length - 1] || "Transcription failed.";
    setLamp("lamp-listen", "");
  } else if (hasRecording && !transcribed) {
    $("device").textContent = `Recording saved · ${formatClock(duration)}. Press Transcribe.`;
    setLamp("lamp-listen", "");
  } else if (transcribed) {
    $("device").textContent = `Transcribed · ${formatClock(duration)}.`;
    setLamp("lamp-listen", "");
  } else {
    $("device").textContent = route;
    setLamp("lamp-listen", "");
  }
  renderJob(busy ? job : null);
  renderRecordings(state.recordings, busy || listening);
  ensurePoll(busy);
  setLamp("lamp-sum", state.summarizing ? "busy" : state.notes ? "on" : "");

  sel.disabled = listening || busy || hasRecording;
  $("spk-min").disabled = busy;
  $("spk-max").disabled = busy;
  $("btn-new").disabled = listening || busy;
  $("btn-transcribe").disabled = listening || busy || !hasRecording || transcribed;
  syncCaptureButton(listening, Boolean(state.capture && state.capture.running), busy, hasRecording);

  const roster = $("roster");
  const renameInput = document.querySelector("#roster .rename input");
  const preserveRename = Boolean(renamingId && renameInput && document.activeElement === renameInput);
  if (!preserveRename) {
    roster.innerHTML = "";
    for (const sp of state.speakers || []) {
      roster.appendChild(speakerChip(sp, renamingId === sp.id));
    }
  }
  const hint = $("rename-hint");
  if (hint) hint.hidden = !(state.speakers || []).length;

  const list = $("transcript");
  list.innerHTML = "";
  for (const block of state.blocks || []) {
    const li = document.createElement("li");
    li.className = "block";
    li.style.setProperty("--spk", speakerColor(block.speaker));
    const langs = (block.langs || []).filter((code) => code && code !== "und");
    const langLabel = langs.length ? langs.join("/") : "—";
    li.innerHTML = `<div class="who"><button type="button" title="Rename ${escapeHtml(block.name)}">${escapeHtml(block.name)}</button><span>${escapeHtml(formatWhen(block.start))} · ${escapeHtml(block.duration)} · ${escapeHtml(langLabel)}</span></div><div class="body">${escapeHtml(block.text)}</div>`;
    li.querySelector("button").addEventListener("click", () => renameSpeaker(block.speaker, block.name));
    list.appendChild(li);
  }
  $("tx-empty").hidden = (state.blocks || []).length > 0;
  if (!$("tx-empty").hidden) {
    if (state.transcribing || (pendingJob && pendingJob.kind === "transcribe")) {
      $("tx-empty").innerHTML = "Speakers first, on the whole recording. Whisper writes each turn as it finishes.";
    } else if (listening) {
      $("tx-empty").innerHTML = `Recording ${formatClock(duration)}. Transcription starts when you stop.`;
    } else if (hasRecording) {
      $("tx-empty").innerHTML = "Recording saved. Transcription should already be running. Press <strong>Transcribe</strong> if it is not.";
    } else {
      $("tx-empty").innerHTML = "Choose StenoPod Speakers or Headphones, then press <strong>Start recording</strong>. Transcription starts when you stop.";
    }
  }
  $("notes").innerHTML = renderMarkdown(state.notes || "");
  $("nt-empty").hidden = Boolean(state.notes) || Boolean(state.summary_error) || Boolean(state.transcribe_error);
  const err = $("nt-err");
  if (state.transcribe_error) {
    err.hidden = false;
    err.textContent = state.transcribe_error;
  } else if (state.summary_error) {
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

function syncCaptureButton(listening, running, busy, hasRecording) {
  const btn = $("btn-capture");
  if (!btn) return;
  if (listening) {
    btn.disabled = Boolean(busy);
    btn.classList.add("stop");
    btn.textContent = "Stop recording";
    return;
  }
  if (running && !busy) {
    btn.disabled = true;
    btn.classList.remove("stop");
    btn.textContent = "Starting…";
    return;
  }
  if (hasRecording) {
    btn.disabled = true;
    btn.classList.remove("stop");
    btn.textContent = "Start recording";
    return;
  }
  btn.disabled = Boolean(busy);
  btn.classList.remove("stop");
  btn.textContent = "Start recording";
}

let boundsSession = null;

function applySavedBounds(state) {
  if (!state || state.id === boundsSession) return;
  boundsSession = state.id;
  if (!state.speaker_bounds_saved) return;
  const minBox = $("spk-min");
  const maxBox = $("spk-max");
  if (document.activeElement === minBox || document.activeElement === maxBox) return;
  minBox.value = state.min_speakers ? String(state.min_speakers) : "";
  maxBox.value = state.max_speakers ? String(state.max_speakers) : "";
}

function boundsPhrase() {
  const body = speakerBounds();
  const min = body.min_speakers;
  const max = body.max_speakers;
  if (min && max && min === max) return `exactly ${min} speaker${min === 1 ? "" : "s"}`;
  if (min || max) return `speakers from ${min || "auto"} to ${max || "auto"}`;
  return "automatic speaker count";
}

function speakerBounds() {
  const read = (id) => {
    const raw = $(id).value.trim();
    if (!raw) return null;
    const n = Number(raw);
    if (!Number.isInteger(n) || n < 1 || n > 12) {
      throw new Error("Speaker counts must be whole numbers from 1 to 12, or left blank.");
    }
    return n;
  };
  const min = read("spk-min");
  const max = read("spk-max");
  if (min !== null && max !== null && min > max) {
    throw new Error("Minimum speakers cannot be higher than the maximum.");
  }
  const body = {};
  if (min !== null) body.min_speakers = min;
  if (max !== null) body.max_speakers = max;
  try {
    localStorage.setItem("stenopod.speakers", JSON.stringify({ min, max }));
  } catch (err) {
    /* storage can be blocked; the request still uses the fields */
  }
  return body;
}

function restoreSpeakerBounds() {
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem("stenopod.speakers") || "null");
  } catch (err) {
    saved = null;
  }
  if (!saved) return;
  if (saved.min) $("spk-min").value = String(saved.min);
  if (saved.max) $("spk-max").value = String(saved.max);
}

function apiError(body, fallback) {
  const d = body && body.detail;
  if (typeof d === "string") return d;
  if (d) return JSON.stringify(d);
  return fallback;
}

$("btn-capture").addEventListener("click", async () => {
  const btn = $("btn-capture");
  const listening = btn.classList.contains("stop");
  btn.disabled = true;
  try {
    if (listening) {
      btn.textContent = "Stopping…";
      pendingJob = { kind: "transcribe", label: "Stopping recording. Transcription starts next.", progress: null };
      if (lastState) render(lastState);
      const r = await fetch("/api/capture/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(speakerBounds()),
      });
      const body = await r.json().catch(() => ({}));
      pendingJob = null;
      if (!r.ok) throw new Error(apiError(body, r.statusText));
      if (body.state) render(body.state);
      else fetchState();
    } else {
      btn.textContent = "Starting…";
      const startBody = JSON.stringify({
        source: $("capture-source").value,
      });
      let r = await fetch("/api/capture/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: startBody,
      });
      let body = await r.json().catch(() => ({}));
      if (r.status === 409 && /already running/i.test(apiError(body, ""))) {
        await fetch("/api/capture/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(speakerBounds()),
        });
        r = await fetch("/api/capture/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: startBody,
        });
        body = await r.json().catch(() => ({}));
      }
      if (!r.ok) throw new Error(apiError(body, r.statusText));
      if (body.state) render(body.state);
    }
  } catch (err) {
    pendingJob = null;
    alert(err.message || String(err));
    fetchState().catch(() => {});
  }
});

async function deleteRecording(row) {
  const title = row.title || "this recording";
  if (!confirm(`Delete “${title}”? This removes the audio file and its transcript.`)) return;
  try {
    const r = await fetch("/api/recordings/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: row.path }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(apiError(body, r.statusText));
    render(body);
  } catch (err) {
    alert(err.message || String(err));
    fetchState();
  }
}

async function postRecording(action, row) {
  if (action === "again") {
    const title = row.title || "this recording";
    let phrase = "the speaker counts in the header";
    try {
      phrase = boundsPhrase();
    } catch (err) {
      alert(err.message || String(err));
      return;
    }
    if (!confirm(`Transcribe “${title}” again? This redoes the speakers (${phrase}), the transcript, and the notes.`)) return;
  }
  if (action !== "open") {
    pendingJob = { kind: "transcribe", label: "Starting transcription", progress: null };
    if (lastState) render(lastState);
  }
  const path = action === "open" ? "/api/recordings/open" : "/api/recordings/transcribe";
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: row.path,
        restart: action === "again",
        ...(action === "open" ? {} : speakerBounds()),
      }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(apiError(body, r.statusText));
    pendingJob = null;
    render(body);
  } catch (err) {
    pendingJob = null;
    alert(err.message || String(err));
    fetchState();
  }
}

$("rec-toggle").addEventListener("click", () => {
  const panel = $("rec-panel");
  const open = panel.hidden;
  panel.hidden = !open;
  $("rec-toggle").setAttribute("aria-expanded", open ? "true" : "false");
  $("rec-toggle").querySelector(".rec-chevron").textContent = open ? "▾" : "▸";
});

$("recordings").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button");
  if (!btn || btn.disabled) return;
  const row = (lastState && lastState.recordings ? lastState.recordings : [])[Number(btn.dataset.index)];
  if (!row) return;
  if (btn.dataset.action === "delete") {
    deleteRecording(row);
    return;
  }
  postRecording(btn.dataset.action, row);
});

$("btn-transcribe").addEventListener("click", async () => {
  pendingJob = { kind: "transcribe", label: "Starting transcription", progress: null };
  if (lastState) render(lastState);
  try {
    const r = await fetch("/api/transcribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(speakerBounds()),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(apiError(body, r.statusText));
    pendingJob = null;
    render(body);
  } catch (err) {
    pendingJob = null;
    alert(err.message || String(err));
    fetchState();
  }
});

const downloadBtn = $("btn-download");
const downloadPanel = $("download-panel");
if (downloadBtn && downloadPanel) {
  downloadBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    const open = downloadPanel.hidden;
    downloadPanel.hidden = !open;
    downloadBtn.setAttribute("aria-expanded", open ? "true" : "false");
  });
  document.addEventListener("click", (ev) => {
    if (ev.target.closest("#download-menu")) return;
    downloadPanel.hidden = true;
    downloadBtn.setAttribute("aria-expanded", "false");
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") return;
    downloadPanel.hidden = true;
    downloadBtn.setAttribute("aria-expanded", "false");
  });
}

$("btn-new").addEventListener("click", async () => {
  if (!confirm("Start a new meeting? This recording stays on disk.")) return;
  const r = await fetch("/api/session", { method: "POST" });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(apiError(body, r.statusText));
    return;
  }
  render(body);
});

$("capture-source").addEventListener("change", () => {
  if (lastState) render(lastState);
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

restoreSpeakerBounds();
fetchState();
fetchHealth();
setInterval(fetchHealth, 8000);
connectWs();
