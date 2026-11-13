import { Viewer, drawLandmarks } from "./viewer.js";

const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
};
const native = () => window.pywebview && window.pywebview.api;
let toastTimer;
function toast(msg, err = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = err ? "err" : "";
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), err ? 8000 : 3500);
}

const S = { analysis: null, character: null, viewer: null, anim: null, playing: false, t: 0, layout: "side" };

// ------------------------------------------------------------------ navigation
function showStep(name) {
  const keysMode = name === "keys";
  const sec = keysMode ? "viewer" : name;
  document.querySelectorAll(".step").forEach((s) => s.classList.toggle("active", s.id === "step-" + sec));
  document.querySelectorAll("#steps button").forEach((b) => b.classList.toggle("active", b.dataset.step === name));
  document.body.classList.toggle("keys-mode", keysMode);
  $("keyPanel").hidden = !keysMode;
  if (keysMode) refreshKeyPanel();
  if (sec === "viewer") { S.viewer?.resize(); setTimeout(() => { S.viewer?.resize(); if (S.anim) { sizeLm(); drawTrack(); } }, 30); }
  if (name === "analysis") drawCamChart();
}
document.querySelectorAll("#steps button").forEach((b) => b.addEventListener("click", () => showStep(b.dataset.step)));

// ------------------------------------------------------------------ step 1: files
async function choose(kind) {
  try {
    if (native()) {
      const path = await native().pick_file(kind);
      if (!path) return;
      const res = await api("/api/open", kind === "video" ? { video: path } : { vrm: path });
      onOpened(res);
    } else {
      $(kind === "video" ? "fileVideo" : "fileVrm").click();
    }
  } catch (e) { toast(e.message, true); }
}
async function upload(kind, file) {
  if (!file) return;
  setJob({ message: `Loading ${file.name}…`, progress: 0.1, done: false });
  try {
    const r = await fetch(`/api/upload?kind=${kind}&name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    onOpened(j);
  } catch (e) { toast(e.message, true); }
  setJob(null);
}
$("pickVideo").onclick = () => choose("video");
$("pickVrm").onclick = () => choose("vrm");
$("fileVideo").onchange = (e) => upload("video", e.target.files[0]);
$("fileVrm").onchange = (e) => upload("vrm", e.target.files[0]);

function onOpened(res) {
  if (res.video) {
    $("videoPath").textContent = res.video.path;
    const v = res.video;
    $("videoInfo").textContent = `${v.width}×${v.height} · ${v.fps.toFixed(2)} fps · ${v.frames} frames (${(v.frames / v.fps / 60).toFixed(1)} min)` + (v.cached_analysis ? " · saved analysis found" : "");
    S.analysis = null;
    S.anim = null;
    $("steps").children[2].disabled = true;
    if (v.cached_analysis) loadAnalysis();
    else $("steps").children[1].disabled = true;
  }
  if (res.vrm) {
    $("vrmPath").textContent = res.vrm.path;
    const kind = res.vrm.version === "fbx" ? "FBX" : `VRM ${res.vrm.version === "1" ? "1.0" : "0.x"}`;
    $("vrmInfo").textContent = `${res.vrm.title || "Avatar"} · ${kind} · ${res.vrm.bones} humanoid bones`;
    S.vrmLoaded = false;
    S.anim = null;
    $("steps").children[2].disabled = true;
  }
  refreshState();
}
async function refreshState() {
  const st = await api("/api/state");
  $("analyzeBtn").disabled = !st.video;
  $("analyzeBtn").textContent = st.analysis ? "Open analysis" : "Analyze video";
  S.state = st;
  if (S.analysis) renderCharacters();
  return st;
}
$("analyzeBtn").onclick = async () => {
  const force = $("forceAnalysis").checked;
  if (S.state?.analysis && !force) { await loadAnalysis(); showStep("analysis"); return; }
  try {
    await api("/api/analyze", { force });
    const st = await watchJob();
    if (!st.error) { await loadAnalysis(); showStep("analysis"); }
  } catch (e) { toast(e.message, true); }
};

// ------------------------------------------------------------------ jobs
function setJob(st) {
  const bar = $("jobbar");
  if (!st || st.done) { bar.hidden = true; return; }
  bar.hidden = false;
  $("jobmsg").textContent = st.message || "";
  $("jobprog").style.width = `${Math.round((st.progress || 0) * 100)}%`;
}
$("jobcancel").onclick = () => api("/api/cancel", {});
async function watchJob() {
  for (;;) {
    const st = await api("/api/status");
    setJob(st);
    if (st.done) {
      if (st.error) toast(st.error === "Cancelled" ? "Cancelled." : `Failed: ${st.error}`, st.error !== "Cancelled");
      return st;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
}

// ------------------------------------------------------------------ step 2: analysis
async function loadAnalysis() {
  S.analysis = await api("/api/analysis");
  $("steps").children[1].disabled = false;
  await refreshState();
  renderCharacters();
}
function fmtTime(s) { const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }
function renderCharacters() {
  const A = S.analysis;
  if (!A) return;
  const nShots = A.shots.length;
  $("analysisSummary").textContent = `${A.characters.length} character${A.characters.length === 1 ? "" : "s"} · ${nShots} camera shot${nShots === 1 ? "" : "s"} · ${A.frames} frames analysed`;
  const grid = $("charGrid");
  grid.innerHTML = "";
  const animated = new Set(S.state?.animated || []);
  if (!A.characters.length) grid.innerHTML = `<div class="card muted">No people were found in this video.</div>`;
  for (const c of A.characters) {
    const el = document.createElement("div");
    el.className = "char";
    const rh = c.relative_height != null ? `${Math.round(c.relative_height * 100)}% of tallest` : "n/a";
    const hc = c.height_vs_camera != null ? `${c.height_vs_camera.toFixed(2)}× camera height` : "n/a";
    el.innerHTML = `
      <div class="thumb" style="background-image:url('/api/thumb/${c.id}.jpg?${Date.now()}')"></div>
      <div class="body">
        <input class="name" value="${c.name.replace(/"/g, "&quot;")}" title="Rename">
        ${animated.has(c.id) ? '<span class="tag">● Animation ready</span>' : ""}
        <dl class="stats">
          <dt>Screen time</dt><dd>${c.screen_time_s.toFixed(1)} s</dd>
          <dt>Appearances</dt><dd>${c.appearances} (in ${c.shots.length} shot${c.shots.length === 1 ? "" : "s"})</dd>
          <dt>First / last seen</dt><dd>${fmtTime(c.first_frame / A.fps)} – ${fmtTime(c.last_frame / A.fps)}</dd>
          <dt>Relative height</dt><dd>${rh}</dd>
          <dt>Height vs scenery</dt><dd title="Character height divided by the camera's height above the floor (horizon method), averaged over shots">${hc}</dd>
        </dl>
        <button class="${animated.has(c.id) ? "" : "primary"}">${animated.has(c.id) ? "Open animation" : "Motion capture this character"}</button>
        ${animated.has(c.id) ? '<button class="ghost small redo" title="Re-solve with the current smoothing / foot-lock settings (re-uses the captured landmarks: takes seconds)">Re-solve with current settings</button>' : ""}
      </div>`;
    el.querySelector("input").onchange = (e) => api("/api/rename", { character: c.id, name: e.target.value }).then(() => (c.name = e.target.value));
    el.querySelector("button").onclick = () => (animated.has(c.id) ? openAnimation(c.id) : startCapture(c.id));
    el.querySelector(".redo")?.addEventListener("click", () => startCapture(c.id));
    grid.appendChild(el);
  }
  drawCamChart();
}
$("smoothing").oninput = (e) => ($("smoothingVal").textContent = Number(e.target.value).toFixed(2));

const SERIES = [
  { i: 0, name: "Yaw", color: "#4fd1c5" }, { i: 1, name: "Pitch", color: "#f6ad55" },
  { i: 2, name: "Roll", color: "#b794f4" }, { i: 3, name: "Vertical FOV", color: "#63b3ed" },
];
function drawCamChart(hoverX) {
  const A = S.analysis;
  const cv = $("camChart");
  if (!A || !cv.clientWidth) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr;
  cv.height = 220 * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cv.clientWidth, H = 220, pad = 34;
  const rows = A.camera, n = rows.length;
  let lo = Infinity, hi = -Infinity;
  for (let f = 0; f < n; f += Math.max(1, Math.floor(n / 2000))) for (const s of SERIES) { lo = Math.min(lo, rows[f][s.i]); hi = Math.max(hi, rows[f][s.i]); }
  lo = Math.floor(lo / 10) * 10; hi = Math.ceil(hi / 10) * 10;
  const X = (f) => pad + (f / Math.max(1, n - 1)) * (W - pad - 8);
  const Y = (v) => 8 + (1 - (v - lo) / Math.max(1e-6, hi - lo)) * (H - 30);
  ctx.font = "11px Segoe UI, sans-serif";
  ctx.fillStyle = "#8b95a3";
  ctx.strokeStyle = "#232a33";
  for (let v = lo; v <= hi; v += Math.max(10, Math.round((hi - lo) / 5 / 10) * 10)) {
    ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(W - 8, Y(v)); ctx.stroke();
    ctx.fillText(`${v}°`, 2, Y(v) + 4);
  }
  ctx.strokeStyle = "rgba(255,255,255,.12)";
  for (const s of A.shots.slice(1)) { ctx.beginPath(); ctx.moveTo(X(s.start), 8); ctx.lineTo(X(s.start), H - 22); ctx.stroke(); }
  for (const s of SERIES) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.3; ctx.beginPath();
    const step = Math.max(1, Math.floor(n / (W * 2)));
    for (let f = 0; f < n; f += step) { const x = X(f), y = Y(rows[f][s.i]); f ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
  }
  for (let t = 0; t <= n / A.fps; t += Math.max(5, Math.round(n / A.fps / 10 / 5) * 5)) ctx.fillText(fmtTime(t), X(t * A.fps) - 10, H - 6);
  $("camLegend").innerHTML = SERIES.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("") + `<span><i style="background:rgba(255,255,255,.3)"></i>Cut</span>`;
  if (hoverX != null) {
    const f = Math.round(((hoverX - pad) / (W - pad - 8)) * (n - 1));
    if (f >= 0 && f < n) {
      ctx.strokeStyle = "#fff"; ctx.beginPath(); ctx.moveTo(X(f), 8); ctx.lineTo(X(f), H - 22); ctx.stroke();
      const r = rows[f];
      $("camHover").textContent = `Frame ${f} (${fmtTime(f / A.fps)}) · shot ${r[5] + 1} · yaw ${r[0].toFixed(1)}° · pitch ${r[1].toFixed(1)}° · roll ${r[2].toFixed(1)}° · vertical FOV ${r[3].toFixed(1)}°`;
    }
  }
}
$("camChart").addEventListener("mousemove", (e) => drawCamChart(e.offsetX));
window.addEventListener("resize", () => drawCamChart());

async function startCapture(cid) {
  try {
    if (!S.state?.vrm) { toast("Choose your avatar (.vrm or .fbx) first (step 1).", true); showStep("setup"); return; }
    await api("/api/capture", { character: cid, smoothing: Number($("smoothing").value), foot_lock: $("footLock").checked });
    const st = await watchJob();
    await refreshState();
    if (!st.error) openAnimation(cid);
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ step 3: viewer
async function ensureViewer() {
  if (!S.viewer) S.viewer = new Viewer($("gl"), $("pane3d"));
  if (!S.vrmLoaded) {
    setJob({ message: "Loading avatar…", progress: 0.5, done: false });
    const info = await api("/api/avatar_info");
    if (info.kind === "fbx") await S.viewer.loadFBX(`/api/vrm?${Date.now()}`, info);
    else await S.viewer.loadVRM(`/api/vrm?${Date.now()}`);
    S.vrmLoaded = true;
    setJob(null);
  }
}
async function openAnimation(cid) {
  try {
    showStep("viewer");
    await ensureViewer();
    const keepT = S.character === cid && S.anim ? S.frame / S.anim.fps : null;
    const fps = Number($("fpsSel").value);
    const meta = await api(`/api/anim/${cid}/meta?fps=${fps}`);
    const buf = await (await fetch(`/api/anim/${cid}/bin?fps=${fps}`)).arrayBuffer();
    S.viewer.setAnimation(meta, buf);
    S.anim = S.viewer.anim;
    setRotation(meta.rotation_deg || 0, false);
    loadAudio();
    const pr = meta.proportions && meta.proportions.share_of_leg;
    const f2 = (v) => (v == null ? "–" : Number(v).toFixed(2));
    $("propInfo").textContent = pr
      ? `Proportions measured over the whole video (share of the leg length): upper arm ${f2(pr.upper_arm)} · forearm ${f2(pr.forearm)} · thigh ${f2(pr.thigh)} · shin ${f2(pr.shin)} · shoulders ${f2(pr.shoulder_width)} · hips ${f2(pr.hip_width)}`
      : "";
    S.character = cid;
    $("steps").children[2].disabled = false;
    $("steps").children[3].disabled = false;
    await loadKeys(cid);
    await loadFacing(cid);
    const c = S.analysis.characters.find((x) => x.id === cid);
    $("viewerChar").textContent = c ? c.name : "";
    $("scrub").max = meta.frames - 1;
    if (S.crop && S.cropFps && S.cropFps !== meta.fps) {  // keep the same moments at another frame rate
      const k = meta.fps / S.cropFps;
      S.crop = [Math.round(S.crop[0] * k), Math.round(S.crop[1] * k)];
    }
    S.cropFps = meta.fps;
    setCrop(S.crop && S.character === cid ? S.crop[0] : null, S.crop ? S.crop[1] : null);
    seek(keepT != null ? keepT * meta.fps : firstValid());
  } catch (e) { toast(e.message, true); }
}
// ------------------------------------------------------------------ rotate the whole animation
let rotTimer = null;
function setRotation(deg, save = true) {
  deg = Math.max(-180, Math.min(180, Math.round(Number(deg) || 0)));
  $("rotSlider").value = deg; $("rotNum").value = deg;
  if (S.viewer) S.viewer.yaw = (deg * Math.PI) / 180;
  if (save && S.character != null) {
    clearTimeout(rotTimer);
    rotTimer = setTimeout(() => api(`/api/rotation/${S.character}`, { degrees: deg }).catch((e) => toast("Rotation not saved: " + e.message, true)), 300);
  }
}
$("rotSlider").oninput = (e) => setRotation(e.target.value);
$("rotNum").onchange = (e) => setRotation(e.target.value);
$("rotReset").onclick = () => setRotation(0);

// ------------------------------------------------------------------ facing per shot (close-ups)
const F = { hints: [], dirty: false };
async function loadFacing(cid) {
  try { F.hints = (await api(`/api/facing/${cid}`)).hints || []; } catch (e) { F.hints = []; }
  F.dirty = false;
  $("facingApply").disabled = true;
  refreshFacing();
}
function facingAt(src) {
  const h = F.hints.find((x) => x.start <= src && src < x.end);
  return h ? h.facing : "auto";
}
function refreshFacing() {
  if (!S.anim) return;
  const cur = facingAt(S.anim.src[S.frame]);
  document.querySelectorAll("#facingSeg button").forEach((b) => b.classList.toggle("active", b.dataset.facing === cur));
  const n = F.hints.length;
  $("facingInfo").textContent = (n ? `${n} shot${n > 1 ? "s" : ""} set` : "") + (F.dirty ? " · changed: apply to re-solve" : "");
}
document.querySelectorAll("#facingSeg button").forEach((b) => (b.onclick = async () => {
  if (S.character == null || !S.anim) return;
  try {
    const r = await api(`/api/facing/${S.character}`, { frame: S.anim.src[S.frame], facing: b.dataset.facing });
    F.hints = r.hints || [];
    F.dirty = true;
    $("facingApply").disabled = false;
    refreshFacing();
    drawTrack();
  } catch (e) { toast(e.message, true); }
}));
$("facingApply").onclick = () => { if (S.character != null) startCapture(S.character); };

// ------------------------------------------------------------------ step 4: fingers & face keys
const K = { channels: {}, rig: [], expressions: [], defaults: {}, tab: "hands", saveTimer: null, cid: null };
const FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"];
function kSlopes(t, v) {  // monotone cubic (same as keys.py)
  const n = t.length, m = new Array(n).fill(0);
  if (n < 2) return m;
  const h = [], d = [];
  for (let i = 0; i < n - 1; i++) { h.push(t[i + 1] - t[i]); d.push((v[i + 1] - v[i]) / (h[i] > 0 ? h[i] : 1e-9)); }
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] * d[i] <= 0) m[i] = 0;
    else { const w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1]; m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]); }
  }
  for (let i = 0; i < n - 1; i++) if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; }
  return m;
}
function kEval(keys, time, def) {
  if (!keys || !keys.length) return def;
  if (keys.length === 1 || time <= keys[0][0]) return keys[0][1];
  const n = keys.length;
  if (time >= keys[n - 1][0]) return keys[n - 1][1];
  const t = keys.map((k) => k[0]), v = keys.map((k) => k[1]);
  if (!keys._m) keys._m = kSlopes(t, v);  // cached per edit (arrays are replaced on change)
  const m = keys._m;
  let i = 0;
  while (i < n - 2 && t[i + 1] <= time) i++;
  const h = t[i + 1] - t[i], u = Math.min(1, Math.max(0, (time - t[i]) / (h > 0 ? h : 1e-9)));
  const h00 = 2 * u ** 3 - 3 * u ** 2 + 1, h10 = u ** 3 - 2 * u ** 2 + u, h01 = -2 * u ** 3 + 3 * u ** 2, h11 = u ** 3 - u ** 2;
  return h00 * v[i] + h10 * h * m[i] + h01 * v[i + 1] + h11 * h * m[i + 1];
}
function kValue(ch, time) { return kEval(K.channels[ch], time, K.defaults[ch] ?? 0); }
function kAll(time) { const o = {}; for (const ch of Object.keys(K.defaults)) o[ch] = kValue(ch, time); return o; }
const kTol = () => 0.5 / (S.anim?.fps || 30);
const kNow = () => S.frame / S.anim.fps;
function kKeyAt(ch, time) { const ks = K.channels[ch]; return ks ? ks.findIndex((k) => Math.abs(k[0] - time) < kTol()) : -1; }
function kSet(ch, time, v) {
  const ks = (K.channels[ch] || []).filter((k) => Math.abs(k[0] - time) >= kTol());
  ks.push([Math.round(time * 10000) / 10000, v]);
  ks.sort((a, b) => a[0] - b[0]);
  K.channels[ch] = ks;
  kChanged();
}
function kDel(ch, time) {
  if (!K.channels[ch]) return;
  const ks = K.channels[ch].filter((k) => Math.abs(k[0] - time) >= kTol());
  if (ks.length) K.channels[ch] = ks; else delete K.channels[ch];
  kChanged();
}
function kChanged() {
  clearTimeout(K.saveTimer);
  const cid = K.cid;
  const body = { channels: Object.fromEntries(Object.entries(K.channels).map(([c, ks]) => [c, ks.map((k) => [k[0], k[1]])])) };
  K.saveTimer = setTimeout(() => api(`/api/keys/${cid}`, body).catch((e) => toast("Keys not saved: " + e.message, true)), 400);
  refreshKeyPanel();
  drawTrack();
}
async function loadKeys(cid) {
  try {
    const info = await api(`/api/keys/${cid}`);
    Object.assign(K, { channels: info.channels || {}, rig: info.rig || [], expressions: info.expressions || [], defaults: info.defaults || {}, cid });
    S.viewer.setKeys(K.rig, kAll, K.expressions);
    buildKeyPanel();
  } catch (e) { toast("Finger / face keys unavailable: " + e.message, true); }
}
function tabChannels(tab = K.tab) {
  return Object.keys(K.defaults).filter((c) => (tab === "face") === c.startsWith("expr."));
}
function kRow(ch, label, lo, hi, step) {
  const row = document.createElement("div");
  row.className = "kp-row";
  row.dataset.ch = ch;
  row.innerHTML = `<span class="nm">${label}</span><input type="range" min="${lo}" max="${hi}" step="${step}"><input type="number" min="${lo}" max="${hi}" step="${step}"><span class="kd" title="Key at this frame (click to add / remove)">◆</span>`;
  const [rng, num, kd] = [row.children[1], row.children[2], row.children[3]];
  const set = (v) => { v = Math.max(lo, Math.min(hi, Number(v))); if (!Number.isFinite(v)) return; kSet(ch, kNow(), v); };
  rng.oninput = () => set(rng.value);
  num.onchange = () => set(num.value);
  kd.onclick = () => { const t = kNow(); if (kKeyAt(ch, t) >= 0) kDel(ch, t); else kSet(ch, t, kValue(ch, t)); };
  return row;
}
const PRESETS = {
  Relaxed: { Thumb: [10, 2], Index: [15, 2], Middle: [15, 2], Ring: [15, 2], Little: [15, 2] },
  Open: { Thumb: [0, 6], Index: [0, 6], Middle: [0, 6], Ring: [0, 6], Little: [0, 6] },
  Fist: { Thumb: [70, 0], Index: [100, 0], Middle: [100, 0], Ring: [100, 0], Little: [100, 0] },
  Point: { Thumb: [60, 0], Index: [0, 0], Middle: [100, 0], Ring: [100, 0], Little: [100, 0] },
  Peace: { Thumb: [70, 0], Index: [0, 6], Middle: [0, 6], Ring: [100, 0], Little: [100, 0] },
};
function buildKeyPanel() {
  const hands = $("kpHands");
  hands.innerHTML = "";
  for (const side of ["left", "right"]) {
    const h = document.createElement("h4");
    h.innerHTML = `${side === "left" ? "Left" : "Right"} hand <span class="presets"></span>`;
    for (const name of Object.keys(PRESETS)) {
      const b = document.createElement("button");
      b.className = "ghost small"; b.textContent = name; b.title = `Key the ${side} hand as “${name}” at this frame`;
      b.onclick = () => { const t = kNow(); for (const [f, [c, sp]] of Object.entries(PRESETS[name])) { kSetQuiet(`${side}${f}.curl`, t, c); kSetQuiet(`${side}${f}.splay`, t, sp); } kChanged(); };
      h.querySelector(".presets").appendChild(b);
    }
    const mb = document.createElement("button");
    mb.className = "ghost small"; mb.textContent = "⇄"; mb.title = `Copy this hand's values at this frame to the other hand (as keys)`;
    mb.onclick = () => { const t = kNow(), o = side === "left" ? "right" : "left"; for (const f of FINGER_NAMES) for (const k of ["curl", "splay"]) kSetQuiet(`${o}${f}.${k}`, t, kValue(`${side}${f}.${k}`, t)); kChanged(); };
    h.querySelector(".presets").appendChild(mb);
    hands.appendChild(h);
    for (const f of FINGER_NAMES) {
      if (!(`${side}${f}.curl` in K.defaults)) continue;
      const box = document.createElement("div");
      box.className = "kp-finger";
      box.appendChild(kRow(`${side}${f}.curl`, `${f} curl`, -10, 100, 1));
      box.appendChild(kRow(`${side}${f}.splay`, "splay", 0, 10, 0.1));
      hands.appendChild(box);
    }
  }
  const list = $("kpExprList");
  list.innerHTML = "";
  if (!K.expressions.length) list.innerHTML = `<p class="muted small">This avatar has no expressions or blend shapes.</p>`;
  for (const nm of K.expressions) list.appendChild(kRow("expr." + nm, nm, 0, 100, 1));
  refreshKeyPanel();
}
function kSetQuiet(ch, time, v) {
  const ks = (K.channels[ch] || []).filter((k) => Math.abs(k[0] - time) >= kTol());
  ks.push([Math.round(time * 10000) / 10000, v]);
  ks.sort((a, b) => a[0] - b[0]);
  K.channels[ch] = ks;
}
function refreshKeyPanel() {
  if ($("keyPanel").hidden || !S.anim) return;
  const t = kNow();
  for (const row of document.querySelectorAll("#keyPanel .kp-row")) {
    const ch = row.dataset.ch, v = kValue(ch, t);
    const [rng, num, kd] = [row.children[1], row.children[2], row.children[3]];
    if (document.activeElement !== rng) rng.value = v;
    if (document.activeElement !== num) num.value = Math.round(v * 10) / 10;
    kd.classList.toggle("on", kKeyAt(ch, t) >= 0);
    kd.classList.toggle("has", !!K.channels[ch]);
  }
  const f = $("kpFilter").value.trim().toLowerCase();
  for (const row of $("kpExprList").children) if (row.dataset.ch) row.hidden = f && !row.dataset.ch.slice(5).toLowerCase().includes(f);
  const n = tabChannels().reduce((a, c) => a + (K.channels[c]?.length || 0), 0);
  $("kpCount").textContent = `${n} key${n === 1 ? "" : "s"} in this tab`;
}
function keyTimes(tab) {
  const ts = new Set();
  for (const c of tabChannels(tab)) for (const k of K.channels[c] || []) ts.add(k[0]);
  return [...ts].sort((a, b) => a - b);
}
function jumpKey(dir) {
  const t = kNow(), ts = keyTimes();
  const tgt = dir > 0 ? ts.find((x) => x > t + kTol()) : [...ts].reverse().find((x) => x < t - kTol());
  if (tgt != null) { S.playing = false; updatePlay(); seek(Math.round(tgt * S.anim.fps)); }
}
document.querySelectorAll("#kpTabs button").forEach((b) => (b.onclick = () => {
  K.tab = b.dataset.tab;
  document.querySelectorAll("#kpTabs button").forEach((x) => x.classList.toggle("active", x === b));
  $("kpHands").hidden = K.tab !== "hands";
  $("kpFace").hidden = K.tab !== "face";
  refreshKeyPanel();
}));
$("kpPrev").onclick = () => jumpKey(-1);
$("kpNext").onclick = () => jumpKey(1);
$("kpKeyAll").onclick = () => { const t = kNow(); for (const c of tabChannels()) kSetQuiet(c, t, kValue(c, t)); kChanged(); };
$("kpDelHere").onclick = () => { const t = kNow(); for (const c of tabChannels()) { const ks = (K.channels[c] || []).filter((k) => Math.abs(k[0] - t) >= kTol()); if (ks.length) K.channels[c] = ks; else delete K.channels[c]; } kChanged(); };
$("kpClear").onclick = () => {
  if (!confirm(`Delete every ${K.tab === "face" ? "expression" : "finger"} key of this character?`)) return;
  for (const c of tabChannels()) delete K.channels[c];
  kChanged();
};
$("kpFilter").oninput = () => refreshKeyPanel();

function firstValid() { const v = S.anim.valid; for (let i = 0; i < v.length; i++) if (v[i] > 0.5) return i; return 0; }
$("otherChar").onclick = () => showStep("analysis");

let imgBusy = false, wantFrame = -1;
function loadFrameImage(f) {
  wantFrame = f;
  if (imgBusy) return;
  imgBusy = true;
  const targets = [$("frameImg"), S.ovImg].filter(Boolean);
  const url = `/api/frame/${S.anim.src[f]}.jpg?w=${S.layout === "overlay" ? 1280 : 960}`;
  const img = new Image();
  img.onload = () => {
    for (const t of targets) t.src = img.src;
    imgBusy = false;
    sizeLm();
    if (wantFrame !== f) loadFrameImage(wantFrame);
  };
  img.onerror = () => { imgBusy = false; };
  img.src = url;
}
function sizeLm() {
  const img = $("frameImg"), cv = $("lmCanvas");
  if (!img.naturalWidth || !S.anim) return;
  cv.width = img.clientWidth; cv.height = img.clientHeight;
  cv.style.left = img.offsetLeft + "px"; cv.style.top = img.offsetTop + "px";
  const ctx = cv.getContext("2d");
  if ($("showLm").checked) drawLandmarks(ctx, S.anim, S.frame, img.clientWidth / S.anim.meta.width, img.clientHeight / S.anim.meta.height);
  else ctx.clearRect(0, 0, cv.width, cv.height);
}
function seek(f) {
  if (!S.anim) return;
  f = Math.max(0, Math.min(S.anim.frames - 1, Math.round(f)));
  S.frame = f;
  S.t = f / S.anim.fps;
  S.viewer.setFrame(f);
  $("scrub").value = f;
  const fps = S.anim.fps;
  $("timecode").textContent = `${fmtTime(f / fps)}.${String(Math.floor(((f / fps) % 1) * 100)).padStart(2, "0")} · frame ${f} / ${S.anim.frames - 1} @ ${S.anim.fps} fps`;
  const shot = S.anim.shot[f];
  const row = S.analysis?.camera?.[S.anim.src[f]];
  $("frameInfo").textContent = (S.anim.valid[f] > 0.5 ? "Tracked" : "Character not visible (pose interpolated)") +
    ` · shot ${shot + 1}` + (row ? ` · camera yaw ${row[0].toFixed(1)}° pitch ${row[1].toFixed(1)}° roll ${row[2].toFixed(1)}° · FOV ${S.anim.vfov[f].toFixed(1)}°` : "") +
    ` · hands ${S.anim.hs[f * 2] >= 0.5 ? "L✓" : "L–"} ${S.anim.hs[f * 2 + 1] >= 0.5 ? "R✓" : "R–"}`;
  $("camBadge").textContent = S.viewer.mode === "video" || S.layout === "overlay" ? "Video camera" : "Free camera (drag to orbit, right-drag to pan, wheel to zoom)";
  if (S.layout !== "3d") loadFrameImage(f);
  else sizeLm();
  refreshKeyPanel();
  refreshFacing();
}
$("scrub").oninput = (e) => { S.playing = false; updatePlay(); seek(Number(e.target.value)); };
$("stepBack").onclick = () => { S.playing = false; updatePlay(); seek(S.frame - 1); };
$("stepFwd").onclick = () => { S.playing = false; updatePlay(); seek(S.frame + 1); };
function updatePlay() { $("playBtn").textContent = S.playing ? "⏸" : "▶"; syncAudio(); }
$("playBtn").onclick = () => {
  S.playing = !S.playing;
  const [a, b] = playRange();
  if (S.playing && (S.frame >= b || S.frame < a)) seek(a);
  S.last = performance.now(); updatePlay();
};
// ---- range (crop) for looping and export
function playRange() {
  if (S.crop) return [S.crop[0], S.crop[1]];
  return [0, S.anim.frames - 1];
}
function setCrop(a, b) {
  if (a == null) S.crop = null;
  else S.crop = [Math.max(0, Math.min(a, b)), Math.min(S.anim.frames - 1, Math.max(a, b))];
  $("cropClear").disabled = !S.crop;
  const fps = S.anim.fps;
  $("cropInfo").textContent = S.crop
    ? `Range ${fmtTime(S.crop[0] / fps)} – ${fmtTime(S.crop[1] / fps)} (frames ${S.crop[0]}–${S.crop[1]}, ${S.crop[1] - S.crop[0] + 1} frames)`
    : "Whole clip";
  drawTrack();
}
$("cropIn").onclick = () => setCrop(S.frame, S.crop ? Math.max(S.crop[1], S.frame) : S.anim.frames - 1);
$("cropOut").onclick = () => setCrop(S.crop ? Math.min(S.crop[0], S.frame) : 0, S.frame);
$("cropClear").onclick = () => setCrop(null);
document.addEventListener("keydown", (e) => {
  if (!$("step-viewer").classList.contains("active") || e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if (e.target.closest && e.target.closest("#keyPanel") && e.target.type === "range" && e.code.startsWith("Arrow")) return;
  if (e.code === "Space") { e.preventDefault(); $("playBtn").click(); }
  if (e.shiftKey && !$("keyPanel").hidden && (e.code === "ArrowLeft" || e.code === "ArrowRight")) { e.preventDefault(); jumpKey(e.code === "ArrowLeft" ? -1 : 1); return; }
  if (e.code === "ArrowLeft") { e.preventDefault(); $("stepBack").click(); }
  if (e.code === "ArrowRight") { e.preventDefault(); $("stepFwd").click(); }
  if (e.code === "KeyK" && !$("keyPanel").hidden) $("kpKeyAll").click();
  if (e.code === "KeyI") $("cropIn").click();
  if (e.code === "KeyO") $("cropOut").click();
  if (e.code === "KeyL") $("loop").click();
  if (e.code === "KeyM") $("muteBtn").click();
});
function tick(now) {
  requestAnimationFrame(tick);
  if (!S.playing || !S.anim) return;
  const dt = (now - (S.last || now)) / 1000;
  S.last = now;
  S.t += dt * Number($("speed").value);
  const [a, b] = playRange();
  let f = Math.floor(S.t * S.anim.fps);
  if (f > b || f >= S.anim.frames - 1 && b >= S.anim.frames - 1) {
    if ($("loop").checked) { S.t = a / S.anim.fps; f = a; }
    else { S.playing = false; updatePlay(); seek(b); return; }
  }
  if (f !== S.frame) { const t = S.t; seek(f); S.t = t; }
  syncAudio();
}
requestAnimationFrame(tick);

// ---- sound of the reference video: follows playback (speed, seeking, loop); off by default
const audio = $("refAudio");
let soundOn = false;
try { const v = localStorage.getItem("vm.volume"); if (v != null) $("volume").value = v; } catch (e) { /* no storage */ }
audio.volume = Number($("volume").value);
function loadAudio() {
  audio.pause();
  audio.removeAttribute("src");
  if (soundOn) { audio.src = `/api/media?t=${Date.now()}`; }
  updateSound();
}
function updateSound() {
  $("muteBtn").textContent = soundOn ? "🔊" : "🔇";
  $("muteBtn").closest(".audio").classList.toggle("off", !soundOn);
}
// where the sound should be (s) for the current playback position
function audioTime() {
  const A = S.anim;
  const sfps = A.meta.src_fps || A.fps;
  const f = S.frame;
  return A.src[f] / sfps + Math.max(0, S.t - f / A.fps);
}
function syncAudio() {
  if (!S.anim || !soundOn || !S.playing) {
    if (!audio.paused) audio.pause();
    return;
  }
  if (!audio.getAttribute("src")) audio.src = `/api/media?t=${Date.now()}`;
  const rate = Number($("speed").value);
  if (audio.playbackRate !== rate) audio.playbackRate = rate;
  const want = audioTime();
  if (audio.paused) {
    try { audio.currentTime = want; } catch (e) { /* not loaded yet */ }
    audio.play().catch(() => {});
  } else if (Math.abs(audio.currentTime - want) > 0.15 * Math.max(rate, 1)) {
    audio.currentTime = want;  // jumped (seek, loop, a cut in the animation) or drifted
  }
}
$("muteBtn").onclick = () => { soundOn = !soundOn; updateSound(); syncAudio(); };
$("volume").oninput = (e) => {
  audio.volume = Number(e.target.value);
  try { localStorage.setItem("vm.volume", e.target.value); } catch (er) { /* no storage */ }
  if (!soundOn && audio.volume > 0) { soundOn = true; updateSound(); syncAudio(); }
};
audio.addEventListener("error", () => {
  if (soundOn && audio.getAttribute("src")) toast("This video's sound cannot be played here", true);
});

function drawTrack() {
  const cv = $("trackCanvas"), A = S.anim;
  if (!A) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr; cv.height = 18 * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cv.clientWidth, n = A.frames;
  for (let x = 0; x < W; x++) {
    const f = Math.floor((x / W) * n);
    ctx.fillStyle = A.valid[f] > 0.5 ? "#2c6e68" : "#2a2f36";
    ctx.fillRect(x, 0, 1, 8);
    const h = (A.hs[f * 2] >= 0.5) + (A.hs[f * 2 + 1] >= 0.5);
    ctx.fillStyle = ["#2a2f36", "#7b5a2a", "#c98a3a"][h];
    ctx.fillRect(x, 10, 1, 6);
    if (F.hints.length && A.src) {  // shots whose facing was set: blue = front, purple = back
      const fc = facingAt(A.src[f]);
      if (fc !== "auto") { ctx.fillStyle = fc === "front" ? "#5aa9e6" : "#a77be6"; ctx.fillRect(x, 16, 1, 2); }
    }
  }
  if (K.cid === S.character) {  // hand-set keys: amber ticks (bright in step 4)
    ctx.fillStyle = $("keyPanel").hidden ? "rgba(246,173,85,.45)" : "#f6ad55";
    for (const tab of ["hands", "face"]) for (const t of keyTimes(tab)) ctx.fillRect(Math.round((t * A.fps / n) * W), tab === "hands" ? 0 : 9, 2, 9);
  }
  if (S.crop) {
    const x0 = (S.crop[0] / n) * W, x1 = ((S.crop[1] + 1) / n) * W;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(0, 0, x0, 18); ctx.fillRect(x1, 0, W - x1, 18);
    ctx.fillStyle = "#e8c547";
    ctx.fillRect(x0, 0, 2, 18); ctx.fillRect(x1 - 2, 0, 2, 18);
  }
}
$("trackCanvas").title = "Top: character tracked · bottom: hands tracked (dim = one hand, bright = both). Click to jump, drag to select a range.";
// click: jump there · drag: select a range (for looping and export)
{
  const tc = $("trackCanvas");
  let down = null;
  const frameAt = (e) => Math.max(0, Math.min(S.anim.frames - 1, Math.round((e.offsetX / tc.clientWidth) * S.anim.frames)));
  tc.onmousedown = (e) => { if (S.anim) down = frameAt(e); };
  tc.onmousemove = (e) => { if (down != null && Math.abs(frameAt(e) - down) > 2) setCrop(down, frameAt(e)); };
  window.addEventListener("mouseup", (e) => {
    if (down == null) return;
    const f = e.target === tc ? frameAt(e) : null;
    if (f != null && Math.abs(f - down) <= 2) seek(f);
    else if (S.crop) seek(S.crop[0]);
    down = null;
  });
}

document.querySelectorAll("#layoutSeg button").forEach((b) => (b.onclick = () => setLayout(b.dataset.layout)));
function setLayout(l) {
  S.layout = l;
  document.querySelectorAll("#layoutSeg button").forEach((x) => x.classList.toggle("active", x.dataset.layout === l));
  const st = $("stage");
  st.classList.toggle("overlay", l === "overlay");
  st.classList.toggle("only3d", l === "3d");
  if (l === "overlay") {
    if (!S.ovImg) { S.ovImg = document.createElement("img"); S.ovImg.className = "ovimg"; $("pane3d").insertBefore(S.ovImg, $("gl")); }
    S.ovImg.hidden = false;
    S.viewer.setOverlay(S.ovImg);
  } else {
    if (S.ovImg) S.ovImg.hidden = true;
    S.viewer.setOverlay(null);
  }
  setTimeout(() => { S.viewer.resize(); seek(S.frame); }, 30);
}
document.querySelectorAll("#camSeg button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll("#camSeg button").forEach((x) => x.classList.toggle("active", x === b));
  S.viewer.setMode(b.dataset.cam);
  seek(S.frame);
}));
$("lockMove").onchange = (e) => { S.viewer.lockMove = e.target.checked; $("lockHeight").disabled = !e.target.checked; $("exInPlace").checked = e.target.checked; };
$("lockHeight").onchange = (e) => { S.viewer.lockHeight = e.target.checked; $("exLockHeight").checked = e.target.checked; };
$("showLm").onchange = () => sizeLm();
$("showGrid").onchange = (e) => (S.viewer.showFloor = e.target.checked);
new ResizeObserver(() => sizeLm()).observe($("paneVideo"));
$("frameImg").addEventListener("load", () => sizeLm());

$("exportBtn").onclick = () => {
  $("exFps").textContent = `${S.anim.frames} frames at exactly ${S.anim.fps} fps (change it with “Frame rate” in the toolbar).`;
  $("exCropRow").hidden = !S.crop;
  if (S.crop) $("exCropInfo").textContent = `(${$("cropInfo").textContent.replace(/^Range /, "")})`;
  $("exportDlg").showModal();
};
$("fpsSel").onchange = () => { if (S.character != null) openAnimation(S.character); };
$("exportDlg").addEventListener("close", async () => {
  if ($("exportDlg").returnValue !== "ok") return;
  const opts = { character: S.character, in_place: $("exInPlace").checked, lock_height: $("exLockHeight").checked, camera: $("exCamera").checked, fps: S.anim.fps, model: $("exModel").checked };
  if (S.crop && $("exCrop").checked) { opts.start = S.crop[0]; opts.end = S.crop[1]; }
  const c = S.analysis.characters.find((x) => x.id === S.character);
  const base = (S.state.video || "video").split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
  const rng = opts.start != null ? `_${opts.start}-${opts.end}` : "";
  const name = `${base}_${(c?.name || "character").replace(/[^\w-]+/g, "_")}${rng}.fbx`;
  try {
    if (native()) {
      const path = await native().save_file(name);
      if (!path) return;
      setJob({ message: "Writing FBX…", progress: 0.5, done: false });
      await api("/api/export", { ...opts, path });
      setJob(null);
      toast(`Saved ${path}`);
    } else {
      const q = new URLSearchParams({ character: S.character, in_place: opts.in_place ? 1 : 0, lock_height: opts.lock_height ? 1 : 0, camera: opts.camera ? 1 : 0, fps: opts.fps, model: opts.model ? 1 : 0 });
      if (opts.start != null) { q.set("start", opts.start); q.set("end", opts.end); }
      const a = document.createElement("a");
      a.href = `/api/export.fbx?${q}`;
      a.download = name;
      a.click();
    }
  } catch (e) { setJob(null); toast(e.message, true); }
});

// ------------------------------------------------------------------ boot
(async () => {
  const st = await refreshState();
  if (st.video) {
    $("videoPath").textContent = st.video;
    if (st.analysis) await loadAnalysis();
  }
  if (st.vrm) $("vrmPath").textContent = st.vrm;
  if (st.job && !st.job.done) watchJob().then(() => refreshState());
})();
