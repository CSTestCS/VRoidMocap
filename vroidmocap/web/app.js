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
  document.querySelectorAll(".step").forEach((s) => s.classList.toggle("active", s.id === "step-" + name));
  document.querySelectorAll("#steps button").forEach((b) => b.classList.toggle("active", b.dataset.step === name));
  if (name === "viewer") S.viewer?.resize();
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
    $("vrmInfo").textContent = `${res.vrm.title || "Avatar"} · VRM ${res.vrm.version === "1" ? "1.0" : "0.x"} · ${res.vrm.bones} humanoid bones`;
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
    if (!S.state?.vrm) { toast("Choose your .vrm avatar first (step 1).", true); showStep("setup"); return; }
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
    await S.viewer.loadVRM(`/api/vrm?${Date.now()}`);
    S.vrmLoaded = true;
    setJob(null);
  }
}
async function openAnimation(cid) {
  try {
    showStep("viewer");
    await ensureViewer();
    const meta = await api(`/api/anim/${cid}/meta`);
    const buf = await (await fetch(`/api/anim/${cid}/bin`)).arrayBuffer();
    S.viewer.setAnimation(meta, buf);
    S.anim = S.viewer.anim;
    S.character = cid;
    $("steps").children[2].disabled = false;
    const c = S.analysis.characters.find((x) => x.id === cid);
    $("viewerChar").textContent = c ? c.name : "";
    $("scrub").max = meta.frames - 1;
    drawTrack();
    seek(firstValid());
  } catch (e) { toast(e.message, true); }
}
function firstValid() { const v = S.anim.valid; for (let i = 0; i < v.length; i++) if (v[i] > 0.5) return i; return 0; }
$("otherChar").onclick = () => showStep("analysis");

let imgBusy = false, wantFrame = -1;
function loadFrameImage(f) {
  wantFrame = f;
  if (imgBusy) return;
  imgBusy = true;
  const targets = [$("frameImg"), S.ovImg].filter(Boolean);
  const url = `/api/frame/${f}.jpg?w=${S.layout === "overlay" ? 1280 : 960}`;
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
  $("timecode").textContent = `${fmtTime(f / fps)}.${String(Math.floor(((f / fps) % 1) * 100)).padStart(2, "0")} · frame ${f} / ${S.anim.frames - 1}`;
  const shot = S.anim.shot[f];
  const row = S.analysis?.camera?.[f];
  $("frameInfo").textContent = (S.anim.valid[f] > 0.5 ? "Tracked" : "Character not visible (pose interpolated)") +
    ` · shot ${shot + 1}` + (row ? ` · camera yaw ${row[0].toFixed(1)}° pitch ${row[1].toFixed(1)}° roll ${row[2].toFixed(1)}° · FOV ${S.anim.vfov[f].toFixed(1)}°` : "") +
    ` · hands ${S.anim.hs[f * 2] >= 0.5 ? "L✓" : "L–"} ${S.anim.hs[f * 2 + 1] >= 0.5 ? "R✓" : "R–"}`;
  $("camBadge").textContent = S.viewer.mode === "video" || S.layout === "overlay" ? "Video camera" : "Free camera (drag to orbit, right-drag to pan, wheel to zoom)";
  if (S.layout !== "3d") loadFrameImage(f);
  else sizeLm();
}
$("scrub").oninput = (e) => { S.playing = false; updatePlay(); seek(Number(e.target.value)); };
$("stepBack").onclick = () => { S.playing = false; updatePlay(); seek(S.frame - 1); };
$("stepFwd").onclick = () => { S.playing = false; updatePlay(); seek(S.frame + 1); };
function updatePlay() { $("playBtn").textContent = S.playing ? "⏸" : "▶"; }
$("playBtn").onclick = () => { S.playing = !S.playing; if (S.playing && S.frame >= S.anim.frames - 1) seek(0); S.last = performance.now(); updatePlay(); };
document.addEventListener("keydown", (e) => {
  if (!$("step-viewer").classList.contains("active") || e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if (e.code === "Space") { e.preventDefault(); $("playBtn").click(); }
  if (e.code === "ArrowLeft") { e.preventDefault(); $("stepBack").click(); }
  if (e.code === "ArrowRight") { e.preventDefault(); $("stepFwd").click(); }
});
function tick(now) {
  requestAnimationFrame(tick);
  if (!S.playing || !S.anim) return;
  const dt = (now - (S.last || now)) / 1000;
  S.last = now;
  S.t += dt * Number($("speed").value);
  const f = Math.floor(S.t * S.anim.fps);
  if (f >= S.anim.frames - 1) { S.playing = false; updatePlay(); seek(S.anim.frames - 1); return; }
  if (f !== S.frame) { const t = S.t; seek(f); S.t = t; }
}
requestAnimationFrame(tick);

function drawTrack() {
  const cv = $("trackCanvas"), A = S.anim;
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
  }
}
$("trackCanvas").title = "Top: character tracked · bottom: hands tracked (dim = one hand, bright = both). Click to jump.";
$("trackCanvas").onclick = (e) => seek((e.offsetX / e.target.clientWidth) * S.anim.frames);

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

$("exportBtn").onclick = () => $("exportDlg").showModal();
$("exportDlg").addEventListener("close", async () => {
  if ($("exportDlg").returnValue !== "ok") return;
  const opts = { character: S.character, in_place: $("exInPlace").checked, lock_height: $("exLockHeight").checked, camera: $("exCamera").checked };
  const c = S.analysis.characters.find((x) => x.id === S.character);
  const base = (S.state.video || "video").split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
  const name = `${base}_${(c?.name || "character").replace(/[^\w-]+/g, "_")}.fbx`;
  try {
    if (native()) {
      const path = await native().save_file(name);
      if (!path) return;
      setJob({ message: "Writing FBX…", progress: 0.5, done: false });
      await api("/api/export", { ...opts, path });
      setJob(null);
      toast(`Saved ${path}`);
    } else {
      const q = new URLSearchParams({ character: S.character, in_place: opts.in_place ? 1 : 0, lock_height: opts.lock_height ? 1 : 0, camera: opts.camera ? 1 : 0 });
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
