const $ = s => document.querySelector(s);
const canvas = $("#map"), ctx = canvas.getContext("2d");

const THERMAL = ["#0b1d3a", "#1b4f8a", "#2aa198", "#e9d66b", "#f28c3b", "#c1352b"];
const SPEED = ["#0a1220", "#1d3f6e", "#3fa7ff", "#bfe6ff", "#ffffff"];
const LAND = "#1a2434";

let D = null;
let state = { field: "T", day: 0, pinn: true, ref: true, flow: false, timer: null };
let geom = null;

const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
function ramp(stops) {
  const rgb = stops.map(hex);
  return v => {
    if (v == null || Number.isNaN(v)) return LAND;
    const s = Math.min(Math.max(v, 0), 0.9999) * (rgb.length - 1);
    const i = Math.floor(s), f = s - i;
    const c = rgb[i].map((a, k) => Math.round(a + (rgb[i + 1][k] - a) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
}
const cmapT = ramp(THERMAL), cmapS = ramp(SPEED);

function range(frames) {
  let lo = Infinity, hi = -Infinity;
  for (const f of frames) for (const row of f) for (const v of row) if (v != null) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

function speedFrames(U, V) {
  return U.map((f, k) => f.map((row, j) => row.map((u, i) => (u == null || V[k][j][i] == null) ? null : Math.hypot(u, V[k][j][i]))));
}

function layout() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const nx = D.x.length, ny = D.y.length;
  const dx = D.x[1] - D.x[0], dy = D.y[1] - D.y[0];
  const kmW = nx * dx, kmH = ny * dy;
  const pad = 14;
  const s = Math.min((w - 2 * pad) / kmW, (h - 2 * pad) / kmH);
  const ox = (w - kmW * s) / 2, oy = (h - kmH * s) / 2;
  geom = { w, h, s, ox, oy, dx, dy, nx, ny, kmH,
    px: i => ox + (i * dx) * s,
    py: j => oy + (kmH - (j + 1) * dy) * s };
}

function drawField() {
  const { s, dx, dy, nx, ny, px, py } = geom;
  let frames, cmap, unit;
  if (state.field === "speed") { frames = D._speed; cmap = cmapS; unit = "m/s"; }
  else { frames = D[state.field]; cmap = cmapT; unit = "°C"; }
  const [lo, hi] = state.field === "speed" ? D._speedRange : D._Trange;
  const f = frames[state.day];
  ctx.fillStyle = "#060c16"; ctx.fillRect(0, 0, geom.w, geom.h);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = f[j][i];
    ctx.fillStyle = v == null ? LAND : cmap((v - lo) / (hi - lo));
    ctx.fillRect(px(i), py(j), dx * s + 0.6, dy * s + 0.6);
  }
  $("#cb-lo").textContent = lo.toFixed(state.field === "speed" ? 2 : 1) + " " + unit;
  $("#cb-hi").textContent = hi.toFixed(state.field === "speed" ? 2 : 1) + " " + unit;
  $("#cb-ramp").style.background = `linear-gradient(90deg, ${(state.field === "speed" ? SPEED : THERMAL).join(",")})`;
}

function arrows(U, V, color, width) {
  const { s, dx, dy, nx, ny, px, py } = geom;
  const stepX = Math.max(1, Math.round(nx / 24)), stepY = Math.max(1, Math.round(ny / 18));
  const pxPerMs = Math.min(dx, dy) * s * 2.2 / 0.1;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width; ctx.lineCap = "round";
  for (let j = Math.floor(stepY / 2); j < ny; j += stepY) for (let i = Math.floor(stepX / 2); i < nx; i += stepX) {
    const u = U[j][i], v = V[j][i];
    if (u == null || v == null) continue;
    const x0 = px(i) + dx * s / 2, y0 = py(j) + dy * s / 2;
    let L = Math.hypot(u, v) * pxPerMs;
    const cap = Math.min(dx, dy) * s * Math.max(stepX, stepY) * 0.95;
    const k = L > cap ? cap / L : 1;
    const ex = u * pxPerMs * k, ey = -v * pxPerMs * k;
    L *= k;
    if (L < 1.5) { ctx.beginPath(); ctx.arc(x0, y0, 1.2, 0, 7); ctx.fill(); continue; }
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + ex, y0 + ey); ctx.stroke();
    const a = Math.atan2(ey, ex), hl = Math.min(6, L * 0.45);
    ctx.beginPath();
    ctx.moveTo(x0 + ex, y0 + ey);
    ctx.lineTo(x0 + ex - hl * Math.cos(a - 0.5), y0 + ey - hl * Math.sin(a - 0.5));
    ctx.lineTo(x0 + ex - hl * Math.cos(a + 0.5), y0 + ey - hl * Math.sin(a + 0.5));
    ctx.closePath(); ctx.fill();
  }
}

function draw() {
  if (!D) return;
  layout();
  drawField();
  const k = state.day;
  if (state.flow) arrows(D.u_flow[k], D.v_flow[k], "rgba(95,211,255,.85)", 1.2);
  if (state.ref) arrows(D.u_ref[k], D.v_ref[k], "rgba(255,181,71,.95)", 1.6);
  if (state.pinn) arrows(D.u[k], D.v[k], "rgba(245,247,250,.95)", 1.6);
  $("#day-n").textContent = `day ${k}`;
  $("#day-date").textContent = D.dates ? D.dates[k] : `t = ${D.t[k]} d (synthetic twin)`;
}

function hover(ev) {
  if (!geom) return;
  const r = canvas.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  const i = Math.floor((x - geom.ox) / (geom.dx * geom.s));
  const j = Math.floor((geom.kmH - (y - geom.oy) / geom.s) / geom.dy);
  const box = $("#hover");
  if (i < 0 || j < 0 || i >= geom.nx || j >= geom.ny) { box.style.display = "none"; return; }
  const k = state.day;
  const fmt = (a, d = 3) => a == null ? "   —  " : a.toFixed(d).padStart(6);
  const pos = D.lat ? `${D.lat[j].toFixed(2)}°N  ${Math.abs(D.lon[i]).toFixed(2)}°${D.lon[i] < 0 ? "W" : "E"}` : `x ${D.x[i].toFixed(0)} km  y ${D.y[j].toFixed(0)} km`;
  box.textContent =
    `${pos}\nSST obs   ${fmt(D.T[k][j][i], 2)} °C   fit ${fmt(D.T_fit[k][j][i], 2)}\n` +
    `PINN      u ${fmt(D.u[k][j][i])}  v ${fmt(D.v[k][j][i])} m/s\n` +
    `ref       u ${fmt(D.u_ref[k][j][i])}  v ${fmt(D.v_ref[k][j][i])} m/s`;
  box.style.display = "block";
}

function metrics() {
  const m = D.metrics, rows = [
    ["vector RMSE", "rmse_vec", "m/s", 3, "lo"],
    ["speed RMSE", "rmse_speed", "m/s", 3, "lo"],
    ["correlation u", "corr_u", "", 2, "hi"],
    ["correlation v", "corr_v", "", 2, "hi"],
    ["angle error", "angle_deg", "°", 1, "lo"],
    ["speed bias", "speed_bias", "m/s", 3, "abs"],
  ];
  const tb = $("#metrics tbody"); tb.innerHTML = "";
  for (const [label, key, unit, d, best] of rows) {
    const a = m.pinn[key], b = m.optflow[key];
    const better = best === "hi" ? (a > b ? 0 : 1) : best === "abs" ? (Math.abs(a) < Math.abs(b) ? 0 : 1) : (a < b ? 0 : 1);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td class="${better === 0 ? "win" : ""}">${a.toFixed(d)}${unit}</td><td class="${better === 1 ? "win" : ""}">${b.toFixed(d)}${unit}</td>`;
    tb.appendChild(tr);
  }
  $("#metrics-note").textContent = `${m.pinn.n.toLocaleString("en")} points compared · mean reference speed ${m.pinn.ref_speed.toFixed(3)} m/s · reference: ${D.reference}`;
  $("#kappa").textContent = D.kappa_m2s.toFixed(1);
}

function lossCurve() {
  const H = D.history, svg = $("#loss");
  const W = 320, Hh = 150, pad = { l: 34, r: 8, t: 8, b: 20 };
  const xs = H.map(h => h.it), ys1 = H.map(h => Math.log10(h.data)), ys2 = H.map(h => Math.log10(h.pde));
  const xmin = xs[0], xmax = xs[xs.length - 1];
  const ymin = Math.floor(Math.min(...ys1, ...ys2)), ymax = Math.ceil(Math.max(...ys1, ...ys2));
  const X = v => pad.l + (v - xmin) / (xmax - xmin || 1) * (W - pad.l - pad.r);
  const Y = v => pad.t + (ymax - v) / (ymax - ymin || 1) * (Hh - pad.t - pad.b);
  let out = "";
  let prev = null;
  for (let i = 0; i < H.length; i++) {
    if (H[i].t_hi !== prev) {
      const x0 = X(H[i].it), x1 = X(xmax);
      out += `<rect x="${x0}" y="${pad.t}" width="${x1 - x0}" height="${Hh - pad.t - pad.b}" fill="#3fa7ff" opacity="${0.04 + 0.05 * H[i].t_hi}"/>`;
      prev = H[i].t_hi;
    }
  }
  for (let g = ymin; g <= ymax; g++) {
    out += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(g)}" y2="${Y(g)}" stroke="#22324a" stroke-width="1"/>`;
    out += `<text x="${pad.l - 4}" y="${Y(g) + 3}" fill="#5f7191" font-size="9" text-anchor="end" font-family="monospace">1e${g}</text>`;
  }
  const path = ys => ys.map((v, i) => `${i ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  out += `<path d="${path(ys1)}" fill="none" stroke="#9fd0ff" stroke-width="1.8"/>`;
  out += `<path d="${path(ys2)}" fill="none" stroke="#ffb547" stroke-width="1.8"/>`;
  out += `<text x="${W - pad.r}" y="${Hh - 6}" fill="#5f7191" font-size="9" text-anchor="end" font-family="monospace">${xmax} iterations</text>`;
  svg.innerHTML = out;
}

function play() {
  if (state.timer) { clearInterval(state.timer); state.timer = null; $("#play").textContent = "▶"; return; }
  $("#play").textContent = "❚❚";
  state.timer = setInterval(() => {
    state.day = (state.day + 1) % D.t.length;
    $("#day").value = state.day;
    draw();
  }, 450);
}

async function boot() {
  D = await (await fetch("data/run.json")).json();
  D._speed = speedFrames(D.u, D.v);
  D._speedRange = [0, range(D._speed)[1]];
  const tr = range(D.T);
  D._Trange = [tr[0], tr[1]];
  $("#day").max = D.t.length - 1;
  $("#badge-source").textContent = D.source === "sst" ? "Ifremer ODYSSEA SST + HF radar" : "synthetic twin";
  $("#badge-dates").textContent = D.dates ? `${D.dates[0]} → ${D.dates[D.dates.length - 1]}` : `${D.t.length} days`;
  $("#ref-label").textContent = D.source === "sst" ? "HF radar" : "truth";
  metrics(); lossCurve(); draw();

  $("#day").addEventListener("input", e => { state.day = +e.target.value; draw(); });
  $("#play").addEventListener("click", play);
  for (const b of document.querySelectorAll("#field-seg button")) b.addEventListener("click", () => {
    document.querySelectorAll("#field-seg button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); state.field = b.dataset.field; draw();
  });
  for (const id of ["pinn", "ref", "flow"]) $(`#lay-${id}`).addEventListener("change", e => { state[id] = e.target.checked; draw(); });
  canvas.addEventListener("mousemove", hover);
  canvas.addEventListener("mouseleave", () => $("#hover").style.display = "none");
  window.addEventListener("resize", draw);
  window.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") { state.day = Math.min(D.t.length - 1, state.day + 1); $("#day").value = state.day; draw(); }
    if (e.key === "ArrowLeft") { state.day = Math.max(0, state.day - 1); $("#day").value = state.day; draw(); }
    if (e.key === " ") { e.preventDefault(); play(); }
  });
}
boot();
