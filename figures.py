"""Figures for the README, from the same JSON the viewer reads.

    python figures.py                       # web/data/run.json, else the demo
    python figures.py --json web/data/demo.json --out docs
"""
import argparse
import io
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

BG, PANEL, LINE = "#0a1220", "#121f33", "#22324a"
INK, INK2, DIM = "#e8eef7", "#9fb0c7", "#5f7191"
PINN, REF, FLOW, ACCENT = "#f5f7fa", "#ffb547", "#5fd3ff", "#3fa7ff"
LAND = "#1a2434"

THERMAL = LinearSegmentedColormap.from_list("thermal", ["#0b1d3a", "#1b4f8a", "#2aa198", "#e9d66b", "#f28c3b", "#c1352b"])
SPEED = LinearSegmentedColormap.from_list("speed", ["#0a1220", "#10304f", "#1f5f99", "#3f8fd6", "#7fbff0"])
THERMAL.set_bad(LAND)
SPEED.set_bad(LAND)

fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "Inter" if "Inter" in fonts else "DejaVu Sans",
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": DIM, "ytick.color": DIM, "axes.edgecolor": LINE,
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
})


def arr(a):
    return np.array([[np.nan if v is None else v for v in row] for row in a], dtype=float) if np.ndim(a) == 2 \
        else np.array([arr(f) for f in a])


def read(path):
    D = json.loads(pathlib.Path(path).read_text())
    for k in ("T", "T_fit", "u", "v", "u_flow", "v_flow", "u_ref", "v_ref"):
        D[k] = arr(D[k])
    for k in ("x", "y", "t"):
        D[k] = np.array(D[k], dtype=float)
    return D


def frame_label(D, k):
    return D["dates"][k] if "dates" in D else f"day {int(D['t'][k])}"


def bare(ax, D):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(D["x"][0], D["x"][-1]); ax.set_ylim(D["y"][0], D["y"][-1])
    ax.set_aspect("equal")


def extent(D):
    dx, dy = D["x"][1] - D["x"][0], D["y"][1] - D["y"][0]
    return (D["x"][0] - dx / 2, D["x"][-1] + dx / 2, D["y"][0] - dy / 2, D["y"][-1] + dy / 2)


def stream(ax, D, u, v, vmax, color=PINN):
    sp = np.hypot(u, v)
    ax.imshow(sp, origin="lower", extent=extent(D), cmap=SPEED, vmin=0, vmax=vmax, interpolation="bilinear")
    ax.streamplot(D["x"], D["y"], np.ma.masked_invalid(u), np.ma.masked_invalid(v), density=1.25,
                  color=color, linewidth=0.75, arrowsize=0.7, broken_streamlines=True)
    bare(ax, D)


def sst(ax, D, k, lo, hi):
    ax.imshow(D["T"][k], origin="lower", extent=extent(D), cmap=THERMAL, vmin=lo, vmax=hi, interpolation="bilinear")
    bare(ax, D)


def scale_bar(ax, D, km=100):
    x0 = D["x"][0] + 0.06 * np.ptp(D["x"])
    y0 = D["y"][0] + 0.05 * np.ptp(D["y"])
    ax.plot([x0, x0 + km], [y0, y0], color=INK, lw=2, solid_capstyle="butt")
    ax.text(x0 + km / 2, y0 + 0.025 * np.ptp(D["y"]), f"{km} km", color=INK, ha="center", va="bottom", fontsize=8)


def to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------- hero

def hero_frame(D, k, lims, vmax):
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 4.9), dpi=100)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.86, bottom=0.07, wspace=0.04)
    sst(axs[0], D, k, *lims)
    stream(axs[1], D, D["u"][k], D["v"][k], vmax)
    stream(axs[2], D, D["u_ref"][k], D["v_ref"][k], vmax, color=REF)
    scale_bar(axs[0], D)
    ref = "radar HF currents" if D["source"] == "sst" else "true currents"
    for ax, title, sub in zip(axs, ["SST", "Inferred by the PINN", ref.capitalize()],
                              ["the only input", "never sees a velocity", "held out, for scoring"]):
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=18)
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")
    fig.text(0.015, 0.025, frame_label(D, k), color=INK, fontsize=10, fontweight="bold")
    fig.text(0.985, 0.025, "synthetic twin" if D["source"] == "synth" else D.get("reference", ""),
             color=DIM, fontsize=9, ha="right")
    return to_image(fig)


def hero(D, out):
    finite = D["T"][np.isfinite(D["T"])]
    lims = (np.percentile(finite, 1), np.percentile(finite, 99))
    vmax = np.nanpercentile(np.hypot(D["u_ref"], D["v_ref"]), 99)
    frames = [hero_frame(D, k, lims, vmax) for k in range(len(D["t"]))]
    strip = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for i, f in enumerate(frames[:: max(1, len(frames) // 3)][:3]):
        strip.paste(f, (0, i * f.height))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    gif[0].save(out / "hero.gif", save_all=True, append_images=gif[1:], duration=320, loop=0, optimize=True)
    frames[len(frames) // 2].save(out / "hero.png")
    return frames[len(frames) // 2]


# ---------------------------------------------------------------- comparison

def comparison(D, out, k=None):
    k = len(D["t"]) // 2 if k is None else k
    vmax = np.nanpercentile(np.hypot(D["u_ref"], D["v_ref"]), 99)
    m = D["metrics"]
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 5.3), dpi=110)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.86, bottom=0.17, wspace=0.04)
    ref = "Radar HF" if D["source"] == "sst" else "Truth"
    panels = [(ref, D["u_ref"], D["v_ref"], REF, None),
              ("PINN", D["u"], D["v"], PINN, m["pinn"]),
              ("Optical flow (Horn–Schunck)", D["u_flow"], D["v_flow"], FLOW, m["optflow"])]
    for ax, (title, U, V, c, s) in zip(axs, panels):
        stream(ax, D, U[k], V[k], vmax, color=c)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
        if s:
            ang = f"{s['angle_deg']:.0f}°" if s.get("angle_deg") is not None else "-"
            ax.text(0, -0.055, f"angle error {ang}    corr u {s['corr_u']:+.2f}  v {s['corr_v']:+.2f}    skill {s['skill']:+.2f}",
                    transform=ax.transAxes, color=INK2, fontsize=9.5, va="top")
        else:
            ax.text(0, -0.055, "held out, used only for scoring", transform=ax.transAxes, color=DIM, fontsize=9.5, va="top")
    scale_bar(axs[0], D)
    cax = fig.add_axes([0.70, 0.05, 0.28, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SPEED, norm=plt.Normalize(0, vmax)), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
    cb.set_label("speed (m/s)", color=INK2, fontsize=8.5)
    fig.text(0.015, 0.03, f"{frame_label(D, k)}  ·  one colour scale for the three panels", color=DIM, fontsize=9)
    fig.savefig(out / "comparison.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- across isotherms

def across_parts(D, U, V, q=70):
    T, x, y = D["T"], D["x"], D["y"]
    gy, gx = np.gradient(T, y, x, axis=(1, 2))
    g = np.hypot(gx, gy)
    ok = np.isfinite(g) & np.isfinite(U) & np.isfinite(V) & np.isfinite(D["u_ref"]) & np.isfinite(D["v_ref"]) & (g > 0)
    ok &= g >= np.percentile(g[ok], q)
    nx, ny = gx / np.where(g > 0, g, 1), gy / np.where(g > 0, g, 1)
    return (D["u_ref"] * nx + D["v_ref"] * ny)[ok], (U * nx + V * ny)[ok]


def across(D, out):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.9), dpi=110)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.13, wspace=0.22)
    parts = [("PINN", D["u"], D["v"], PINN, D["metrics"]["across"]["pinn"]),
             ("Optical flow", D["u_flow"], D["v_flow"], FLOW, D["metrics"]["across"]["optflow"])]
    ref, _ = across_parts(D, D["u"], D["v"])
    lim = np.percentile(np.abs(ref), 99.5) * 1.15
    for ax, (name, U, V, c, s) in zip(axs, parts):
        r, e = across_parts(D, U, V)
        ax.hexbin(r, e, gridsize=55, extent=(-lim, lim, -lim, lim), bins="log", mincnt=1,
                  cmap=LinearSegmentedColormap.from_list("h", [PANEL, "#1f5f99", c]), linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], color=REF, lw=1, ls=(0, (4, 3)))
        ax.axhline(0, color=LINE, lw=0.8); ax.axvline(0, color=LINE, lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("true speed across isotherms (m/s)")
        ax.set_ylabel(f"{name} (m/s)")
        for sp in ax.spines.values():
            sp.set_color(LINE)
        ax.set_title(name, loc="left", fontsize=12.5, fontweight="bold", pad=8)
        ss = f"{s['same_sign']:.0%}" if s.get("same_sign") is not None else "-"
        slope = np.polyfit(r, e, 1)[0]
        ax.text(0.03, 0.97, f"corr {s['corr']:+.2f}\nslope {slope:.2f}\nRMSE {s['rmse']:.3f} m/s\nright sign {ss}",
                transform=ax.transAxes, va="top", fontsize=9.5, color=INK, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.45", fc=BG, ec=LINE))
    fig.text(0.08, 0.93, "Flow across isotherms", fontsize=13.5, fontweight="bold")
    fig.text(0.08, 0.875, "A tracer only shows the part of the current that crosses its contours. Strongest 30% of SST gradients.",
             fontsize=9.5, color=INK2)
    fig.savefig(out / "across.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- training

def training(D, out, kappa_true=None):
    H = D["history"]
    it = np.array([h["it"] for h in H])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=110, gridspec_kw=dict(width_ratios=[1.6, 1]))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.15, wspace=0.28)
    edges = [0] + [H[i]["it"] for i in range(1, len(H)) if H[i]["t_hi"] != H[i - 1]["t_hi"]] + [it[-1]]
    for ax in (a1, a2):
        for j in range(len(edges) - 1):
            frac = next(h["t_hi"] for h in H if h["it"] >= edges[j])
            ax.axvspan(edges[j], edges[j + 1], color=ACCENT, alpha=0.03 + 0.06 * frac, lw=0)
        for sp in ax.spines.values():
            sp.set_color(LINE)
        ax.grid(color=LINE, lw=0.6, alpha=0.6)
        ax.set_xlabel("iteration")
        ax.set_xlim(0, it[-1])
    a1.semilogy(it, [h["data"] for h in H], color="#9fd0ff", lw=1.6, label="data misfit")
    a1.semilogy(it, [h["pde"] for h in H], color=REF, lw=1.6, label="PDE residual")
    tw = a1.twinx()
    tw.plot(it, [h["lam"] for h in H], color=FLOW, lw=1.1, ls=(0, (3, 2)), label="PDE weight λ")
    tw.set_ylim(0, max(1.05, max(h["lam"] for h in H) * 1.05))
    tw.tick_params(colors=DIM)
    tw.set_ylabel("λ", color=FLOW)
    for sp in tw.spines.values():
        sp.set_visible(False)
    l1, n1 = a1.get_legend_handles_labels()
    l2, n2 = tw.get_legend_handles_labels()
    a1.legend(l1 + l2, n1 + n2, frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
    a1.set_title("Losses", loc="left", fontsize=12, fontweight="bold")

    a2.plot(it, [h["kappa_m2s"] for h in H], color=PINN, lw=1.6, label="learned κ")
    if kappa_true:
        a2.axhline(kappa_true, color=REF, lw=1.1, ls=(0, (4, 3)), label=f"true κ = {kappa_true:.0f}")
        a2.set_ylim(0, kappa_true * 1.25)
    a2.set_ylabel("κ (m²/s)")
    a2.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    a2.set_title("Diffusivity", loc="left", fontsize=12, fontweight="bold")
    fig.text(0.07, 0.93, "Training", fontsize=13.5, fontweight="bold")
    fig.text(0.19, 0.935, "background shade = share of the time window the curriculum exposes", fontsize=9, color=DIM)
    fig.savefig(out / "training.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- social preview

def social(D, out, still):
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(BG)
    w, h = still.size
    third = still.crop((w // 3, 0, 2 * w // 3, h - int(0.06 * h)))
    ax = fig.add_axes([0.53, 0.06, 0.45, 0.88])
    ax.imshow(third); ax.axis("off")
    fig.text(0.06, 0.66, "hidden-currents", fontsize=44, fontweight="bold", color=INK)
    fig.text(0.06, 0.55, "Ocean currents inferred from sea surface\ntemperature by a physics-informed network.",
             fontsize=17, color=INK2, linespacing=1.4, va="top")
    fig.text(0.06, 0.22, "advection–diffusion PINN  ·  PyTorch  ·  checked against HF radar",
             fontsize=12.5, color=ACCENT)
    fig.text(0.06, 0.15, "Abderrazzak El Ammari", fontsize=12, color=DIM)
    fig.savefig(out / "social.png", dpi=100)
    plt.close(fig)


def main(src, out, kappa_true=None):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    D = read(src)
    if kappa_true is None and D["source"] == "synth" and pathlib.Path("data/synth.npz").exists():
        kappa_true = float(np.load("data/synth.npz")["kappa"]) * 1e6 / 86400
    still = hero(D, out)
    comparison(D, out)
    across(D, out)
    training(D, out, kappa_true)
    social(D, out, still)
    m, f = D["metrics"], D["metrics"]["across"]
    print("\n| | PINN | optical flow |\n|---|---|---|")
    rows = [("Angle error", "angle_deg", "{:.0f}°", m, min), ("Correlation u", "corr_u", "{:.2f}", m, max),
            ("Correlation v", "corr_v", "{:.2f}", m, max), ("Vector RMSE (m/s)", "rmse_vec", "{:.3f}", m, min),
            ("Skill vs no current", "skill", "{:+.2f}", m, max), ("Across isotherms, correlation", "corr", "{:.2f}", f, max),
            ("Across isotherms, right sign", "same_sign", "{:.0%}", f, max)]
    for label, key, fmt, src_, best in rows:
        a, b = src_["pinn"].get(key), src_["optflow"].get(key)
        win = best(a, b) if a is not None and b is not None else None
        cell = lambda x: "-" if x is None else (f"**{fmt.format(x)}**" if x == win else fmt.format(x))
        print(f"| {label} | {cell(a)} | {cell(b)} |")
    print(f"\nκ learned {D['kappa_m2s']:.1f} m²/s" + (f", true {kappa_true:.0f}" if kappa_true else ""))
    print("->", ", ".join(sorted(p.name for p in out.glob("*.*"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default="docs")
    ap.add_argument("--kappa-true", type=float, default=None)
    a = ap.parse_args()
    src = a.json or next(p for p in ("web/data/run.json", "web/data/demo.json") if pathlib.Path(p).exists())
    main(src, a.out, a.kappa_true)
