import argparse
import json
import math
import pathlib

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import optflow
from pinn import flat3, load, tidy


def radar_daily(path, time, lat0, lon0, min_hours=12, max_gdop=2.0):
    import xarray as xr
    ds = tidy(xr.open_dataset(path))
    u, v = flat3(ds["EWCT"]), flat3(ds["NSCT"])
    g = flat3(ds["GDOP"]) if "GDOP" in ds else None
    tt = u["time"].values
    if len(tt) > 1 and np.median(np.diff(tt)) >= np.timedelta64(20, "h"):
        min_hours = 1                         # already daily, fetch.py did the averaging
    U, V = u.values, v.values
    bad = ~(np.isfinite(U) & np.isfinite(V))
    if g is not None:
        bad |= ~(g.values <= max_gdop)        # radar geometry too poor, standard EU HFR cut
    U, V = np.where(bad, 0, U), np.where(bad, 0, V)
    ud, vd = [], []
    for d in np.array(time, dtype="datetime64[D]"):
        sel = (tt >= d) & (tt < d + np.timedelta64(1, "D"))
        n = (~bad[sel]).sum(0)
        mu = U[sel].sum(0) / np.maximum(n, 1)
        mv = V[sel].sum(0) / np.maximum(n, 1)
        mu[n < min_hours] = np.nan            # a daily mean from 3 hours is mostly tide
        mv[n < min_hours] = np.nan
        ud.append(mu)
        vd.append(mv)
    lat = ds["latitude"].values.astype(float)
    lon = ds["longitude"].values.astype(float)
    x = (lon - lon0) * 111.32 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111.32
    return x, y, np.stack(ud).astype(np.float32), np.stack(vd).astype(np.float32)


def regrid(xs, ys, F, xt, yt):
    Xt, Yt = np.meshgrid(xt, yt)
    out = []
    for f in F:
        itp = RegularGridInterpolator((ys, xs), np.nan_to_num(f), bounds_error=False, fill_value=np.nan)
        g = itp(np.stack([Yt.ravel(), Xt.ravel()], -1)).reshape(Xt.shape)
        msk = RegularGridInterpolator((ys, xs), np.isfinite(f).astype(float), bounds_error=False, fill_value=0)
        g[msk(np.stack([Yt.ravel(), Xt.ravel()], -1)).reshape(Xt.shape) < 0.99] = np.nan
        out.append(g)
    return np.stack(out)


def score(u, v, ur, vr):
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(ur) & np.isfinite(vr)
    if ok.sum() < 10:
        return dict(n=int(ok.sum()))
    u, v, ur, vr = u[ok], v[ok], ur[ok], vr[ok]
    sp, spr = np.hypot(u, v), np.hypot(ur, vr)
    dot = u * ur + v * vr
    cosang = np.clip(dot / (sp * spr + 1e-9), -1, 1)
    w = spr > 0.03
    m = dict(
        n=int(ok.sum()),
        rmse_vec=float(np.sqrt(np.mean((u - ur) ** 2 + (v - vr) ** 2))),
        rmse_speed=float(np.sqrt(np.mean((sp - spr) ** 2))),
        corr_u=float(np.corrcoef(u, ur)[0, 1]),
        corr_v=float(np.corrcoef(v, vr)[0, 1]),
        angle_deg=float(np.degrees(np.arccos(cosang[w])).mean()) if w.any() else None,
        speed_bias=float(np.mean(sp - spr)),
        ref_speed=float(spr.mean()),
        rmse_zero=float(np.sqrt(np.mean(ur ** 2 + vr ** 2))),
    )
    m["skill"] = 1 - m["rmse_vec"] ** 2 / m["rmse_zero"] ** 2      # > 0: beats "no current at all"
    return clean(m)


def clean(m):
    return {k: (None if isinstance(x, float) and not np.isfinite(x) else x) for k, x in m.items()}


def across(u, v, ur, vr, T, x, y, q=70):
    # a tracer only shows the flow that crosses its isolines; along them it is blind
    gy, gx = np.gradient(T, y, x, axis=(1, 2))
    g = np.hypot(gx, gy)
    ok = np.isfinite(g) & np.isfinite(u) & np.isfinite(v) & np.isfinite(ur) & np.isfinite(vr) & (g > 0)
    if ok.sum() < 50:
        return dict(n=int(ok.sum()))
    ok &= g >= np.percentile(g[ok], q)
    nx, ny = gx / np.where(g > 0, g, 1), gy / np.where(g > 0, g, 1)
    a, ar = (u * nx + v * ny)[ok], (ur * nx + vr * ny)[ok]
    b, br = (v * nx - u * ny)[ok], (vr * nx - ur * ny)[ok]
    strong = np.abs(ar) > 0.02
    return clean(dict(
        n=int(ok.sum()),
        corr=float(np.corrcoef(a, ar)[0, 1]),
        rmse=float(np.sqrt(np.mean((a - ar) ** 2))),
        rmse_zero=float(np.sqrt(np.mean(ar ** 2))),
        same_sign=float(np.mean(np.sign(a[strong]) == np.sign(ar[strong]))) if strong.any() else float("nan"),
        corr_along=float(np.corrcoef(b, br)[0, 1]),
    ))


def table(m, f):
    P, O = m["pinn"], m["optflow"]
    fp, fo = f["pinn"], f["optflow"]
    def r(label, a, b, fmt):
        cell = lambda x: format(x, fmt) if x is not None else "-"
        print(f"  {label:<28}{cell(a):>10}{cell(b):>10}")
    print(f"\n  {'':<28}{'PINN':>10}{'optflow':>10}")
    r("vector RMSE (m/s)", P.get("rmse_vec"), O.get("rmse_vec"), ".3f")
    r("  'no current' would give", P.get("rmse_zero"), O.get("rmse_zero"), ".3f")
    r("skill vs no current", P.get("skill"), O.get("skill"), "+.2f")
    r("angle error (deg, 90 = random)", P.get("angle_deg"), O.get("angle_deg"), ".1f")
    r("corr u", P.get("corr_u"), O.get("corr_u"), "+.2f")
    r("corr v", P.get("corr_v"), O.get("corr_v"), "+.2f")
    r("speed bias (m/s)", P.get("speed_bias"), O.get("speed_bias"), "+.3f")
    print(f"  across isotherms, strongest 30% gradients, n={fp.get('n')}")
    r("  corr", fp.get("corr"), fo.get("corr"), "+.2f")
    r("  RMSE (m/s)", fp.get("rmse"), fo.get("rmse"), ".3f")
    r("  'no current' would give", fp.get("rmse_zero"), fo.get("rmse_zero"), ".3f")
    r("  same sign (50% = random)", fp.get("same_sign"), fo.get("same_sign"), ".0%")
    r("  corr along isotherms", fp.get("corr_along"), fo.get("corr_along"), "+.2f")
    print(f"  {P.get('n')} points, reference mean speed {P.get('ref_speed') or 0:.3f} m/s\n")


def main(run, data, radar=None, out="web/data/run.json", alpha=optflow.ALPHA):
    run = pathlib.Path(run)
    F = np.load(run / "fields.npz")
    d = load(data)
    x, y, t, T = d["x"], d["y"], d["t"], d["T"]
    u, v = F["u"], F["v"]
    uo, vo = optflow.currents(T, x, y, t, alpha=alpha)

    if d["src"] == "synth":
        ur = np.repeat(d["u_ref"][None], len(t), 0)
        vr = np.repeat(d["v_ref"][None], len(t), 0)
        ref_name = "synthetic truth"
    else:
        xr_, yr_, ur_, vr_ = radar_daily(radar, d["time"], d["lat"].mean(), d["lon"].mean())
        ur = regrid(xr_, yr_, ur_, x, y)
        vr = regrid(xr_, yr_, vr_, x, y)
        ref_name = "EUSKOOS HF radar (daily mean)"

    m_pinn = score(u, v, ur, vr)
    m_flow = score(uo, vo, ur, vr)
    front = dict(pinn=across(u, v, ur, vr, T, x, y), optflow=across(uo, vo, ur, vr, T, x, y))
    if m_pinn["n"] < 500:
        print(f"warning: only {m_pinn['n']} points overlap the reference, the scores mean little")
    hist = json.loads((run / "history.json").read_text())

    def pack(a, nd=3):
        a = np.asarray(a, dtype=float)
        return np.where(np.isfinite(a), np.round(a, nd), None).tolist()

    payload = dict(
        source=d["src"], reference=ref_name, kappa_m2s=float(F["kappa_m2s"]),
        x=pack(x, 1), y=pack(y, 1), t=pack(t, 2),
        T=pack(T, 2), T_fit=pack(F["T_fit"], 2),
        u=pack(u), v=pack(v), u_flow=pack(uo), v_flow=pack(vo), u_ref=pack(ur), v_ref=pack(vr),
        metrics=dict(pinn=m_pinn, optflow=m_flow, across=front), history=hist,
    )
    if d["src"] == "sst":
        payload["lat"] = pack(d["lat"], 3)
        payload["lon"] = pack(d["lon"], 3)
        payload["dates"] = [str(np.datetime64(s, "D")) for s in d["time"]]
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(payload, allow_nan=False))
    table(payload["metrics"], front)
    print("->", out)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/synth")
    ap.add_argument("--data", default="data/synth.npz")
    ap.add_argument("--radar", default=None)
    ap.add_argument("--out", default="web/data/run.json")
    ap.add_argument("--alpha", type=float, default=optflow.ALPHA, help="Horn–Schunck smoothing, pixel units")
    a = ap.parse_args()
    main(a.run, a.data, a.radar, a.out, a.alpha)
