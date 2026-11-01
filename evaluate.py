import argparse
import json
import math
import pathlib

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import optflow
from pinn import load


def radar_daily(path, time, lat0, lon0):
    import xarray as xr
    ds = xr.open_dataset(path)
    u = ds["EWCT"].squeeze("depth", drop=True)
    v = ds["NSCT"].squeeze("depth", drop=True)
    days = np.array(time, dtype="datetime64[D]")
    ud, vd = [], []
    for d in days:
        sel = (u["time"].values >= d) & (u["time"].values < d + np.timedelta64(1, "D"))
        ud.append(u.values[sel].mean(0) if sel.any() else np.full(u.shape[1:], np.nan))
        vd.append(v.values[sel].mean(0) if sel.any() else np.full(v.shape[1:], np.nan))
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
    return dict(
        n=int(ok.sum()),
        rmse_vec=float(np.sqrt(np.mean((u - ur) ** 2 + (v - vr) ** 2))),
        rmse_speed=float(np.sqrt(np.mean((sp - spr) ** 2))),
        corr_u=float(np.corrcoef(u, ur)[0, 1]),
        corr_v=float(np.corrcoef(v, vr)[0, 1]),
        angle_deg=float(np.degrees(np.arccos(cosang[w])).mean()) if w.any() else float("nan"),
        speed_bias=float(np.mean(sp - spr)),
        ref_speed=float(spr.mean()),
    )


def main(run, data, radar=None, out="web/data/run.json"):
    run = pathlib.Path(run)
    F = np.load(run / "fields.npz")
    d = load(data)
    x, y, t, T = d["x"], d["y"], d["t"], d["T"]
    u, v = F["u"], F["v"]
    uo, vo = optflow.currents(T, x, y, t)

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
    hist = json.loads((run / "history.json").read_text())

    def pack(a, nd=3):
        a = np.asarray(a, dtype=float)
        return np.where(np.isfinite(a), np.round(a, nd), None).tolist()

    payload = dict(
        source=d["src"], reference=ref_name, kappa_m2s=float(F["kappa_m2s"]),
        x=pack(x, 1), y=pack(y, 1), t=pack(t, 2),
        T=pack(T, 2), T_fit=pack(F["T_fit"], 2),
        u=pack(u), v=pack(v), u_flow=pack(uo), v_flow=pack(vo), u_ref=pack(ur), v_ref=pack(vr),
        metrics=dict(pinn=m_pinn, optflow=m_flow), history=hist,
    )
    if d["src"] == "sst":
        payload["lat"] = pack(d["lat"], 3)
        payload["lon"] = pack(d["lon"], 3)
        payload["dates"] = [str(np.datetime64(s, "D")) for s in d["time"]]
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(payload, allow_nan=False))
    print(json.dumps(dict(pinn=m_pinn, optflow=m_flow), indent=1))
    print("->", out)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/synth")
    ap.add_argument("--data", default="data/synth.npz")
    ap.add_argument("--radar", default=None)
    ap.add_argument("--out", default="web/data/run.json")
    a = ap.parse_args()
    main(a.run, a.data, a.radar, a.out)
