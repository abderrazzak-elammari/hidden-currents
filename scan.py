"""Which periods have SST that actually moves with the radar currents?

For each 21-day window: fetch (skipped if the files are there), then the same
model-free test as check_signal.py. Ten minutes of downloads, no training.
"""
import argparse
import pathlib
import subprocess
import sys
import warnings

import numpy as np

from check_signal import KM_PER_DAY, corr, smooth_days
from evaluate import radar_daily, regrid
from pinn import load

WINDOWS = [
    ("2020-10-01", "2020-10-21"),
    ("2021-01-15", "2021-02-04"), ("2021-04-15", "2021-05-05"),
    ("2021-07-15", "2021-08-04"), ("2021-10-15", "2021-11-04"),
    ("2022-01-15", "2022-02-04"), ("2022-04-15", "2022-05-05"),
    ("2022-07-15", "2022-08-04"), ("2022-10-15", "2022-11-04"),
]


def signal(spath, rpath):
    d = load(spath)
    x, y, t, T = d["x"], d["y"], d["t"], d["T"]
    xr_, yr_, U, V = radar_daily(rpath, d["time"], d["lat"].mean(), d["lon"].mean())
    ur, vr = regrid(xr_, yr_, U, x, y), regrid(xr_, yr_, V, x, y)
    gy, gx = np.gradient(T, y, x, axis=(1, 2))
    dt = np.full_like(T, np.nan)
    dt[1:-1] = (T[2:] - T[:-2]) / (t[2:] - t[:-2])[:, None, None]
    dt -= np.nanmean(dt, axis=(1, 2), keepdims=True)
    g = np.hypot(gx, gy)
    fronts = g >= np.nanpercentile(g, 70)
    out = dict(obs=float(np.nanstd(dt)), grad=float(np.nanmedian(g[fronts])) * 100)
    for w in (1, 3, 5):
        u, v = smooth_days(ur, w), smooth_days(vr, w)
        adv = -(u * KM_PER_DAY * gx + v * KM_PER_DAY * gy)
        adv -= np.nanmean(adv, axis=(1, 2), keepdims=True)
        out[w], out["n"] = corr(np.where(fronts, dt, np.nan), np.where(fronts, adv, np.nan))
        out["adv"] = float(np.nanstd(adv))
    return out


def main(windows):
    warnings.simplefilter("ignore", RuntimeWarning)
    rows = []
    for s, e in windows:
        spath = pathlib.Path(f"data/sst_{s}_{e}.nc")
        rpath = pathlib.Path(f"data/radar_{s}_{e}.nc")
        if not (spath.exists() and rpath.exists()):
            print(f"fetching {s} -> {e} ...", flush=True)
            p = subprocess.run([sys.executable, "fetch.py", "--start", s, "--end", e], capture_output=True, text=True)
            if p.returncode != 0:
                last = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["?"]
                print(f"   failed: {last[0][:120]}")
                rows.append((s, e, None))
                continue
        try:
            rows.append((s, e, signal(spath, rpath)))
        except Exception as ex:                      # one bad window should not kill the scan
            print(f"   {s}: {type(ex).__name__}: {ex}")
            rows.append((s, e, None))

    print(f"\n  {'window':<26}{'corr at fronts, radar over':^27}{'obs':>7}{'adv':>7}{'front':>10}{'n':>7}")
    print(f"  {'':<26}{'1 day':>9}{'3 days':>9}{'5 days':>9}{'°C/d':>7}{'°C/d':>7}{'°C/100km':>10}")
    for s, e, r in rows:
        if r is None:
            print(f"  {s} -> {e}    no data")
            continue
        c = lambda k: f"{r[k]:+.2f}" if np.isfinite(r[k]) else "  -"
        print(f"  {s} -> {e}{c(1):>9}{c(3):>9}{c(5):>9}{r['obs']:>7.2f}{r['adv']:>7.2f}{r['grad']:>10.1f}{r['n']:>7}")

    ok = [(max(np.nan_to_num(r[k], nan=-1) for k in (1, 3, 5)), s, e) for s, e, r in rows if r]
    if not ok:
        print("\n  nothing could be tested.")
        return
    best, s, e = max(ok)
    print(f"\n  best window: {s} -> {e}, corr {best:+.2f}")
    if best >= 0.3:
        print("  the signal is there. Train on it:")
        print(f"    python pinn.py --data data/sst_{s}_{e}.nc --out runs/{s} --steps 4000 --batch 2048")
        print(f"    python evaluate.py --run runs/{s} --data data/sst_{s}_{e}.nc --radar data/radar_{s}_{e}.nc")
    elif best >= 0.1:
        print("  weak signal. A PINN run there may beat chance, not by much.")
    else:
        print("  no season works: MUR, being a gap-filled analysis, does not carry these currents.")
        print("  next step is real satellite passes (L3 SST), not another model.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", help="comma list of start:end, e.g. 2021-07-15:2021-08-04,2022-01-15:2022-02-04")
    a = ap.parse_args()
    w = [tuple(x.split(":")) for x in a.windows.split(",")] if a.windows else WINDOWS
    main(w)
