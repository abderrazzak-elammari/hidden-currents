"""Is the radar current visible in the SST at all? No network involved.

If the radar currents are what moves the temperature, then day to day
    dT/dt  ~  -(u dT/dx + v dT/dy)  + something smooth (heating, cooling)
so the observed tendency should correlate with the tendency the radar predicts.
If it does not, no method that reads these SST maps can recover these currents.
"""
import argparse
import warnings

import numpy as np

from evaluate import radar_daily, regrid
from pinn import load

KM_PER_DAY = 86.4


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def smooth_days(F, w):
    if w <= 1:
        return F
    out = np.full_like(F, np.nan)
    h = w // 2
    for k in range(len(F)):
        out[k] = np.nanmean(F[max(0, k - h):k + h + 1], 0)
    return out


def main(data, radar=None, lowpass=(1, 3, 5)):
    warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN land columns and edge days
    d = load(data)
    x, y, t, T = d["x"], d["y"], d["t"], d["T"]
    if d["src"] == "synth":
        ur = np.repeat(d["u_ref"][None], len(t), 0)
        vr = np.repeat(d["v_ref"][None], len(t), 0)
    else:
        xr_, yr_, U, V = radar_daily(radar, d["time"], d["lat"].mean(), d["lon"].mean())
        ur, vr = regrid(xr_, yr_, U, x, y), regrid(xr_, yr_, V, x, y)

    gy, gx = np.gradient(T, y, x, axis=(1, 2))            # °C / km
    dt = np.full_like(T, np.nan)
    dt[1:-1] = (T[2:] - T[:-2]) / (t[2:] - t[:-2])[:, None, None]   # °C / day, centred
    dt -= np.nanmean(dt, axis=(1, 2), keepdims=True)       # remove the basin-wide heating/cooling
    g = np.hypot(gx, gy)
    fronts = g >= np.nanpercentile(g, 70)

    print(f"\n  {d['src']}: {len(t)} days, observed tendency std {np.nanstd(dt):.3f} °C/day")
    print(f"  {'radar mean over':<22}{'corr all':>10}{'corr fronts':>13}{'advection std':>16}{'n':>8}")
    fr = []
    for w in lowpass:
        u, v = smooth_days(ur, w), smooth_days(vr, w)
        adv = -(u * KM_PER_DAY * gx + v * KM_PER_DAY * gy)
        adv -= np.nanmean(adv, axis=(1, 2), keepdims=True)
        c_all, n = corr(dt, adv)
        c_fr, _ = corr(np.where(fronts, dt, np.nan), np.where(fronts, adv, np.nan))
        fr.append(c_fr)
        print(f"  {str(w) + ' day(s)':<22}{c_all:>+10.2f}{c_fr:>+13.2f}{np.nanstd(adv):>13.3f} °C/d{n:>8}")

    fr = [c for c in fr if np.isfinite(c)] or [0.0]
    best, worst = max(fr), min(fr)
    if worst < -0.3 and best < 0.1:
        verdict = "strongly ANTI-correlated: that smells of a sign or orientation error in the data, not physics."
    elif best < 0.1:
        verdict = "no: the SST changes do not follow the radar currents. Change the data, not the model."
    elif best < 0.3:
        verdict = "weak: some signal, buried. A model can only do a little better than chance."
    else:
        verdict = "yes: the information is in the SST. If the PINN misses it, the PINN is the problem."
    print(f"\n  is the radar current visible in this SST?  {verdict}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sst_2020-10-01_2020-10-21.nc")
    ap.add_argument("--radar", default="data/radar_2020-10-01_2020-10-21.nc")
    a = ap.parse_args()
    main(a.data, None if a.data.endswith(".npz") else a.radar)
