import argparse
import datetime as dt
import os
import pathlib

import requests

# SST: MUR 0.01°, NOAA CoastWatch. Ifremer dropped ODYSSEA from its ERDDAP in 2026.
SST_SERVER = "https://coastwatch.pfeg.noaa.gov/erddap"
SST_ID = "jplMURSST41"
SST_STRIDE = 5          # 0.05°, about the radar grid

# EUSKOOS HF radar totals, SE Bay of Biscay, hourly since 2018-12, EMODnet Physics.
# Ifremer mirrors it too but under hashed IDs that change on every re-harvest.
RADAR_SERVER = "https://erddap.emodnet-physics.eu/erddap"
RADAR_ID = "EUHFR_NRTcurrent_HFR-EUSKOOS-Total"

PAD = 0.4               # SST box = radar box + this, degrees
MAX_GDOP = 2.0          # usual EU HFR cut, poorer geometry is noise
MIN_HOURS = 12          # a daily mean from 3 hours is mostly tide


def info(server, did):
    r = requests.get(f"{server}/info/{did}/index.json", timeout=120)
    if r.status_code != 200:
        raise SystemExit(f"{did} @ {server}: {r.status_code}\n{r.text[:300]}")
    meta = dict(server=server, id=did, attrs={}, vars={}, ranges={})
    for kind, name, attr, _t, value in r.json()["table"]["rows"]:
        if kind == "attribute" and name == "NC_GLOBAL":
            meta["attrs"][attr] = value
        elif kind == "variable":
            meta["vars"][name] = [s.strip() for s in value.split(",")]
        elif kind == "attribute" and attr == "actual_range":
            try:
                lo, hi = (float(v) for v in str(value).split(","))
                meta["ranges"][name] = (min(lo, hi), max(lo, hi))
            except ValueError:
                pass
    return meta


def day(sec):
    return dt.datetime.fromtimestamp(sec, dt.timezone.utc).date()


def describe(meta):
    print(f"\n{meta['id']}  @ {meta['server']}")
    print(f"  {meta['attrs'].get('title', '')}")
    r = meta["ranges"]
    if "time" in r:
        print(f"  time  {day(r['time'][0])} -> {day(r['time'][1])}")
    if "latitude" in r and "longitude" in r:
        print(f"  lat   {r['latitude'][0]:.2f} / {r['latitude'][1]:.2f}    lon {r['longitude'][0]:.2f} / {r['longitude'][1]:.2f}")
    for v, dims in meta["vars"].items():
        print(f"  {v:<14}({', '.join(dims)})")


def check_dates(meta, t0, t1):
    if "time" not in meta["ranges"]:
        return
    lo, hi = (day(s) for s in meta["ranges"]["time"])
    if dt.date.fromisoformat(t0) < lo or dt.date.fromisoformat(t1) > hi:
        raise SystemExit(f"{meta['id']} covers {lo} -> {hi}. Pick --start/--end inside that.")


def num(v):
    return f"{v:.4f}".rstrip("0").rstrip(".")


def selector(meta, var, t0, t1, box, stride, flip):
    lat0, lat1, lon0, lon1 = (num(v) for v in box)
    if flip:
        lat0, lat1 = lat1, lat0
    out = var
    for dim in meta["vars"][var]:
        if dim == "time":
            out += f"[({t0}):1:({t1})]"
        elif dim == "latitude":
            out += f"[({lat0}):{stride}:({lat1})]"
        elif dim == "longitude":
            out += f"[({lon0}):{stride}:({lon1})]"
        else:
            out += "[0:1:0]"
    return out


def grab(meta, varlist, t0, t1, box, stride, out):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for flip in (False, True):
        q = ",".join(selector(meta, v, t0, t1, box, stride, flip) for v in varlist)
        url = f"{meta['server']}/griddap/{meta['id']}.nc?{q}"
        print("GET", url)
        r = requests.get(url, stream=True, timeout=900)
        if r.status_code == 200:
            with open(out, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            print("->", out, f"{out.stat().st_size/1e6:.1f} MB raw")
            return
        body = r.text[:400]
        if not flip and "greater than" in body.lower():
            print("  latitude is stored north->south, retrying flipped")
            continue
        raise SystemExit(f"erddap answered {r.status_code}:\n{body}")


def save(ds, path):
    for v in ds.variables.values():
        v.encoding = {}
    tmp = pathlib.Path(str(path) + ".tmp")
    ds.to_netcdf(tmp)
    os.replace(tmp, path)


def clean_sst(path, var):
    import xarray as xr
    with xr.open_dataset(path) as ds:
        ds = ds[[var]].load()
    ds = ds.rename({var: "analysed_sst"}) if var != "analysed_sst" else ds
    ds = ds.sortby("latitude").sortby("longitude")
    ds["analysed_sst"] = ds["analysed_sst"].astype("float32")
    save(ds, path)
    T = ds["analysed_sst"]
    sea = int(T.notnull().all("time").sum())
    print(f"   sst  {T.sizes['time']} days, {T.sizes['latitude']} x {T.sizes['longitude']} grid, {sea} sea cells")


def daily_radar(path):
    import numpy as np
    import xarray as xr
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    ds = ds.sortby("latitude").sortby("longitude")
    u, v = ds["EWCT"], ds["NSCT"]
    ok = u.notnull() & v.notnull()
    if "GDOP" in ds:
        ok &= ds["GDOP"] <= MAX_GDOP
    n = ok.resample(time="1D").sum()
    ud = u.where(ok).resample(time="1D").mean()
    vd = v.where(ok).resample(time="1D").mean()
    keep = n >= MIN_HOURS
    out = xr.Dataset({"EWCT": ud.where(keep).astype("float32"), "NSCT": vd.where(keep).astype("float32")}, attrs=ds.attrs)
    out = out.assign_coords(time=out["time"] + np.timedelta64(12, "h"))
    save(out, path)
    cells = int(keep.sum())
    print(f"   radar {out.sizes['time']} days, {cells} cell-days kept (GDOP <= {MAX_GDOP}, >= {MIN_HOURS} h)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-10-01")
    ap.add_argument("--end", default="2020-10-21")
    ap.add_argument("--sst-id", default=SST_ID)
    ap.add_argument("--radar-id", default=RADAR_ID)
    ap.add_argument("--info", action="store_true", help="show what both datasets hold, download nothing")
    a = ap.parse_args()

    rad = info(RADAR_SERVER, a.radar_id)
    sst = info(SST_SERVER, a.sst_id)
    if a.info:
        describe(rad)
        describe(sst)
        raise SystemExit(0)

    check_dates(rad, a.start, a.end)
    check_dates(sst, a.start, a.end)
    (y0, y1), (x0, x1) = rad["ranges"]["latitude"], rad["ranges"]["longitude"]
    rbox = (y0, y1, x0, x1)
    sbox = (y0 - PAD, y1 + PAD, x0 - PAD, x1 + PAD)

    rvars = [v for v in ("EWCT", "NSCT", "GDOP") if v in rad["vars"]]
    svar = "analysed_sst" if "analysed_sst" in sst["vars"] else next(iter(sst["vars"]))
    rpath = f"data/radar_{a.start}_{a.end}.nc"
    spath = f"data/sst_{a.start}_{a.end}.nc"

    grab(rad, rvars, f"{a.start}T00:00:00Z", f"{a.end}T23:00:00Z", rbox, 1, rpath)
    daily_radar(rpath)
    # MUR is stamped 09:00; T00:00 snaps to the same day at both ends
    grab(sst, [svar], f"{a.start}T00:00:00Z", f"{a.end}T00:00:00Z", sbox, SST_STRIDE, spath)
    clean_sst(spath, svar)
    print("\nready. next:")
    print(f"  python pinn.py --data {spath} --out runs/oct2020 --steps 4000 --batch 2048")
    print(f"  python evaluate.py --run runs/oct2020 --data {spath} --radar {rpath}")
