import argparse
import datetime as dt
import pathlib
import sys

import requests

# SST: MUR 0.01°, NOAA CoastWatch. Ifremer dropped ODYSSEA from its ERDDAP in 2026.
SST_SERVER = "https://coastwatch.pfeg.noaa.gov/erddap"
SST_ID = "jplMURSST41"
SST_STRIDE = 5          # 0.05°, about the radar grid

# EUSKOOS HF radar totals, SE Bay of Biscay, hourly since 2018-12, EMODnet Physics.
# Ifremer mirrors it too but under hashed IDs (HF_ac49_84ad_3eb6 in Nov 2026) that change
# on every re-harvest. `--list euskoos` searches the radar server.
RADAR_SERVER = "https://erddap.emodnet-physics.eu/erddap"
RADAR_ID = "EUHFR_NRTcurrent_HFR-EUSKOOS-Total"

PAD = 0.4               # SST box = radar box + this, degrees


def role(dim):
    d = dim.lower()
    if d == "time":
        return "t"
    if d in ("latitude", "lat"):
        return "y"
    if d in ("longitude", "lon"):
        return "x"
    return None


def info(server, did):
    r = requests.get(f"{server}/info/{did}/index.json", timeout=120)
    if r.status_code != 200:
        raise SystemExit(f"{did} @ {server}: {r.status_code}\n{r.text[:300]}\n\ntry: python fetch.py --list radar")
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


def axis(meta, which):
    for name, rng in meta["ranges"].items():
        if role(name) == which:
            return name, rng
    return None, None


def day(sec):
    return dt.datetime.fromtimestamp(sec, dt.timezone.utc).date()


def describe(meta):
    a = meta["attrs"]
    print(f"\n{meta['id']}  @ {meta['server']}")
    print(f"  {a.get('title', '')}")
    for k in ("platform_code", "site_code", "network", "area", "institution"):
        if k in a:
            print(f"  {k}: {a[k]}")
    _, t = axis(meta, "t")
    _, y = axis(meta, "y")
    _, x = axis(meta, "x")
    if t:
        print(f"  time  {day(t[0])} -> {day(t[1])}")
    if y and x:
        print(f"  lat   {y[0]:.2f} / {y[1]:.2f}    lon {x[0]:.2f} / {x[1]:.2f}")
    for v, dims in meta["vars"].items():
        print(f"  {v:<14}({', '.join(dims)})")


def check_dates(meta, t0, t1):
    _, t = axis(meta, "t")
    if not t:
        return
    lo, hi = day(t[0]), day(t[1])
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
        r = role(dim)
        if r == "t":
            out += f"[({t0}):1:({t1})]"
        elif r == "y":
            out += f"[({lat0}):{stride}:({lat1})]"
        elif r == "x":
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
            print("->", out, f"{out.stat().st_size/1e6:.1f} MB")
            return
        body = r.text[:400]
        if not flip and "greater than" in body.lower():
            print("  latitude is stored north->south, retrying flipped")
            continue
        raise SystemExit(f"erddap answered {r.status_code}:\n{body}")


def catalogue(server):
    cols = "datasetID,title,minLongitude,maxLongitude,minLatitude,maxLatitude,minTime,maxTime,griddap"
    r = requests.get(f"{server}/tabledap/allDatasets.json?{cols}", timeout=120)
    r.raise_for_status()
    t = r.json()["table"]
    return [dict(zip(t["columnNames"], row)) for row in t["rows"]]


def show(server, words):
    hits = []
    for d in catalogue(server):
        if not d.get("griddap"):
            continue
        hay = f"{d['datasetID']} {d['title'] or ''}".lower()
        if all(w.lower() in hay for w in words):
            hits.append(d)
    if not hits:
        print(f"nothing matching {words} on {server}")
        return
    for d in sorted(hits, key=lambda d: d["datasetID"]):
        span = f"{(d['minTime'] or '?')[:10]} -> {(d['maxTime'] or '?')[:10]}"
        bbox = ""
        if d["minLatitude"] not in (None, ""):
            bbox = f"  lat {d['minLatitude']:.1f}/{d['maxLatitude']:.1f}  lon {d['minLongitude']:.1f}/{d['maxLongitude']:.1f}"
        print(f"{d['datasetID']}\n    {d['title']}\n    {span}{bbox}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-10-01")
    ap.add_argument("--end", default="2020-10-21")
    ap.add_argument("--sst-server", default=SST_SERVER)
    ap.add_argument("--sst-id", default=SST_ID)
    ap.add_argument("--radar-server", default=RADAR_SERVER)
    ap.add_argument("--radar-id", default=RADAR_ID)
    ap.add_argument("--info", action="store_true", help="print what both datasets contain, download nothing")
    ap.add_argument("--list", nargs="*", metavar="WORD", help="search the radar server catalogue")
    a = ap.parse_args()

    if a.list is not None:
        show(a.radar_server, a.list or ["hf"])
        raise SystemExit(0)

    rad = info(a.radar_server, a.radar_id)
    sst = info(a.sst_server, a.sst_id)
    if a.info:
        describe(rad)
        describe(sst)
        raise SystemExit(0)

    check_dates(rad, a.start, a.end)
    check_dates(sst, a.start, a.end)

    _, y = axis(rad, "y")
    _, x = axis(rad, "x")
    if not (y and x):
        raise SystemExit(f"{rad['id']}: no latitude/longitude range in its metadata, run --info")
    rbox = (y[0], y[1], x[0], x[1])
    sbox = (y[0] - PAD, y[1] + PAD, x[0] - PAD, x[1] + PAD)
    print(f"radar box lat {rbox[0]:.2f}/{rbox[1]:.2f} lon {rbox[2]:.2f}/{rbox[3]:.2f}   sst box +{PAD}°")

    rvars = [v for v in ("EWCT", "NSCT", "GDOP") if v in rad["vars"]]
    if not {"EWCT", "NSCT"} <= set(rvars):
        raise SystemExit(f"{rad['id']} has no EWCT/NSCT, it is not a total-current radar product. Variables: {list(rad['vars'])}")
    svar = "analysed_sst" if "analysed_sst" in sst["vars"] else next(iter(sst["vars"]))

    grab(rad, rvars, f"{a.start}T00:00:00Z", f"{a.end}T23:00:00Z", rbox, 1, f"data/radar_{a.start}_{a.end}.nc")
    # MUR is stamped 09:00; T00:00 snaps to the same day at both ends, 23:59 would pull the next one
    grab(sst, [svar], f"{a.start}T00:00:00Z", f"{a.end}T00:00:00Z", sbox, SST_STRIDE, f"data/sst_{a.start}_{a.end}.nc")
