import argparse
import pathlib
import sys

import requests

ERDDAP = "https://erddap.ifremer.fr/erddap"

# these move around when Ifremer retires or renames a product.
# `python fetch.py --list sst` asks the catalogue what exists today.
SST_ID = "IFR-L4-SSTfnd-ODYSSEA-GLOB_010"
RADAR_ID = "HF_75c7_5b60_95d8"

SST_BOX = (42.5, 46.5, -6.0, -1.2)      # wide box, SE Bay of Biscay
RADAR_BOX = (43.3, 44.6, -3.2, -1.2)    # EUSKOOS coverage


def catalogue():
    cols = "datasetID,title,minLongitude,maxLongitude,minLatitude,maxLatitude,minTime,maxTime,griddap"
    r = requests.get(f"{ERDDAP}/tabledap/allDatasets.json?{cols}", timeout=120)
    r.raise_for_status()
    t = r.json()["table"]
    return [dict(zip(t["columnNames"], row)) for row in t["rows"]]


def overlaps(d, box):
    if box is None:
        return True
    lat0, lat1, lon0, lon1 = box
    try:
        return not (d["maxLatitude"] < lat0 or d["minLatitude"] > lat1
                    or d["maxLongitude"] < lon0 or d["minLongitude"] > lon1)
    except TypeError:
        return True


def show(words, box=None):
    rows = []
    for d in catalogue():
        if not d.get("griddap"):
            continue
        hay = f"{d['datasetID']} {d['title'] or ''}".lower()
        if not all(w.lower() in hay for w in words):
            continue
        if not overlaps(d, box):
            continue
        rows.append(d)
    if not rows:
        print(f"nothing matching {words} in the griddap catalogue")
        return
    rows.sort(key=lambda d: d["datasetID"])
    for d in rows:
        span = f"{(d['minTime'] or '?')[:10]} -> {(d['maxTime'] or '?')[:10]}"
        bbox = ""
        if d["minLatitude"] is not None:
            bbox = f"  lat {d['minLatitude']:.1f}/{d['maxLatitude']:.1f}  lon {d['minLongitude']:.1f}/{d['maxLongitude']:.1f}"
        print(f"{d['datasetID']}\n    {d['title']}\n    {span}{bbox}\n")
    print(f"{len(rows)} dataset(s). Pass one with --sst-id or --radar-id.")


def pull(url, out):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("GET", url)
    r = requests.get(url, stream=True, timeout=900)
    if r.status_code != 200:
        print(f"erddap answered {r.status_code}: {r.text[:300]}", file=sys.stderr)
        if r.status_code == 404:
            print("\nthat datasetID is gone. what the catalogue has now:\n", file=sys.stderr)
            show(["sst"] if "sst" in str(out) else ["radar"])
        raise SystemExit(1)
    with open(out, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print("->", out, f"{out.stat().st_size/1e6:.1f} MB")


def sst(did, t0, t1, box, out):
    lat0, lat1, lon0, lon1 = box
    q = f"analysed_sst[({t0}T00:00:00Z):1:({t1}T00:00:00Z)][({lat0}):1:({lat1})][({lon0}):1:({lon1})]"
    pull(f"{ERDDAP}/griddap/{did}.nc?{q}", out)


def radar(did, t0, t1, box, out):
    lat0, lat1, lon0, lon1 = box
    idx = f"[({t0}T00:00:00Z):1:({t1}T23:00:00Z)][(0.0):1:(0.0)][({lat0}):1:({lat1})][({lon0}):1:({lon1})]"
    pull(f"{ERDDAP}/griddap/{did}.nc?EWCT{idx},NSCT{idx},GDOP{idx}", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-10-01")
    ap.add_argument("--end", default="2020-10-21")
    ap.add_argument("--what", choices=["sst", "radar", "all"], default="all")
    ap.add_argument("--sst-id", default=SST_ID)
    ap.add_argument("--radar-id", default=RADAR_ID)
    ap.add_argument("--list", nargs="*", metavar="WORD",
                    help="search the griddap catalogue, e.g. --list sst  /  --list radar current")
    ap.add_argument("--near-biscay", action="store_true", help="with --list, keep only datasets covering the box")
    a = ap.parse_args()

    if a.list is not None:
        show(a.list or ["sst"], SST_BOX if a.near_biscay else None)
        raise SystemExit(0)

    if a.what in ("sst", "all"):
        sst(a.sst_id, a.start, a.end, SST_BOX, f"data/sst_{a.start}_{a.end}.nc")
    if a.what in ("radar", "all"):
        radar(a.radar_id, a.start, a.end, RADAR_BOX, f"data/radar_{a.start}_{a.end}.nc")
