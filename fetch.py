import argparse
import pathlib
import requests

ERDDAP = "https://erddap.ifremer.fr/erddap/griddap"
SST_ID = "IFR-L4-SSTfnd-ODYSSEA-GLOB_010"
RADAR_ID = "HF_75c7_5b60_95d8"

# wide SST box (SE Bay of Biscay), radar box = EUSKOOS coverage
SST_BOX = (42.5, 46.5, -6.0, -1.2)
RADAR_BOX = (43.3, 44.6, -3.2, -1.2)


def pull(url, out):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("GET", url)
    r = requests.get(url, stream=True, timeout=600)
    if r.status_code != 200:
        raise SystemExit(f"erddap answered {r.status_code}: {r.text[:300]}")
    with open(out, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print("->", out, f"{out.stat().st_size/1e6:.1f} MB")


def sst(t0, t1, box, out):
    lat0, lat1, lon0, lon1 = box
    q = f"analysed_sst[({t0}T00:00:00Z):1:({t1}T00:00:00Z)][({lat0}):1:({lat1})][({lon0}):1:({lon1})]"
    pull(f"{ERDDAP}/{SST_ID}.nc?{q}", out)


def radar(t0, t1, box, out):
    lat0, lat1, lon0, lon1 = box
    idx = f"[({t0}T00:00:00Z):1:({t1}T23:00:00Z)][(0.0):1:(0.0)][({lat0}):1:({lat1})][({lon0}):1:({lon1})]"
    pull(f"{ERDDAP}/{RADAR_ID}.nc?EWCT{idx},NSCT{idx},GDOP{idx}", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-10-01")
    ap.add_argument("--end", default="2020-10-21")
    ap.add_argument("--what", choices=["sst", "radar", "all"], default="all")
    a = ap.parse_args()
    if a.what in ("sst", "all"):
        sst(a.start, a.end, SST_BOX, f"data/sst_{a.start}_{a.end}.nc")
    if a.what in ("radar", "all"):
        radar(a.start, a.end, RADAR_BOX, f"data/radar_{a.start}_{a.end}.nc")
