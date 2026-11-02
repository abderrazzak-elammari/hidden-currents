import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from evaluate import main as evaluate
from pinn import load, train


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("real")
    rng = np.random.default_rng(0)

    # shaped like jplMURSST41 after stride: daily at 09:00, ascending lat, °C, land = NaN
    t = pd.date_range("2020-10-01T09:00", periods=5, freq="D")
    lat = np.round(np.arange(43.2, 44.8, 0.05), 2)
    lon = np.round(np.arange(-3.4, -1.0, 0.05), 2)
    LO, LA = np.meshgrid(lon, lat)
    T = np.stack([16 + (LA - 43) + 0.5 * np.sin(3 * (LO + 0.03 * k)) for k in range(5)]).astype(np.float32)
    T[:, (LA < 43.4) | (LO > -1.45)] = np.nan
    xr.Dataset({"analysed_sst": (("time", "latitude", "longitude"), T)},
               coords=dict(time=t, latitude=lat, longitude=lon)).to_netcdf(d / "sst.nc")

    # shaped like EUHFR_NRTcurrent_HFR-EUSKOOS-Total: hourly with holes, depth=[0], lat north->south
    th = pd.date_range("2020-10-01T00:00", "2020-10-05T23:00", freq="h")
    th = th[rng.random(len(th)) > 0.1]
    rlat = np.linspace(44.58, 43.32, 29)
    rlon = np.linspace(-3.196593, -1.203407, 33)
    RLO, RLA = np.meshgrid(rlon, rlat)
    n = len(th)
    E = np.broadcast_to(0.1 * np.sin(2 * RLO), (n, 29, 33)).astype(np.float32).copy()
    N = np.broadcast_to(0.1 * np.cos(2 * RLA), (n, 29, 33)).astype(np.float32).copy()
    G = np.full((n, 29, 33), 1.0, np.float32)
    G[:, :, :4] = 5.0                                   # bad geometry at the western edge
    for a in (E, N):
        a[:, (RLA < 43.45) | (RLO > -1.5)] = np.nan
    xr.Dataset({k: (("time", "depth", "latitude", "longitude"), a[:, None]) for k, a in (("EWCT", E), ("NSCT", N), ("GDOP", G))},
               coords=dict(time=th, depth=[0.0], latitude=rlat, longitude=rlon)).to_netcdf(d / "radar.nc")
    return d


def test_load_sorts_and_masks(files):
    d = load(files / "sst.nc")
    assert d["src"] == "sst"
    assert d["T"].shape == (5, len(d["y"]), len(d["x"]))
    assert np.all(np.diff(d["y"]) > 0)
    assert np.isnan(d["T"]).any()
    assert d["t"][0] == 0 and d["t"][-1] == pytest.approx(4)


def test_against_radar(files, tmp_path):
    d = load(files / "sst.nc")
    run = tmp_path / "run"
    train(d, run, steps=40, batch=512, log_every=20)
    p = evaluate(run, files / "sst.nc", radar=files / "radar.nc", out=tmp_path / "run.json")
    assert p["dates"][0] == "2020-10-01" and len(p["dates"]) == 5
    assert p["metrics"]["pinn"]["n"] > 1000
    json.loads((tmp_path / "run.json").read_text())       # allow_nan=False held
    ur = np.array(p["u_ref"], dtype=float)                  # None -> nan
    assert np.isnan(ur[:, :, np.array(p["lon"]) < -3.0]).all()   # GDOP 5 dropped, not averaged in
