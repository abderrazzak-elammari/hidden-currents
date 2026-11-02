import json
import numpy as np
import pytest

import synth
import optflow
from pinn import train
from evaluate import main as evaluate


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    d = synth.run(days=6, noise=0.02)
    p = tmp_path_factory.mktemp("d") / "synth.npz"
    np.savez(p, **d)
    return p, d


def test_twin_shapes(twin):
    _, d = twin
    assert d["T"].shape == (6, 40, 48)
    assert np.isnan(d["T"]).any()
    assert np.nanmax(np.hypot(d["u"], d["v"])) < 0.5


def test_optflow_direction(twin):
    _, d = twin
    u, v = optflow.currents(d["T"], d["x"], d["y"], d["t"])
    assert u.shape == d["T"].shape
    assert np.isfinite(u).sum() > 0
    # the old baseline mixed per-km gradients with a per-pixel alpha and returned ~0.3% of the speed
    sp, sp_ref = np.nanmean(np.hypot(u, v)), np.nanmean(np.hypot(d["u"], d["v"]))
    assert sp > 0.3 * sp_ref
    ok = np.isfinite(u[2]) & np.isfinite(d["u"])
    assert np.corrcoef(u[2][ok], d["u"][ok])[0, 1] > 0.4


def test_train_and_evaluate(twin, tmp_path):
    p, d = twin
    run = tmp_path / "run"
    train(dict(x=d["x"], y=d["y"], t=d["t"], T=d["T"], u_ref=d["u"], v_ref=d["v"], src="synth"),
          run, steps=120, batch=512, log_every=40)
    assert (run / "fields.npz").exists()
    hist = json.loads((run / "history.json").read_text())
    assert hist[-1]["data"] < hist[0]["data"]
    out = tmp_path / "run.json"
    payload = evaluate(run, p, out=out)
    assert out.exists()
    assert payload["metrics"]["pinn"]["n"] > 100
    assert len(payload["u"]) == 6
