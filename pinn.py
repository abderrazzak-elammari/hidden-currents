import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

KM_PER_DAY = 86.4


# ---------------------------------------------------------------- data

def tidy(ds):
    ren = {}
    for k in set(ds.dims) | set(ds.coords):
        low = k.lower()
        for want, names in (("latitude", ("lat", "latitude")), ("longitude", ("lon", "longitude")), ("time", ("time",))):
            if low in names and k != want:
                ren[k] = want
    if ren:
        ds = ds.rename(ren)
    return ds.sortby("latitude").sortby("longitude")


def flat3(a):
    extra = [d for d in a.dims if d not in ("time", "latitude", "longitude")]
    if extra:
        a = a.isel({d: 0 for d in extra})
    return a.transpose("time", "latitude", "longitude")


def load(path):
    p = pathlib.Path(path)
    if p.suffix == ".npz":
        d = np.load(p)
        return dict(x=d["x"], y=d["y"], t=d["t"], T=d["T"], u_ref=d["u"], v_ref=d["v"], src="synth")
    import xarray as xr
    ds = tidy(xr.open_dataset(p))
    name = "analysed_sst" if "analysed_sst" in ds else next(v for v in ds.data_vars if ds[v].ndim >= 3)
    sst = flat3(ds[name])
    lat = sst["latitude"].values.astype(float)
    lon = sst["longitude"].values.astype(float)
    T = sst.values.astype(np.float32)
    if np.nanmean(T) > 100:
        T = T - 273.15
    lat0 = lat.mean()
    x = (lon - lon.mean()) * 111.32 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111.32
    tt = sst["time"].values
    t = (tt - tt[0]) / np.timedelta64(1, "D")
    return dict(x=x, y=y, t=t.astype(float), T=T, lat=lat, lon=lon, time=tt, src="sst")


class Scaler:
    def __init__(self, d):
        self.xc, self.yc = d["x"].mean(), d["y"].mean()
        self.L = max(np.ptp(d["x"]), np.ptp(d["y"])) / 2
        self.Tdays = d["t"].max()
        self.Tm, self.Ts = np.nanmean(d["T"]), np.nanstd(d["T"])
        self.U = self.L / self.Tdays            # km/day for a dimensionless velocity of 1

    def xyt(self, x, y, t):
        return np.stack([(x - self.xc) / self.L, (y - self.yc) / self.L, t / self.Tdays], -1).astype(np.float32)

    def T(self, T):
        return (T - self.Tm) / self.Ts


def tabulate(d, sc):
    X, Y = np.meshgrid(d["x"], d["y"])
    pts, vals = [], []
    for k, tk in enumerate(d["t"]):
        ok = np.isfinite(d["T"][k])
        pts.append(sc.xyt(X[ok], Y[ok], np.full(ok.sum(), tk)))
        vals.append(sc.T(d["T"][k][ok]))
    return np.concatenate(pts), np.concatenate(vals).astype(np.float32)[:, None]


# ---------------------------------------------------------------- networks

class Fourier(nn.Module):
    def __init__(self, n_feat, sigma):
        super().__init__()
        sigma = torch.tensor(sigma, dtype=torch.float32).reshape(-1, 1)
        self.register_buffer("B", torch.randn(3, n_feat) * sigma)

    def forward(self, p):
        z = 2 * math.pi * p @ self.B
        return torch.cat([torch.sin(z), torch.cos(z)], -1)


class Net(nn.Module):
    def __init__(self, width=128, depth=4, n_feat=64, sigma=(2.0, 2.0, 0.5)):
        super().__init__()
        self.ff = Fourier(n_feat, sigma)
        layers, d = [], 2 * n_feat
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, p):
        return self.mlp(self.ff(p))


def grad(f, p):
    return torch.autograd.grad(f, p, torch.ones_like(f), create_graph=True)[0]


class HiddenCurrents(nn.Module):
    def __init__(self, sigma_T=(2.5, 2.5, 0.6), sigma_psi=(1.2, 1.2, 0.3), sigma_S=(0.4, 0.4, 0.2), kappa0=1e-3):
        super().__init__()
        self.T = Net(sigma=sigma_T)
        self.psi = Net(sigma=sigma_psi)
        self.S = Net(width=64, depth=3, n_feat=16, sigma=sigma_S)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa0)))

    def velocity(self, p):
        p = p.detach().requires_grad_(True)
        g = grad(self.psi(p), p)
        return -g[:, 1:2], g[:, 0:1], g[:, 2:3]

    def residual(self, p):
        p = p.detach().requires_grad_(True)
        T = self.T(p)
        gT = grad(T, p)
        Txx = grad(gT[:, 0:1], p)[:, 0:1]
        Tyy = grad(gT[:, 1:2], p)[:, 1:2]
        gp = grad(self.psi(p), p)
        u, v, psi_t = -gp[:, 1:2], gp[:, 0:1], gp[:, 2:3]
        S = self.S(p)
        kappa = torch.exp(self.log_kappa)
        r = gT[:, 2:3] + u * gT[:, 0:1] + v * gT[:, 1:2] - kappa * (Txx + Tyy) - S
        return r, S, psi_t


# ---------------------------------------------------------------- training

def gnorm(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None) + 1e-12)


def train(d, out, steps=8000, batch=4096, lr=1e-3, lam_S=1e-2, lam_psi_t=1e-3, lam_min=0.5, seed=0, device="cpu", log_every=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)

    sc = Scaler(d)
    P, V = tabulate(d, sc)
    P, V = torch.tensor(P, device=device), torch.tensor(V, device=device)
    tmax = P[:, 2].max().item()

    sea = np.isfinite(d["T"]).all(0)
    sea_t = torch.tensor(sea, device=device)
    nx, ny = len(d["x"]), len(d["y"])
    lo = torch.tensor([P[:, 0].min(), P[:, 1].min()], device=device)
    hi = torch.tensor([P[:, 0].max(), P[:, 1].max()], device=device)

    def colloc(n, t_hi):
        xy = lo + (hi - lo) * torch.rand(n * 2, 2, device=device)
        ix = ((xy[:, 0] - lo[0]) / (hi[0] - lo[0]) * (nx - 1)).round().long().clamp(0, nx - 1)
        iy = ((xy[:, 1] - lo[1]) / (hi[1] - lo[1]) * (ny - 1)).round().long().clamp(0, ny - 1)
        xy = xy[sea_t[iy, ix]][:n]
        t = torch.rand(len(xy), 1, device=device) * t_hi
        return torch.cat([xy, t], 1)

    model = HiddenCurrents().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)

    stages = [(0.34, 0.15), (0.67, 0.35), (1.0, 1.0)]      # (fraction of the time window exposed, fraction of the steps)
    lam_pde = 1.0
    hist = []
    t0 = time.time()
    for it in range(steps):
        frac = it / steps
        t_hi = tmax
        for f_t, f_steps in stages:
            if frac < f_steps:
                t_hi = f_t * tmax
                break
        keep = (P[:, 2] <= t_hi).nonzero().squeeze(1)
        idx = keep[torch.randint(len(keep), (batch,), device=device)]
        pred = model.T(P[idx])
        l_data = ((pred - V[idx]) ** 2).mean()

        pc = colloc(batch, t_hi)
        r, S, psi_t = model.residual(pc)
        l_pde = (r ** 2).mean()
        l_reg = lam_S * (S ** 2).mean() + lam_psi_t * (psi_t ** 2).mean()

        if it % 50 == 0 and it > 0:
            gd, gp = gnorm(l_data, params), gnorm(l_pde, params)
            # floor: on real data the residual cannot reach zero, and pure gradient balancing
            # then walks lam to ~0, i.e. switches the physics off and psi stops learning
            lam_pde = max(lam_min, 0.9 * lam_pde + 0.1 * (gd / gp).clamp(1e-3, 1e3).item())

        loss = l_data + lam_pde * l_pde + l_reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if it % log_every == 0 or it == steps - 1:
            kappa_km2d = math.exp(model.log_kappa.item()) * sc.L ** 2 / sc.Tdays
            hist.append(dict(it=it, data=l_data.item(), pde=l_pde.item(), lam=lam_pde,
                             kappa_m2s=kappa_km2d * 1e6 / 86400, t_hi=t_hi / tmax))
            print(f"{it:6d}  data {l_data.item():.3e}  pde {l_pde.item():.3e}  lam {lam_pde:7.2f}  "
                  f"kappa {hist[-1]['kappa_m2s']:7.1f} m²/s  window {t_hi/tmax:.2f}  {time.time()-t0:5.0f}s")

    torch.save(model.state_dict(), out / "model.pt")
    export(model, d, sc, out, hist, device)
    return model


def export(model, d, sc, out, hist, device):
    X, Y = np.meshgrid(d["x"], d["y"])
    U, Vv, Tf, Sf = [], [], [], []
    for tk in d["t"]:
        p = torch.tensor(sc.xyt(X.ravel(), Y.ravel(), np.full(X.size, tk)), device=device)
        u, v, _ = model.velocity(p)
        with torch.no_grad():
            Tf.append((model.T(p) * sc.Ts + sc.Tm).cpu().numpy().reshape(X.shape))
            Sf.append((model.S(p) * sc.Ts / sc.Tdays).cpu().numpy().reshape(X.shape))
        U.append(u.detach().cpu().numpy().reshape(X.shape) * sc.U / KM_PER_DAY)
        Vv.append(v.detach().cpu().numpy().reshape(X.shape) * sc.U / KM_PER_DAY)
    sea = np.isfinite(d["T"]).all(0)
    U, Vv, Tf, Sf = (np.where(sea, np.stack(a), np.nan).astype(np.float32) for a in (U, Vv, Tf, Sf))
    kappa_km2d = math.exp(model.log_kappa.item()) * sc.L ** 2 / sc.Tdays
    np.savez(out / "fields.npz", x=d["x"], y=d["y"], t=d["t"], u=U, v=Vv, T_fit=Tf, S=Sf,
             kappa_m2s=kappa_km2d * 1e6 / 86400)
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print("kappa", f"{kappa_km2d*1e6/86400:.1f} m²/s", "->", out / "fields.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth.npz")
    ap.add_argument("--out", default="runs/synth")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-min", type=float, default=0.5, help="floor on the PDE weight")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    train(load(a.data), a.out, steps=a.steps, batch=a.batch, lr=a.lr, lam_min=a.lam_min, device=a.device)
