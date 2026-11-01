import argparse
import numpy as np
from scipy.ndimage import map_coordinates

KM_PER_DAY = 86.4  # 1 m/s


def flow(x, y, days):
    X, Y = np.meshgrid(x, y)
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]
    psi = (0.12 * KM_PER_DAY) * (Ly / (2 * np.pi)) * np.sin(2 * np.pi * X / Lx) * np.sin(2 * np.pi * Y / Ly)
    psi += (0.05 * KM_PER_DAY) * Y                         # mean westward drift
    psi += (0.08 * KM_PER_DAY) * 60 * np.exp(-((X - 0.7 * Lx) ** 2 + (Y - 0.35 * Ly) ** 2) / 70 ** 2)
    dy, dx = y[1] - y[0], x[1] - x[0]
    u = -np.gradient(psi, dy, axis=0)
    v = np.gradient(psi, dx, axis=1)
    return u, v


def run(nx=48, ny=40, dx=8.0, dy=11.0, days=21, kappa_m2s=50.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    u, v = flow(x, y, days)

    land = (X > 0.86 * x[-1]) & (Y < 0.28 * y[-1])
    u[land] = 0
    v[land] = 0

    T = 15 + 3 * Y / y[-1] + 1.8 * np.exp(-((X - 0.4 * x[-1]) ** 2 + (Y - 0.6 * y[-1]) ** 2) / 70 ** 2)
    T += 0.4 * np.sin(2 * np.pi * X / 160) * np.cos(2 * np.pi * Y / 200)
    S = 0.03 * np.sin(np.pi * X / x[-1]) * np.cos(np.pi * Y / y[-1])   # °C/day

    kappa = kappa_m2s * 86400 / 1e6   # km²/day
    dt = 0.25
    nsub = int(round(1 / dt))
    frames = [T.copy()]

    J, I = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    for d in range(days - 1):
        for _ in range(nsub):
            # semi-Lagrangian: look upstream, cubic interpolation
            jb = J - v * dt / dy
            ib = I - u * dt / dx
            Tb = map_coordinates(T, [jb, ib], order=3, mode="nearest")
            lap = (np.roll(T, -1, axis=1) - 2 * T + np.roll(T, 1, axis=1)) / dx ** 2 \
                + (np.roll(T, -1, axis=0) - 2 * T + np.roll(T, 1, axis=0)) / dy ** 2
            T = Tb + dt * (kappa * lap + S)
            T[:, 0], T[:, -1] = T[:, 1], T[:, -2]
            T[0, :], T[-1, :] = T[1, :], T[-2, :]
        frames.append(T.copy())

    T = np.stack(frames)
    T += noise * rng.standard_normal(T.shape)
    T[:, land] = np.nan
    u[land] = np.nan
    v[land] = np.nan
    return dict(x=x, y=y, t=np.arange(days, dtype=float), T=T.astype(np.float32),
                u=(u / KM_PER_DAY).astype(np.float32), v=(v / KM_PER_DAY).astype(np.float32),
                S=S.astype(np.float32), kappa=kappa)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth.npz")
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--noise", type=float, default=0.05)
    a = ap.parse_args()
    d = run(days=a.days, noise=a.noise)
    np.savez(a.out, **d)
    sp = np.hypot(d["u"], d["v"])
    print(f"{a.out}: {d['T'].shape}, max speed {np.nanmax(sp):.2f} m/s, kappa {d['kappa']:.2f} km²/day")
