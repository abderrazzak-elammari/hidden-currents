import numpy as np
from scipy.ndimage import uniform_filter

KM_PER_DAY = 86.4


def horn_schunck(a, b, dx, dy, dt, alpha=2.0, iters=300):
    a = np.where(np.isfinite(a), a, np.nanmean(a))
    b = np.where(np.isfinite(b), b, np.nanmean(b))
    Ix = (np.gradient(a, dx, axis=1) + np.gradient(b, dx, axis=1)) / 2
    Iy = (np.gradient(a, dy, axis=0) + np.gradient(b, dy, axis=0)) / 2
    It = (b - a) / dt
    scale = np.nanstd(a) + 1e-6
    Ix, Iy, It = Ix / scale, Iy / scale, It / scale
    u = np.zeros_like(a)
    v = np.zeros_like(a)
    for _ in range(iters):
        ub = uniform_filter(u, 3, mode="nearest")
        vb = uniform_filter(v, 3, mode="nearest")
        k = (Ix * ub + Iy * vb + It) / (alpha ** 2 + Ix ** 2 + Iy ** 2)
        u = ub - Ix * k
        v = vb - Iy * k
    return u, v


def currents(T, x, y, t, alpha=2.0):
    dx, dy = x[1] - x[0], y[1] - y[0]
    out_u, out_v = [], []
    for k in range(len(t) - 1):
        u, v = horn_schunck(T[k], T[k + 1], dx, dy, t[k + 1] - t[k], alpha)
        out_u.append(u)
        out_v.append(v)
    out_u.append(out_u[-1])
    out_v.append(out_v[-1])
    sea = np.isfinite(T).all(0)
    u = np.where(sea, np.stack(out_u), np.nan) / KM_PER_DAY
    v = np.where(sea, np.stack(out_v), np.nan) / KM_PER_DAY
    return u.astype(np.float32), v.astype(np.float32)
