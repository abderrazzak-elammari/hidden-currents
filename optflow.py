import numpy as np
from scipy.ndimage import convolve

KM_PER_DAY = 86.4

# classic Horn–Schunck neighbour average, centre excluded
HS_KERNEL = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype=float) / 12

# best of a sweep on the synthetic twin (0.01 to 1), so the baseline gets its best
# setting in hindsight. Pixel units, SST scaled to unit std. The old version took
# gradients per km with alpha=2: regularisation 100x too strong, speeds near zero.
ALPHA = 0.3


def horn_schunck(a, b, alpha=ALPHA, iters=800):
    fill = np.nanmean(a)
    a = np.where(np.isfinite(a), a, fill)
    b = np.where(np.isfinite(b), b, fill)
    s = np.std(a) + 1e-9
    a, b = (a - fill) / s, (b - fill) / s
    Ix = (np.gradient(a, axis=1) + np.gradient(b, axis=1)) / 2     # per pixel
    Iy = (np.gradient(a, axis=0) + np.gradient(b, axis=0)) / 2
    It = b - a                                                     # per frame
    u = np.zeros_like(a)
    v = np.zeros_like(a)
    den = alpha ** 2 + Ix ** 2 + Iy ** 2
    for _ in range(iters):
        ub = convolve(u, HS_KERNEL, mode="nearest")
        vb = convolve(v, HS_KERNEL, mode="nearest")
        k = (Ix * ub + Iy * vb + It) / den
        u = ub - Ix * k
        v = vb - Iy * k
    return u, v                                                    # pixels per frame


def currents(T, x, y, t, alpha=ALPHA):
    dx, dy = x[1] - x[0], y[1] - y[0]
    out_u, out_v = [], []
    for k in range(len(t) - 1):
        u, v = horn_schunck(T[k], T[k + 1], alpha)
        step = t[k + 1] - t[k]
        out_u.append(u * dx / step)                               # km / day
        out_v.append(v * dy / step)
    out_u.append(out_u[-1])
    out_v.append(out_v[-1])
    sea = np.isfinite(T).all(0)
    u = np.where(sea, np.stack(out_u), np.nan) / KM_PER_DAY
    v = np.where(sea, np.stack(out_v), np.nan) / KM_PER_DAY
    return u.astype(np.float32), v.astype(np.float32)
