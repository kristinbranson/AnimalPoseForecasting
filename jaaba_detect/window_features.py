"""Window features: reproduce JAABA's Compute*WindowFeatures / *WindowCore exactly.

Given a per-frame vector x (1-D, length N for one fly) and a window-feature
descriptor (stat, trans, radius r, offset off), produce the length-N window
feature. The window at frame t spans [t-r+off, t+r+off].

Matches the MATLAB "fast path" (imfilter/imerode/imdilate based), which is what
JAABA actually feeds to the classifier -- including its boundary handling:
  - mean core:  full box convolution, boundary windows normalized by partial count
  - min core:   x padded with -inf (width r) then sliding min with +inf outside
                => the first/last r frames are -inf (deliberate JAABA behavior)
  - max core:   x padded with -inf then sliding max with -inf outside
  - std core:   sqrt(max(0, mean(x^2) - mean(x)^2)), skipped for r==0
Transforms none/abs/flip/relative. For min/max the transform is applied BEFORE
the window op; for mean/std the transform (abs/flip) is applied AFTER (relative is
always a separate core on the relative-transformed series). Relative bins are
computed from x itself (data-driven), matching ComputeWindowFeatures.m.

The block-processing + MAXDEPENDENCYRADIUS trimming in JLabelData is only an
optimization; computing on the full per-frame vector gives identical results.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, maximum_filter1d


# ------------------------------------------------------------------ relative
def matlab_prctile(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """MATLAB prctile: (i-0.5)/n plotting positions (Hazen), clamped at the extremes."""
    xs = np.sort(x)
    n = xs.size
    if n == 0:
        return np.full(np.size(p), np.nan)
    if n == 1:
        return np.full(np.size(p), xs[0], dtype=float)
    q = 100.0 * (np.arange(1, n + 1) - 0.5) / n
    return np.interp(p, q, xs)  # np.interp clamps outside [q0,qn] to xs[0]/xs[-1]


def relative_bins(x: np.ndarray, resolution: int = 2, nsamples: int = 5000) -> np.ndarray:
    """Percentile bin edges for the 'relative' transform (ComputeWindowFeatures.m)."""
    prc = np.arange(0, 100 + 1e-9, resolution)
    nx = x[~np.isnan(x)]
    if nx.size > nsamples:
        # MATLAB round() is half-away-from-zero; np.round is half-to-even. For these
        # positive indices, floor(x+0.5) matches MATLAB (matters on long real vectors).
        sx = np.floor(np.linspace(1, nx.size, nsamples) + 0.5).astype(int) - 1
        return matlab_prctile(nx[sx], prc)
    if nx.size == 0:
        return np.linspace(0, 1, prc.size)
    if nx.min() == nx.max():
        return np.linspace(nx.min() - 0.1, nx.max() + 0.1, prc.size)
    return matlab_prctile(nx, prc)


def histc_bin(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """MATLAB histc 2nd output: 1-based bin index (edges[k]<=x<edges[k+1]); the last
    bin also catches x==edges[-1]; 0 for x<edges[0], x>edges[-1], or NaN."""
    edges = np.asarray(edges, float)
    nb = len(edges) - 1
    k = np.searchsorted(edges, x, side="right").astype(float)  # 0..nb+1
    k[x > edges[-1]] = 0
    k[np.isnan(x)] = 0
    k[k > nb] = nb                                              # x==edges[-1] -> last bin
    return k


def convert_to_relative(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Percentile-rank transform of x given bins (convertToRelative.m)."""
    bins = bins.astype(float).copy()
    # enforce non-decreasing bins (MATLAB while-loop)
    for i in range(len(bins) - 1):
        if bins[i] > bins[i + 1]:
            bins[i + 1] = bins[i]
    nb = len(bins)
    # MATLAB histc 2nd output: 1-based bin index; 0 for x<bins(1), x>bins(end), NaN
    k = np.searchsorted(bins, x, side="right").astype(float)  # 0..nb
    k[x > bins[-1]] = 0
    k[np.isnan(x)] = 0
    modX = k.copy()
    modX[modX > nb] = nb
    valid = (~np.isnan(modX)) & (modX < nb) & (modX > 0)
    vi = modX[valid].astype(int)  # 1-based bin index
    extra = x[valid] - bins[vi - 1]
    denom = bins[vi] - bins[vi - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        relExtra = extra / denom
    modX[valid] = modX[valid] + relExtra
    return modX - 1.0


# ------------------------------------------------------------------ cores
def _mean_core(x: np.ndarray, w: int) -> np.ndarray:
    """MeanWindowCore(x,w): full box conv, boundary-partial normalization. len N+w-1."""
    res = np.convolve(x, np.ones(w), "full")  # NaN propagates, matching imfilter
    L = res.size
    res[w - 1:L - w + 1] /= w
    res[0:w - 1] /= np.arange(1, w)
    res[L - w + 1:L] /= np.arange(w - 1, 0, -1)
    return res


def _min_core(x: np.ndarray, r: int) -> np.ndarray:
    """MinWindowCore: x[nan]->+inf, pad -inf(r) each side, sliding min (+inf outside)."""
    w = 2 * r + 1
    xf = np.where(np.isnan(x), np.inf, x)
    x_pad = np.concatenate([np.full(r, -np.inf), xf, np.full(r, -np.inf)])
    return minimum_filter1d(x_pad, size=w, mode="constant", cval=np.inf)


def _max_core(x: np.ndarray, r: int) -> np.ndarray:
    """MaxWindowCore: x[nan]->-inf, pad -inf(r) each side, sliding max (-inf outside)."""
    w = 2 * r + 1
    xf = np.where(np.isnan(x), -np.inf, x)
    x_pad = np.concatenate([np.full(r, -np.inf), xf, np.full(r, -np.inf)])
    return maximum_filter1d(x_pad, size=w, mode="constant", cval=-np.inf)


def _std_core(x: np.ndarray, w: int) -> np.ndarray:
    """sqrt(max(0, mean(x^2) - mean(x)^2)) over the window. len N+w-1."""
    m = _mean_core(x, w)
    m2 = _mean_core(x * x, w)
    d = m2 - m * m
    d[d < 0] = 0
    return np.sqrt(d)


def _apply_offset(res_full: np.ndarray, N: int, r: int, off: int) -> np.ndarray:
    """res1[t] = res_full[t + r + off], NaN outside (padgrab2 semantics)."""
    idx = np.arange(N) + r + off
    out = np.full(N, np.nan)
    ok = (idx >= 0) & (idx < res_full.size)
    out[ok] = res_full[idx[ok]]
    return out


def _shift(res_full: np.ndarray, N: int, shift: int) -> np.ndarray:
    """res1[t] = res_full[t + shift], NaN outside."""
    idx = np.arange(N) + shift
    out = np.full(N, np.nan)
    ok = (idx >= 0) & (idx < res_full.size)
    out[ok] = res_full[idx[ok]]
    return out


def _change(x: np.ndarray, r: int, off: int, change_r: int) -> np.ndarray:
    """ComputeChangeWindowFeatures: (mean over end sub-window - mean over start)/r."""
    N = x.size
    change_w = 2 * change_r + 1
    res_mean = _mean_core(x, change_w)          # len N+2*change_r
    w = 2 * r + 1
    L = res_mean.size
    res = res_mean[w - 1:] - res_mean[:L - w + 1]
    return _shift(res, N, off - r + change_r) / r


def _harmonic_core(x: np.ndarray, w: int, nh: int) -> np.ndarray:
    """HarmonicWindowCore: cosine-correlation of the window (len N, NaN at 2 ends)."""
    r = (w - 1) // 2
    N = x.size
    res = np.full(N, np.nan)
    fil = np.cos(np.linspace(0, np.pi * nh, w)) / w * (nh + 1)
    if N >= w:
        # centered full-window correlation for t in [r, N-r-1]
        c = np.correlate(x, fil, mode="valid")   # length N-w+1, aligned to t=r..N-r-1
        res[r:N - r] = c
    for smallr in range(1, r):
        smallw = 2 * smallr + 1
        if smallw > N:
            res[smallr] = np.nan
            res[N - 1 - smallr] = np.nan
        else:
            filb = np.cos(np.linspace(0, np.pi * nh, smallw)) / smallw * (nh + 1)
            res[smallr] = np.sum(filb * x[:smallw])
            res[N - 1 - smallr] = np.sum(filb * x[N - smallw:])
    return res


# ------------------------------------------------------------------ dispatch
def _trans_after(y, x, trans):
    """Apply a post-window transform (mean/change/harmonic style)."""
    if trans in ("none", "relative"):
        return y
    if trans == "abs":
        return np.abs(y)
    if trans == "flip":
        y = y.copy()
        y[x < 0] = -y[x < 0]
        return y
    raise ValueError(f"trans {trans!r}")


def window_feature(x: np.ndarray, stat: str, trans: str, radius: int, offset: int,
                   relbins: np.ndarray | None = None, extra: dict | None = None) -> np.ndarray:
    """Compute one window feature (length N) for descriptor (stat,trans,radius,offset[,extra])."""
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    r = int(radius)
    off = int(offset)
    w = 2 * r + 1
    extra = extra or {}

    def relx():
        b = relbins if relbins is not None else relative_bins(x)
        return convert_to_relative(x, b)

    if stat == "mean":
        if trans == "relative":
            res = _mean_core(relx(), w)
            return _apply_offset(res, N, r, off)
        res = _mean_core(x, w)
        y = _apply_offset(res, N, r, off)
        if trans == "none":
            return y
        if trans == "abs":
            return np.abs(y)
        if trans == "flip":
            y = y.copy()
            y[x < 0] = -y[x < 0]
            return y
        raise ValueError(f"mean trans {trans!r}")

    if stat == "std":
        if r == 0:
            return np.zeros(N)
        if trans == "relative":
            return _apply_offset(_std_core(relx(), w), N, r, off)
        if trans in ("none", "abs", "flip"):  # std only produces none/relative; treat others as none
            return _apply_offset(_std_core(x, w), N, r, off)
        raise ValueError(f"std trans {trans!r}")

    if stat in ("min", "max"):
        core = _min_core if stat == "min" else _max_core
        if trans == "none":
            return _apply_offset(core(x, r), N, r, off)
        if trans == "abs":
            return _apply_offset(core(np.abs(x), r), N, r, off)
        if trans == "relative":
            return _apply_offset(core(relx(), r), N, r, off)
        if trans == "flip":
            y_orig = _apply_offset(core(x, r), N, r, off)
            y_flip = _apply_offset(core(-x, r), N, r, off)
            y = y_orig.copy()
            neg = x < 0
            y[neg] = y_flip[neg]
            return y
        raise ValueError(f"{stat} trans {trans!r}")

    if stat == "change":
        change_r = int(extra.get("change_window_radius", 0))
        if trans == "relative":
            return _change(relx(), r, off, change_r)
        return _trans_after(_change(x, r, off, change_r), x, trans)

    if stat == "harmonic":
        nh = int(extra.get("num_harmonic", 1))
        if nh >= w:
            return np.full(N, np.nan)
        if trans == "relative":
            return _shift(_harmonic_core(relx(), w, nh), N, off)
        return _trans_after(_shift(_harmonic_core(x, w, nh), N, off), x, trans)

    if stat == "diff_neighbor_mean":
        # mean-style: difference first, transform after (flip = *sign(x))
        def dnm(xt):
            return xt - _apply_offset(_mean_core(xt, w), N, r, off)
        if trans == "relative":
            return dnm(relx())
        y = dnm(x)
        if trans == "none":
            return y
        if trans == "abs":
            return np.abs(y)
        if trans == "flip":
            return y * np.sign(x)
        raise ValueError(f"diff_neighbor_mean trans {trans!r}")

    if stat in ("diff_neighbor_min", "diff_neighbor_max"):
        # min/max-style: transform x first, then window and subtract
        core = _min_core if stat == "diff_neighbor_min" else _max_core

        def dn(xt):
            win = _apply_offset(core(xt, r), N, r, off)
            return (win - xt) if stat == "diff_neighbor_max" else (xt - win)

        if trans == "none":
            return dn(x)
        if trans == "abs":
            return dn(np.abs(x))
        if trans == "relative":
            return dn(relx())
        if trans == "flip":
            y = dn(x).copy()
            y[x < 0] = dn(-x)[x < 0]
            return y
        raise ValueError(f"{stat} trans {trans!r}")

    if stat == "zscore_neighbors":
        if r == 0:
            return np.zeros(N)
        xt = relx() if trans == "relative" else x
        m = _apply_offset(_mean_core(xt, w), N, r, off)
        s = _apply_offset(_std_core(xt, w), N, r, off)
        s = np.where(s == 0, 1.0, s)
        y = (xt - m) / s
        return y if trans in ("none", "relative") else _trans_after(y, x, trans)

    if stat == "prctile":
        p = float(extra["prctile"])

        def prc(xt):
            res = np.full(N + 2 * r, np.nan)
            for m in range(N + 2 * r):
                lo = max(0, m - 2 * r); hi = min(N, m + 1)
                if hi > lo:
                    wv = xt[lo:hi]
                    wv = wv[~np.isnan(wv)]
                    if wv.size:
                        res[m] = matlab_prctile(wv, np.array([p]))[0]
            return _apply_offset(res, N, r, off)
        if trans == "none":
            return prc(x)
        if trans == "abs":
            return np.abs(prc(x))
        if trans == "flip":                       # ComputePrctile: res1 .* sign(x)
            return prc(x) * np.sign(x)
        if trans == "relative":                   # ComputePrctile: resRel1 .* sign(x)
            return prc(relx()) * np.sign(x)
        raise ValueError(f"prctile trans {trans!r}")

    if stat == "hist":
        # one bin per feature. If the full edge array + 1-based bin index are available
        # (exported from windowFeaturesParams), bin exactly via histc; else fall back to
        # the descriptor's two edges (exact for interior bins; only trans=none).
        full = extra.get("hist_edges_full")
        bini = extra.get("hist_bin")
        if trans == "none":
            if full is not None and bini is not None:
                I_b = (histc_bin(x, np.asarray(full, float)) == int(bini)).astype(float)
            else:
                e = np.atleast_1d(np.asarray(extra.get("hist_edges"), float)).ravel()
                I_b = ((x >= float(e[0])) & (x < float(e[-1]))).astype(float)
            return _apply_offset(_mean_core(I_b, w), N, r, off)
        if trans == "relative":
            if full is None or bini is None:
                raise NotImplementedError(
                    "hist/relative needs the full hist_edges array + bin index; re-export "
                    "the classifier so windowFeaturesParams hist_edges are carried.")
            nbins = len(np.asarray(full, float)) - 1
            rel_edges = np.linspace(0, 100, nbins + 1)
            I_b = (histc_bin(relx(), rel_edges) == int(bini)).astype(float)
            return _apply_offset(_mean_core(I_b, w), N, r, off)
        raise NotImplementedError(f"hist trans {trans!r} not emitted by JAABA")

    raise NotImplementedError(f"stat {stat!r} not implemented yet")


def compute_matrix(x: np.ndarray, descs) -> np.ndarray:
    """Compute a (N, len(descs)) matrix for a list of WFDesc-like objects.

    Relative bins are computed once from x and shared (they depend only on x).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    relb = None
    if any(getattr(d, "trans", None) == "relative" for d in descs):
        relb = relative_bins(x)
    X = np.empty((N, len(descs)), dtype=float)
    for j, d in enumerate(descs):
        X[:, j] = window_feature(x, d.stat, d.trans, d.radius, d.offset, relbins=relb)
    return X
