"""Read the exported JAABA classifier (.classifier.mat) produced by export_classifier.m.

The .mat is plain v7, so scipy.io.loadmat reads it directly (no h5py). Everything
downstream is driven by the returned Classifier object -- number of stumps, which
per-frame features / window stats / transforms / radii / offsets are needed, the
score normalization, the post-processing parameters, and the APT skeleton geometry
are all read from here, never hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------- scipy helpers
def _asstr(v) -> str:
    """Normalize a scipy-loaded MATLAB char to a python str."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    a = np.asarray(v).ravel()
    if a.size == 0:
        return ""
    if a.dtype.kind in "US":
        return str(a[0]) if a.size == 1 else "".join(str(x) for x in a)
    # numeric char codes
    return "".join(chr(int(c)) for c in a)


def _asscalar(v, cast=float):
    a = np.asarray(v).ravel()
    if a.size == 0:
        return None
    return cast(a[0])


def _asvec(v) -> np.ndarray:
    return np.asarray(v).ravel()


def _cell_of_str(v) -> list[str]:
    """A MATLAB 1xN cell of char -> list[str]."""
    a = np.asarray(v).ravel()
    return [_asstr(x) for x in a]


def _cell_of_vec(v) -> list[np.ndarray]:
    """A MATLAB 1xN cell of numeric vectors -> list of 1-D int arrays."""
    a = np.asarray(v, dtype=object).ravel()
    out = []
    for x in a:
        xv = np.asarray(x).ravel()
        out.append(xv.astype(int))
    return out


def _cell_of_extra(v) -> list[dict]:
    """The per-stump `extra` cell: each element is a MATLAB {key,val,...} cell or empty."""
    a = np.asarray(v, dtype=object).ravel()
    out = []
    for x in a:
        xa = np.asarray(x, dtype=object).ravel()
        d = {}
        for k in range(0, len(xa) - 1, 2):
            key = _asstr(xa[k])
            val = xa[k + 1]
            va = np.asarray(val).ravel()
            d[key] = va if va.size > 1 else _asscalar(va)
        out.append(d)
    return out


# ---------------------------------------------------------------- data classes
@dataclass(frozen=True)
class WFDesc:
    """A single window-feature descriptor: per-frame feature + window transform."""
    pff: str
    stat: str
    trans: str
    radius: int
    offset: int
    extra: tuple = ()  # tuple of (key, value) so the descriptor is hashable

    def key(self):
        return (self.pff, self.stat, self.trans, self.radius, self.offset, self.extra)


@dataclass
class Classifier:
    # per-stump ensemble (length nstumps)
    dir: np.ndarray
    tr: np.ndarray
    alpha: np.ndarray
    error: np.ndarray
    descs: list[WFDesc]          # per-stump window-feature descriptor
    # scalars / params
    score_norm: float
    pp: dict                     # method, hyst_hi, hyst_lo, filt_size, blen
    apt: dict                    # n_pts, head_tail, pairs, triads, projname
    behavior: str
    scorefilename: str
    trxfilename: str
    perframedir: str
    trkfilename: str
    jabfile: str
    # derived
    unique_descs: list[WFDesc] = field(default_factory=list)
    stump_col: np.ndarray = None  # (nstumps,) index into unique_descs

    @property
    def nstumps(self) -> int:
        return len(self.descs)

    def pff_names(self) -> list[str]:
        seen, out = set(), []
        for d in self.unique_descs:
            if d.pff not in seen:
                seen.add(d.pff)
                out.append(d.pff)
        return out

    def descs_by_pff(self) -> dict[str, list[WFDesc]]:
        out: dict[str, list[WFDesc]] = {}
        for d in self.unique_descs:
            out.setdefault(d.pff, []).append(d)
        return out


def load_classifier(path: str) -> Classifier:
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    dim = _asvec(m["dim"]).astype(int)  # 1-indexed original dims (kept for reference)
    ddir = _asvec(m["dir"]).astype(float)
    tr = _asvec(m["tr"]).astype(float)
    alpha = _asvec(m["alpha"]).astype(float)
    error = _asvec(m["error"]).astype(float) if "error" in m else np.full_like(tr, np.nan)

    pff = _cell_of_str(m["pff"])
    stat = _cell_of_str(m["stat"])
    trans = _cell_of_str(m["trans"])
    radius = _asvec(m["radius"])
    offset = _asvec(m["offset"])
    extra = _cell_of_extra(m["extra"]) if "extra" in m else [{}] * len(pff)

    descs = []
    for i in range(len(pff)):
        ex = tuple(sorted((k, tuple(np.atleast_1d(v).tolist()) if isinstance(v, np.ndarray) else v)
                          for k, v in extra[i].items()))
        descs.append(WFDesc(
            pff=pff[i], stat=stat[i], trans=trans[i],
            radius=int(round(float(radius[i]))), offset=int(round(float(offset[i]))),
            extra=ex,
        ))

    pp = dict(
        method=_asstr(m.get("pp_method", "")),
        hyst_hi=_asscalar(m.get("pp_hyst_hi", 0.0)),
        hyst_lo=_asscalar(m.get("pp_hyst_lo", 0.0)),
        filt_size=_asscalar(m.get("pp_filt_size", 1.0)),
        blen=_asscalar(m.get("pp_blen", 1.0)),
    )

    apt = dict(
        n_pts=_asscalar(m.get("apt_npts"), int) if m.get("apt_npts") is not None else None,
        head_tail=_asvec(m["apt_headtail"]).astype(int) if "apt_headtail" in m and np.asarray(m["apt_headtail"]).size else None,
        pairs=_cell_of_vec(m["apt_pairs"]) if "apt_pairs" in m and np.asarray(m["apt_pairs"], dtype=object).size else [],
        triads=_cell_of_vec(m["apt_triads"]) if "apt_triads" in m and np.asarray(m["apt_triads"], dtype=object).size else [],
        projname=_asstr(m.get("apt_projname", "")),
        fov=_asscalar(m.get("pf_fov", np.pi)) if m.get("pf_fov") is not None else np.pi,
        max_dnose2ell_anglerange=_asscalar(m.get("pf_max_dnose2ell_anglerange", 127.0))
        if m.get("pf_max_dnose2ell_anglerange") is not None else 127.0,
        nbodylengths_near=_asscalar(m.get("pf_nbodylengths_near", 2.5))
        if m.get("pf_nbodylengths_near") is not None else 2.5,
    )

    clf = Classifier(
        dir=ddir, tr=tr, alpha=alpha, error=error, descs=descs,
        score_norm=_asscalar(m["scoreNorm"]),
        pp=pp, apt=apt,
        behavior=_asstr(m.get("behavior", "")),
        scorefilename=_asstr(m.get("scorefilename", "")),
        trxfilename=_asstr(m.get("trxfilename", "")),
        perframedir=_asstr(m.get("perframedir", "perframe")),
        trkfilename=_asstr(m.get("trkfilename", "")),
        jabfile=_asstr(m.get("jabfile", "")),
    )

    # dedup descriptors -> unique_descs + per-stump column index
    uniq: dict = {}
    cols = np.empty(clf.nstumps, dtype=int)
    for i, d in enumerate(clf.descs):
        k = d.key()
        if k not in uniq:
            uniq[k] = len(uniq)
            clf.unique_descs.append(d)
        cols[i] = uniq[k]
    clf.stump_col = cols
    return clf


if __name__ == "__main__":
    import sys
    clf = load_classifier(sys.argv[1])
    print(f"nstumps={clf.nstumps}  unique window features={len(clf.unique_descs)}")
    print(f"scoreNorm={clf.score_norm}  pp={clf.pp}")
    print(f"behavior={clf.behavior!r}  scorefile={clf.scorefilename!r}  trx={clf.trxfilename!r}  trk={clf.trkfilename!r}")
    print(f"apt: n_pts={clf.apt['n_pts']} head_tail={clf.apt['head_tail']} "
          f"npairs={len(clf.apt['pairs'])} ntriads={len(clf.apt['triads'])}")
    pffs = clf.pff_names()
    print(f"unique per-frame features ({len(pffs)}):")
    for p in sorted(pffs):
        print("   ", p)
    stats = sorted({d.stat for d in clf.unique_descs})
    transs = sorted({d.trans for d in clf.unique_descs})
    radii = sorted({d.radius for d in clf.unique_descs})
    offs = sorted({d.offset for d in clf.unique_descs})
    print("stats:", stats)
    print("trans:", transs)
    print("radii:", radii)
    print("offsets:", offs)
