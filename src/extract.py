"""
Peak isolation / deconvolution -> processed single-process peaks for kinetics.

Mel    : broad ~550 C melem->melon->g-C3N4 condensation. Single Fraser-Suzuki on a
         tangential linear baseline (no overlap). Kinetics use raw-minus-baseline.
MelBar : fused decomposition band. The melt spike is excluded by anchoring the baseline
         at the post-melt valley; the envelope is fitted with 2 Fraser-Suzuki peaks
         (Perejon/Sanchez-Jimenez 2011). The dominant CENTRAL peak (g-C3N4
         polycondensation) is extracted for kinetics; kinetics use the analytic central
         FS curve, validated against (data - baseline - shoulder).
MelTbar: single sharp exothermic peak (~300 C). Single Fraser-Suzuki on a linear baseline.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import dsc_lib as L

FIG_DIR = L.FIG_DIR

# ---- per-heating-rate extraction configuration --------------------------------
CFG = {
    "Mel": {  # single broad peak, linear (tangential) baseline
        1.0: dict(win=(500, 560), navg=20),
        3.0: dict(win=(508, 572), navg=20),
        5.0: dict(win=(512, 576), navg=20),
    },
    "MelBar": {  # valley search window (after melt) + right end of envelope
        1.0: dict(valley=(315, 342), right=372),
        3.0: dict(valley=(330, 350), right=382),
        5.0: dict(valley=(343, 353), right=388),
    },
    "MelTbar": {  # single sharp exo peak, linear baseline between flat shoulders
        1.0: dict(win=(285, 304), navg=150, sg=121),
        3.0: dict(win=(282, 310), navg=150, sg=121),
        5.0: dict(win=(286, 310.5), navg=150, sg=121),
    },
}
SMOOTH = 31  # Savitzky-Golay window (points) for the fit target


def _sg(y, w=SMOOTH):
    w = min(w, len(y) - (1 - len(y) % 2))
    if w < 5:
        return y
    if w % 2 == 0:
        w -= 1
    return savgol_filter(y, w, 3)


def extract_mel(beta, df, cfg, kind="Mel"):
    """Single-FS isolation of a single peak on a linear (tilted) baseline.
    Used for the broad melamine peak (~550 C) and the sharp melamine-
    thiobarbiturate exo peak (~300 C); kinetics use raw-minus-baseline."""
    t0, t1 = cfg["win"]
    z = df[(df.Tc >= t0 - 15) & (df.Tc <= t1 + 15)].reset_index(drop=True)
    T = z.Tc.values
    y = _sg(z.DSC.values, cfg.get("sg", 41))
    base = L.linear_baseline(T, y, t0, t1, navg=cfg["navg"])
    m = (T >= t0) & (T <= t1)
    Tf, yf = T[m], (y - base)[m]
    span = Tf[-1] - Tf[0]; ymax = max(yf.max(), 1e-3)
    popt, r2, rms = L.fit_multi_fs(
        Tf, yf, [ymax * 0.9, Tf[np.argmax(yf)], span * 0.3, 0.0],
        [0, Tf[0], 2, -0.9], [ymax * 3, Tf[-1], span * 1.5, 0.9])
    # kinetics target = raw-minus-baseline (clipped) on the window grid
    sig = np.clip(yf, 0, None)
    tsec = (Tf - Tf[0]) / beta * 60.0  # constant-beta heating (s)
    proc, area = L.make_alpha(Tf + 273.15, tsec, sig)
    return dict(beta=beta, Tc=Tf, base=base[m], raw=y[m], sig=sig,
                fs=[popt], center=popt[1], r2=r2, area=area, proc=proc,
                interp=L.interp_alpha(proc), window=(t0, t1), kind=kind)


def extract_melbar(beta, df, cfg):
    """Valley-anchored 2-FS deconvolution; extract the dominant CENTRAL peak."""
    vw = cfg["valley"]; right = cfg["right"]
    zz = df[(df.Tc >= vw[0]) & (df.Tc <= vw[1])]
    Tvalley = zz.Tc.values[np.argmin(_sg(zz.DSC.values, 21))]
    zd = df[(df.Tc >= Tvalley) & (df.Tc <= right)].reset_index(drop=True)
    Td = zd.Tc.values; yd = _sg(zd.DSC.values)
    base = L.linear_baseline(Td, yd, Td[0], Td[-1])
    yc = yd - base
    span = Td[-1] - Td[0]; ymax = yc.max(); c1 = Td[np.argmax(yc)]
    popt, r2, rms = L.fit_multi_fs(
        Td, yc,
        [ymax * 0.8, c1, span * 0.2, 0.0, ymax * 0.3, c1 + span * 0.15, span * 0.12, 0.0],
        [0, Td[0], 1, -0.9, 0, Td[0], 1, -0.9],
        [ymax * 2, Td[-1], span, 0.9, ymax * 2, Td[-1], span, 0.9])
    comps = [popt[i:i + 4] for i in range(0, len(popt), 4)]
    areas = [np.trapezoid(L.fraser_suzuki(Td, *c), Td) for c in comps]
    dom = int(np.argmax(areas))
    central = comps[dom]; shoulder = comps[1 - dom]
    # kinetics target = analytic central FS peak on an extended grid (tails -> 0)
    h, p, w, s = central
    Tg = np.linspace(p - 6 * abs(w), p + 6 * abs(w), 4000)
    Tg = Tg[(Tg >= Td[0] - 25) & (Tg <= right + 15)]
    sig = L.fraser_suzuki(Tg, *central)
    tsec = (Tg - Tg[0]) / beta * 60.0
    proc, area = L.make_alpha(Tg + 273.15, tsec, sig)
    # validation target = data - baseline - shoulder (real noise retained)
    sig_resid = np.clip(yc - L.fraser_suzuki(Td, *shoulder), 0, None)
    tsec_r = (Td - Td[0]) / beta * 60.0
    proc_r, _ = L.make_alpha(Td + 273.15, tsec_r, sig_resid)
    return dict(beta=beta, Tc=Td, base=base, raw=yd, yc=yc,
                fs=comps, areas=areas, central=central, shoulder=shoulder,
                center=central[1], shoulder_center=shoulder[1],
                area_frac=areas[dom] / sum(areas), r2=r2, valley=Tvalley,
                area=area, proc=proc, interp=L.interp_alpha(proc),
                interp_resid=L.interp_alpha(proc_r), proc_resid=proc_r,
                window=(Tvalley, right), kind="MelBar")


def _fs_unit_area(T, p, w, s):
    return np.trapezoid(L.fraser_suzuki(T, 1.0, p, w, s), T)


def extract_melbar_global(data):
    """
    Constrained SIMULTANEOUS deconvolution of all heating rates (ICTAC / Perejon):
    asymmetry s1,s2 and central area-fraction L1 are SHARED across beta; only
    position and width shift with beta. Resolves the FS-shape inconsistency that
    otherwise tilts Friedman Ea(alpha). Returns the per-beta peak dicts.
    """
    betas = sorted(data)
    prep = {}
    for beta in betas:
        cfg = CFG["MelBar"][beta]
        df = data[beta]; vw = cfg["valley"]; right = cfg["right"]
        zz = df[(df.Tc >= vw[0]) & (df.Tc <= vw[1])]
        Tvalley = zz.Tc.values[np.argmin(_sg(zz.DSC.values, 21))]
        zd = df[(df.Tc >= Tvalley) & (df.Tc <= right)].reset_index(drop=True)
        Td = zd.Tc.values; yd = _sg(zd.DSC.values)
        base = L.linear_baseline(Td, yd, Td[0], Td[-1])
        prep[beta] = dict(Td=Td, yc=yd - base, base=base, raw=yd,
                          valley=Tvalley, right=right, c1=Td[np.argmax(yd - base)])

    # parameter vector: [L1, s1, s2,  (Atot,p1,w1,p2,w2) per beta]
    def unpack(theta):
        L1, s1, s2 = theta[0], theta[1], theta[2]
        per = {}
        k = 3
        for beta in betas:
            per[beta] = theta[k:k + 5]; k += 5
        return L1, s1, s2, per

    def model_beta(theta, beta):
        L1, s1, s2, per = unpack(theta)
        Atot, p1, w1, p2, w2 = per[beta]
        Td = prep[beta]["Td"]
        ua1 = _fs_unit_area(Td, p1, w1, s1); ua2 = _fs_unit_area(Td, p2, w2, s2)
        h1 = (L1 * Atot) / ua1 if ua1 > 0 else 0.0
        h2 = ((1 - L1) * Atot) / ua2 if ua2 > 0 else 0.0
        return (L.fraser_suzuki(Td, h1, p1, w1, s1)
                + L.fraser_suzuki(Td, h2, p2, w2, s2), (h1, p1, w1, s1), (h2, p2, w2, s2))

    def resid(theta):
        out = []
        for beta in betas:
            m, _, _ = model_beta(theta, beta)
            out.append(m - prep[beta]["yc"])
        return np.concatenate(out)

    # initial guess + bounds
    theta0 = [0.85, -0.45, -0.30]
    lo = [0.55, -0.9, -0.9]; up = [0.98, 0.3, 0.3]
    for beta in betas:
        P = prep[beta]; Td = P["Td"]; yc = P["yc"]; span = Td[-1] - Td[0]
        Atot = max(np.trapezoid(np.clip(yc, 0, None), Td), 1e-3)
        c1 = P["c1"]
        theta0 += [Atot, c1, span * 0.22, c1 + span * 0.12, span * 0.10]
        lo += [Atot * 0.3, Td[0], 2, Td[0], 1]
        up += [Atot * 3, Td[-1], span, Td[-1], span]

    from scipy.optimize import least_squares
    res = least_squares(resid, theta0, bounds=(lo, up), max_nfev=40000, xtol=1e-12, ftol=1e-12)
    L1, s1, s2, per = unpack(res.x)

    peaks = {}
    for beta in betas:
        P = prep[beta]; Td = P["Td"]
        m, c1p, c2p = model_beta(res.x, beta)
        ss_res = np.sum((P["yc"] - m) ** 2); ss_tot = np.sum((P["yc"] - P["yc"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        # peak1 is central (L1>=0.5 by bound), peak2 shoulder
        central = np.array(c1p); shoulder = np.array(c2p)
        a1 = np.trapezoid(L.fraser_suzuki(Td, *central), Td)
        a2 = np.trapezoid(L.fraser_suzuki(Td, *shoulder), Td)
        # kinetics target = analytic central FS on extended grid
        h, p, w, s = central
        Tg = np.linspace(p - 6 * abs(w), p + 6 * abs(w), 4000)
        Tg = Tg[(Tg >= Td[0] - 25) & (Tg <= P["right"] + 15)]
        sig = L.fraser_suzuki(Tg, *central)
        tsec = (Tg - Tg[0]) / beta * 60.0
        proc, area = L.make_alpha(Tg + 273.15, tsec, sig)
        sig_resid = np.clip(P["yc"] - L.fraser_suzuki(Td, *shoulder), 0, None)
        proc_r, _ = L.make_alpha(Td + 273.15, (Td - Td[0]) / beta * 60.0, sig_resid)
        peaks[beta] = dict(beta=beta, Tc=Td, base=P["base"], raw=P["raw"], yc=P["yc"],
                           fs=[central, shoulder], central=central, shoulder=shoulder,
                           center=central[1], shoulder_center=shoulder[1],
                           area_frac=a1 / (a1 + a2), r2=r2, valley=P["valley"],
                           area=area, proc=proc, interp=L.interp_alpha(proc),
                           interp_resid=L.interp_alpha(proc_r), proc_resid=proc_r,
                           s_shared=(s1, s2), L1_shared=L1, window=(P["valley"], P["right"]),
                           kind="MelBar")
    return peaks


def extract_sample(sample, melbar_global=True):
    data = L.load_sample(sample)
    if sample == "MelBar" and melbar_global:
        return extract_melbar_global(data)
    peaks = {}
    for beta in sorted(data):
        cfg = CFG[sample][beta]
        if sample in ("Mel", "MelTbar"):
            peaks[beta] = extract_mel(beta, data[beta], cfg, kind=sample)
        else:
            peaks[beta] = extract_melbar(beta, data[beta], cfg)
    return peaks


def figure_deconv(sample, peaks, path):
    """Baseline-SUBTRACTED (normalized) view: the baseline is the horizontal axis (y = 0),
    peaks rise from zero. Top row = corrected signal + Fraser-Suzuki components; bottom = residuals."""
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.2), sharex="col",
                            gridspec_kw={"height_ratios": [3, 1]})
    for j, beta in enumerate(sorted(peaks)):
        pk = peaks[beta]; ax = axs[0, j]; axr = axs[1, j]
        T = np.asarray(pk["Tc"])
        corr = np.asarray(pk["raw"]) - np.asarray(pk["base"])   # baseline subtracted
        ax.axhline(0, color="0.5", lw=0.9, ls="--", label="baseline (y = 0)")
        ax.plot(T, corr, color="0.35", lw=1.4, label="DSC - baseline")
        if sample == "MelBar":
            fit = L.multi_fs(T, np.concatenate(pk["fs"]))
            for c in pk["fs"]:
                tag = "central" if np.allclose(c, pk["central"]) else "shoulder"
                ax.fill_between(T, 0, L.fraser_suzuki(T, *c), alpha=0.32,
                                label=f"{tag} {c[1]:.0f} C")
            ax.plot(T, fit, "r-", lw=1.6, label=f"FS sum, R2={pk['r2']:.4f}")
            ax.axvline(pk["valley"], color="m", ls=":", lw=1.2, label=f"valley {pk['valley']:.0f} C")
            resid = corr - fit
        else:
            fitfs = L.fraser_suzuki(T, *pk["fs"][0])
            ax.fill_between(T, 0, np.clip(corr, 0, None), alpha=0.32, color="tab:orange",
                            label=f"peak {pk['center']:.0f} C")
            ax.plot(T, fitfs, "r-", lw=1.4, label=f"Fraser-Suzuki, R2={pk['r2']:.3f}")
            resid = corr - fitfs
        ax.set_title(f"{beta:.0f} K/min"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
        axr.plot(T, resid, color="0.3", lw=0.9); axr.axhline(0, color="r", lw=0.8)
        axr.grid(alpha=0.3); axr.set_xlabel("T, C"); axr.set_ylabel("residual")
    axs[0, 0].set_ylabel("DSC - baseline, mW/mg")
    names = {
        "Mel": "Melamine - isolation of the ~550 C peak (Fraser-Suzuki, baseline subtracted)",
        "MelBar": "Melamine barbiturate - deconvolution (2x Fraser-Suzuki, shared asymmetry and area fraction)",
        "MelTbar": "Melamine thiobarbiturate - isolation of the ~300 C peak (Fraser-Suzuki, baseline subtracted)",
    }
    fig.suptitle(names[sample], fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    for sample in ["Mel", "MelBar", "MelTbar"]:
        peaks = extract_sample(sample)
        figure_deconv(sample, peaks, os.path.join(FIG_DIR, f"fig_deconv_{sample}.png"))
        print(f"\n=== {sample} ===")
        for beta in sorted(peaks):
            pk = peaks[beta]
            extra = ""
            if sample == "MelBar":
                extra = f" shoulder={pk['shoulder_center']:.1f} area_frac={pk['area_frac']:.3f}"
            print(f"  {beta:.0f}K: center={pk['center']:.1f}C  area={pk['area']:.3g}  R2={pk['r2']:.4f}{extra}")
