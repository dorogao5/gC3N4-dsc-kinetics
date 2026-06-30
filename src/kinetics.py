"""
Kinetic analysis driver: isoconversional Ea(alpha) (Friedman/KAS/FWO/Starink/Vyazovkin),
Criado master-plot mechanism selection, compensation effect (lnA), and forward
(inverse-modelling) reconstruction. Operates on the peaks produced by extract.py.
"""
import os
import numpy as np
import pandas as pd
import dsc_lib as L
import extract as X

DATA_OUT_DIR = L.DATA_OUT_DIR
ALPHAS = np.round(np.arange(0.05, 0.951, 0.05), 3)         # isoconversional grid
CRIADO_GRID = np.round(np.arange(0.10, 0.901, 0.025), 4)   # master-plot grid


def analyze(sample, use_resid=False):
    peaks_d = X.extract_sample(sample)
    peaks = []
    for beta in sorted(peaks_d):
        pk = peaks_d[beta]
        interp = pk["interp_resid"] if (use_resid and "interp_resid" in pk) else pk["interp"]
        peaks.append({"beta": beta, "interp": interp, "pk": pk})

    iso = L.isoconversional(peaks, ALPHAS)
    vy = L.vyazovkin_advanced(peaks, ALPHAS)
    iso = iso.merge(vy, on="alpha", how="left")

    mech, rank, ydf = L.rank_models(peaks, iso, CRIADO_GRID)
    iso["lnA"] = L.lnA_from_friedman(iso, mech)
    iso["A"] = np.exp(iso["lnA"])

    return dict(sample=sample, peaks_d=peaks_d, peaks=peaks, iso=iso,
                mech=mech, rank=rank, ydf=ydf)


def model_triplet(res, code, e_choice="Fr", arange=(0.1, 0.9)):
    """Kinetic triplet (E, lnA) for a model from isoconversional E + Friedman intercept."""
    iso = res["iso"]
    m = (iso["alpha"] >= arange[0]) & (iso["alpha"] <= arange[1])
    E = iso.loc[m, f"{e_choice}_Ea"].mean() * 1000.0           # J/mol
    f = L.MODELS[code][0]
    lnA = (iso["Fr_int"] - np.log(f(iso["alpha"]))).loc[m].mean()
    return E, lnA


def forward_reconstruct(res, code, e_choice="Fr", optimize_A=False):
    """Forward-integrate alpha(T) for each beta with model `code`; return RMSE vs exp.
    If optimize_A, the single lnA that best reproduces all beta is fitted (E fixed)."""
    E, lnA = model_triplet(res, code, e_choice)
    if optimize_A:
        from scipy.optimize import minimize_scalar
        f = L.MODELS[code][0]
        data = []
        for pk in res["peaks"]:
            proc = pk["pk"]["proc"]; Tk = proc["Tk"].values; ae = proc["alpha"].values
            o = np.argsort(Tk); Tk = Tk[o]; ae = ae[o]; mm = (ae > 0.02) & (ae < 0.98)
            data.append((pk["beta"], Tk[mm], ae[mm]))

        def _r(la):
            Av = np.exp(la); e = []
            for beta, Tg, ae in data:
                e.append(np.sqrt(np.mean((L.forward_alpha(beta / 60.0, Tg, E, Av, code) - ae) ** 2)))
            return np.mean(e)
        # the upper bound must track Ea: lnA ~ E/(R*T_peak); high-Ea samples need >90
        hi = max(95.0, E / (L.R * 560.0) + 25.0)
        lnA = float(minimize_scalar(_r, bounds=(10, hi), method="bounded").x)
    A = np.exp(lnA)
    per = {}
    rmses = []
    for pk in res["peaks"]:
        beta = pk["beta"]
        proc = pk["pk"]["proc"]
        Tk = proc["Tk"].values
        aexp = proc["alpha"].values
        order = np.argsort(Tk)
        Tk = Tk[order]; aexp = aexp[order]
        # restrict to 0.02..0.98 window for fair comparison
        mm = (aexp > 0.02) & (aexp < 0.98)
        Tg = Tk[mm]
        if len(Tg) < 10:
            continue
        amod = L.forward_alpha(beta / 60.0, Tg, E, A, code)
        rmse = float(np.sqrt(np.mean((amod - aexp[mm]) ** 2)))
        rmses.append(rmse)
        per[beta] = dict(Tg=Tg, aexp=aexp[mm], amod=amod, rmse=rmse)
    return dict(code=code, E=E / 1000.0, lnA=lnA, A=A,
                rmse=float(np.mean(rmses)) if rmses else np.nan, per=per)


def discriminate(res, candidates=("F1", "A2", "A3", "A4", "R3", "D3", "F2")):
    """Rank candidate mechanisms by forward-reconstruction RMSE (inverse modelling)."""
    out = [forward_reconstruct(res, c) for c in candidates]
    out = sorted(out, key=lambda d: d["rmse"])
    return out


def summarize(res):
    s = res["sample"]; iso = res["iso"]
    print(f"\n{'='*64}\n  {L.SAMPLE_NAME[s]} ({s})\n{'='*64}")
    print("Peaks (center, C):", {b: round(res['peaks_d'][b]['center'], 1) for b in sorted(res['peaks_d'])})
    cols = ["alpha", "T_mean_C", "Fr_Ea", "KAS_Ea", "FWO_Ea", "St_Ea", "Vy_Ea", "Fr_R2"]
    print(iso[cols].round(2).to_string(index=False))
    for m in ["Fr", "KAS", "FWO", "St", "Vy"]:
        col = f"{m}_Ea"
        v = iso[col].dropna()
        print(f"  {m:>3}: Ea = {v.mean():6.1f} +/- {v.std():4.1f} kJ/mol  (range {v.min():.0f}-{v.max():.0f})")
    print(f"\n  Mechanism (Criado + KCE): {res['mech']} - {L.MODELS[res['mech']][2]}")
    print("  Top-6 models by Criado deviation (lower err = better):")
    for i in range(min(6, len(res["rank"]))):
        r = res["rank"].iloc[i]
        print(f"    {r['code']:>2}  err={r['err']:.5f}  KCE_R2={r['kce_r2']:.4f}")
    print(f"  ln A (Friedman, mech. {res['mech']}): {iso['lnA'].mean():.2f}  (A~{np.exp(iso['lnA'].mean()):.2e} 1/s)")


if __name__ == "__main__":
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    for sample in ["MelBar", "Mel", "MelTbar"]:
        res = analyze(sample)
        summarize(res)
        res["iso"].to_csv(os.path.join(DATA_OUT_DIR, f"kinetics_{sample}.csv"), index=False)
        res["rank"].to_csv(os.path.join(DATA_OUT_DIR, f"criado_rank_{sample}.csv"), index=False)
