"""
Export the post-processed XY data behind every figure as plain CSV (one file per
figure), so the curves can be re-plotted in Origin / Excel / etc.

Wide layout with padded, adjacent X/Y column pairs: in Origin just highlight two
neighbouring columns and plot. Shorter series are padded with empty cells.
Common-grid figures (Ea(alpha), Criado theory) keep a single shared X column.

Outputs -> results/data/<name>.csv  (mirrors the figure base names)
"""
import os
import numpy as np
import pandas as pd
import dsc_lib as L
import kinetics as K
import extract as X
import figures as F

DDIR = L.DATA_OUT_DIR
os.makedirs(DDIR, exist_ok=True)

SAMPLES = ["MelBar", "Mel", "MelTbar"]


def write_wide(name, cols, header_note=None):
    """cols: list of (column_title, 1D-array). Pads to max length, writes CSV."""
    n = max(len(a) for _, a in cols)
    data = {}
    for title, a in cols:
        a = np.asarray(a, dtype=float)
        if len(a) < n:
            a = np.concatenate([a, np.full(n - len(a), np.nan)])
        data[title] = a
    df = pd.DataFrame(data)
    path = os.path.join(DDIR, name)
    with open(path, "w", encoding="utf-8-sig") as fh:
        if header_note:
            fh.write(f"# {header_note}\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    return path


def write_long(name, df, header_note=None):
    path = os.path.join(DDIR, name)
    with open(path, "w", encoding="utf-8-sig") as fh:
        if header_note:
            fh.write(f"# {header_note}\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    return path


# ----------------------------------------------------------------------
def export_deconv(sample, peaks):
    cols = []
    for b in sorted(peaks):
        pk = peaks[b]
        T = np.asarray(pk["Tc"], float)
        cols.append((f"T_C_{b:.0f}K", T))
        cols.append((f"DSC_smoothed_{b:.0f}K", np.asarray(pk["raw"], float)))
        cols.append((f"baseline_{b:.0f}K", np.asarray(pk["base"], float)))
        if sample == "MelBar":
            fit = pk["base"] + L.multi_fs(T, np.concatenate(pk["fs"]))
            central = pk["base"] + L.fraser_suzuki(T, *pk["central"])
            shoulder = pk["base"] + L.fraser_suzuki(T, *pk["shoulder"])
            cols.append((f"FS_sum_{b:.0f}K", fit))
            cols.append((f"central_{b:.0f}K", central))
            cols.append((f"shoulder_{b:.0f}K", shoulder))
        else:
            cols.append((f"peak_minus_baseline_{b:.0f}K", np.asarray(pk["sig"], float)))
            cols.append((f"FS_fit_{b:.0f}K", pk["base"] + L.fraser_suzuki(T, *pk["fs"][0])))
    write_wide(f"deconv_{sample}.csv", cols,
               "DSC deconvolution: T (C), smoothed signal, baseline, components (mW/mg).")


def export_alpha_rate(sample, res):
    cols = []
    for pk in res["peaks"]:
        b = pk["beta"]; proc = pk["pk"]["proc"]
        T = proc["Tk"].values - 273.15
        cols.append((f"T_C_{b:.0f}K", T))
        cols.append((f"alpha_{b:.0f}K", proc["alpha"].values))
        cols.append((f"rate_dadt_1/s_{b:.0f}K", proc["rate"].values))
    write_wide(f"alpha_rate_{sample}.csv", cols,
               "Conversion and rate: T (C), alpha (-), dalpha/dt (1/s) for each beta.")


def export_Ea(sample, res):
    iso = res["iso"]
    cols = [("alpha", iso["alpha"].values)]
    for c, t in [("Fr_Ea", "Friedman"), ("Vy_Ea", "Vyazovkin"), ("KAS_Ea", "KAS"),
                 ("FWO_Ea", "OFW"), ("St_Ea", "Starink")]:
        if c in iso:
            cols.append((f"Ea_{t}_kJ/mol", iso[c].values))
    for c, t in [("Fr_R2", "Friedman"), ("KAS_R2", "KAS"), ("FWO_R2", "OFW"), ("St_R2", "Starink")]:
        if c in iso:
            cols.append((f"R2_{t}", iso[c].values))
    write_wide(f"Ea_{sample}.csv", cols,
               "Activation energy Ea(alpha) by method, kJ/mol, and regression R-squared.")


def export_friedman(sample, res):
    rows = []
    for a in np.arange(0.1, 0.91, 0.1):
        invT, lnr, betas, Ts = [], [], [], []
        for pk in res["peaks"]:
            fT, fr = pk["interp"]
            T = float(fT(a)); r = float(fr(a))
            if np.isfinite(T) and np.isfinite(r) and r > 0:
                invT.append(1000.0 / T); lnr.append(np.log(r))
                betas.append(pk["beta"]); Ts.append(T)
        if len(invT) < 2:
            continue
        from scipy.stats import linregress
        sl, ic, *_ = linregress(invT, lnr)
        for b, t, x, y in zip(betas, Ts, invT, lnr):
            rows.append({"alpha": round(a, 2), "beta_K/min": b, "T_K": t,
                         "inv1000T_1/K": x, "ln_dadt": y,
                         "fit_slope": sl, "fit_intercept": ic,
                         "Ea_kJ/mol": -sl * L.R})
    write_long(f"friedman_{sample}.csv", pd.DataFrame(rows),
               "Friedman isoconversional lines: points (1000/T, ln dalpha/dt) and linear fit per alpha.")


def export_criado(sample, res):
    ydf = res["ydf"]
    grid = np.arange(0.1, 0.901, 0.005)
    cols = [("alpha_theory", grid)]
    for code in F.CRIADO_ORDER:
        f, g, _ = L.MODELS[code]
        yth = (f(grid) * g(grid)) / (f(0.5) * g(0.5))
        cols.append((f"y_{code}", yth))
    for b in sorted(ydf["beta"].unique()):
        sub = ydf[ydf["beta"] == b].sort_values("alpha")
        cols.append((f"alpha_exp_{b:.0f}K", sub["alpha"].values))
        cols.append((f"y_exp_{b:.0f}K", sub["y"].values))
    write_wide(f"criado_{sample}.csv", cols,
               "Criado master-plot: y(alpha) of 15 theoretical models (common grid) and experiment by beta. "
               f"Selected model: {F.FINAL_MECH[sample]}.")


def export_forward(sample, res):
    code = F.FINAL_MECH[sample]
    fr = K.forward_reconstruct(res, code, optimize_A=True)
    cols = []
    for pk in res["peaks"]:
        b = pk["beta"]
        if b not in fr["per"]:
            continue
        d = fr["per"][b]
        cols.append((f"T_C_{b:.0f}K", d["Tg"] - 273.15))
        cols.append((f"alpha_exp_{b:.0f}K", d["aexp"]))
        cols.append((f"alpha_model_{b:.0f}K", d["amod"]))
    write_wide(f"forward_{sample}.csv", cols,
               f"Forward reconstruction (consistency check), model {code}: E={fr['E']:.0f} kJ/mol, lnA={fr['lnA']:.1f}. "
               "T (C), alpha experimental and modelled for each beta.")


def export_kce(sample, res):
    mech = F.FINAL_MECH[sample]
    comp = L.compensation(res["iso"], mech)
    cols = []
    fit_rows = []
    for name, d in comp.items():
        cols.append((f"Ea_{name}_kJ/mol", d["Ea"]))
        cols.append((f"lnA_{name}", d["lnA"]))
        fit_rows.append({"method": name, "slope_a": d["a"], "intercept_b": d["b"], "R2": d["r2"]})
    write_wide(f"kce_{sample}.csv", cols,
               f"Kinetic compensation effect (model {mech}): points (Ea, lnA) by method.")
    write_long(f"kce_fit_{sample}.csv", pd.DataFrame(fit_rows),
               "Compensation line parameters lnA = a*Ea + b for each method.")


def export_compare(results):
    order = sorted(results.values(),
                   key=lambda r: float(r["peaks"][0]["pk"]["proc"]["Tk"].mean()))
    cols_a, cols_e = [], []
    for res in order:
        s = res["sample"]; lab = {"Mel": "melamine", "MelBar": "mel_barbiturate",
                                  "MelTbar": "mel_thiobarbiturate"}[s]
        for pk in res["peaks"]:
            if pk["beta"] != 3.0:
                continue
            proc = pk["pk"]["proc"]
            cols_a.append((f"T_C_{lab}", proc["Tk"].values - 273.15))
            cols_a.append((f"alpha_{lab}", proc["alpha"].values))
        cols_e.append((f"alpha_{lab}", res["iso"]["alpha"].values))
        cols_e.append((f"Ea_Friedman_{lab}", res["iso"]["Fr_Ea"].values))
    write_wide("compare_alpha_3Kmin.csv", cols_a,
               "Comparison of alpha(T) at 3 K/min for the three precursors.")
    write_wide("compare_Ea.csv", cols_e,
               "Comparison of Ea(alpha) (Friedman) for the three precursors, kJ/mol.")


if __name__ == "__main__":
    results = {}
    for s in SAMPLES:
        peaks = X.extract_sample(s)
        res = K.analyze(s)
        results[s] = res
        export_deconv(s, peaks)
        export_alpha_rate(s, res)
        export_Ea(s, res)
        export_friedman(s, res)
        export_criado(s, res)
        export_forward(s, res)
        export_kce(s, res)
        print(f"{s}: data exported")
    export_compare(results)
    print("compare data exported")
    print("files:", len(os.listdir(DDIR)))
