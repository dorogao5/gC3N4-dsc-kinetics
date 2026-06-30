"""
Core library for the DSC thermal-kinetic analysis of g-C3N4 formation.

- NETZSCH ASCII loader (heating segment only)
- Baseline / tilt correction
- Fraser-Suzuki asymmetric peak model + multi-peak deconvolution
- Conversion (alpha) and rate (dalpha/dt) extraction
- Isoconversional kinetics: Friedman, KAS, OFW (Doyle), Starink, Vyazovkin (advanced)
- Criado master-plot mechanism selection + kinetic compensation effect (KCE)
- Forward (inverse-modelling) ODE reconstruction

Sign convention: the raw NETZSCH DSC column is kept as-is; in these files
positive = endothermic-up (melamine melting shows as a large positive spike).
For kinetic processing each target peak is converted to a positive area after
local baseline subtraction (the sign of the heat effect is reported separately).
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import linregress

R = 8.314  # J/(mol K)

# ----------------------------------------------------------------------
# Repository paths (resolved relative to this file, so the project is portable)
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_OUT_DIR = os.path.join(RESULTS_DIR, "data")

# Sample -> {heating rate (K/min): raw data file}, relative to data/<sample dir>/
FILES = {
    "Mel": {
        1.0: "melamine_1K.txt",
        3.0: "melamine_3K.txt",
        5.0: "melamine_5K.txt",
    },
    "MelBar": {
        1.0: "melamine-barbiturate_1K.txt",
        3.0: "melamine-barbiturate_3K.txt",
        5.0: "melamine-barbiturate_5K.txt",
    },
    "MelTbar": {
        1.0: "melamine-thiobarbiturate_1K.txt",
        3.0: "melamine-thiobarbiturate_3K.txt",
        5.0: "melamine-thiobarbiturate_5K.txt",
    },
}
SAMPLE_DIR = {
    "Mel": os.path.join(DATA_DIR, "melamine"),
    "MelBar": os.path.join(DATA_DIR, "melamine-barbiturate"),
    "MelTbar": os.path.join(DATA_DIR, "melamine-thiobarbiturate"),
}
SAMPLE_NAME = {
    "Mel": "melamine",
    "MelBar": "melamine barbiturate",
    "MelTbar": "melamine thiobarbiturate",
}
COLORS = {1.0: "#1f77b4", 3.0: "#2ca02c", 5.0: "#ff7f0e"}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_seg1(fp):
    """Load a NETZSCH ASCII export, return the heating segment (Segment==1) only."""
    with open(fp, encoding="cp1251") as f:
        lines = f.readlines()
    hr = next(i for i, l in enumerate(lines)
              if l.startswith("##") and "Temp" in l and "DSC" in l)
    df = pd.read_csv(fp, skiprows=hr, sep=";", engine="python", encoding="cp1251")
    df.columns = [c.replace("##", "").strip() for c in df.columns]
    cols = df.columns
    tc = [c for c in cols if "Temp" in c][0]
    dc = [c for c in cols if "DSC" in c][0]
    ti = [c for c in cols if "Time" in c][0]
    seg = [c for c in cols if "Segment" in c]
    out = pd.DataFrame({
        "Tc": df[tc].astype(float),
        "tmin": df[ti].astype(float),
        "DSC": df[dc].astype(float),
    })
    out["seg"] = df[seg[0]].astype(float) if seg else 1.0
    out = out[out["seg"] == 1].reset_index(drop=True)
    out["Tk"] = out["Tc"] + 273.15
    out["tsec"] = out["tmin"] * 60.0
    return out


def load_sample(sample):
    """Return {beta: dataframe} for a sample name (Mel / MelBar / MelTbar)."""
    d = SAMPLE_DIR[sample]
    return {b: load_seg1(os.path.join(d, fn)) for b, fn in FILES[sample].items()}


# ----------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------
def linear_baseline(T, y, t0, t1, navg=15):
    """Straight line between the mean signal at the two anchor temperatures."""
    i0 = np.argmin(np.abs(T - t0))
    i1 = np.argmin(np.abs(T - t1))
    y0 = np.mean(y[max(0, i0 - navg):i0 + navg + 1])
    y1 = np.mean(y[max(0, i1 - navg):i1 + navg + 1])
    return y0 + (y1 - y0) * (T - T[i0]) / (T[i1] - T[i0])


def tangential_sigmoidal_baseline(T, y, t0, t1, navg=15, n_iter=60):
    """
    Area-progress (sigmoidal/tangential) baseline: the baseline blends from the
    pre-peak tangent level to the post-peak tangent level in proportion to the
    fraction of peak area already developed. Standard for solid-state DSC peaks
    where the heat-capacity level changes across the transition.
    """
    i0 = np.argmin(np.abs(T - t0))
    i1 = np.argmin(np.abs(T - t1))
    y0 = np.mean(y[max(0, i0 - navg):i0 + navg + 1])
    y1 = np.mean(y[max(0, i1 - navg):i1 + navg + 1])
    base = y0 + (y1 - y0) * (T - T[i0]) / (T[i1] - T[i0])  # start from linear
    for _ in range(n_iter):
        corr = y - base
        corr = np.clip(corr, 0, None)
        area = cumulative_trapezoid(corr, T, initial=0)
        frac = area / area[-1] if area[-1] > 0 else np.zeros_like(area)
        base = y0 + (y1 - y0) * frac
    return base


# ----------------------------------------------------------------------
# Fraser-Suzuki asymmetric peak (Perejon / Sanchez-Jimenez parameterization)
# ----------------------------------------------------------------------
def fraser_suzuki(T, h, p, w, s):
    """
    Asymmetric Fraser-Suzuki peak.
      h : amplitude (height)
      p : position of maximum
      w : width parameter (FWHM-related)
      s : asymmetry (s -> 0 => Gaussian)
    Returns 0 outside the support 1 + 2 s (T-p)/w > 0.
    """
    T = np.asarray(T, dtype=float)
    if abs(s) < 1e-6:
        return h * np.exp(-np.log(2.0) * (2.0 * (T - p) / w) ** 2)
    arg = 1.0 + 2.0 * s * (T - p) / w
    out = np.zeros_like(T)
    m = arg > 1e-12
    out[m] = h * np.exp(-np.log(2.0) / s**2 * (np.log(arg[m])) ** 2)
    return out


def multi_fs(T, params):
    """Sum of N Fraser-Suzuki peaks. params = [h1,p1,w1,s1, h2,...]."""
    T = np.asarray(T, dtype=float)
    y = np.zeros_like(T)
    for i in range(0, len(params), 4):
        y = y + fraser_suzuki(T, *params[i:i + 4])
    return y


def fit_multi_fs(T, y, p0, lower, upper, max_nfev=20000):
    """Bounded least-squares multi-FS fit. Returns (popt, r2, resid_rms)."""
    def resid(p):
        return multi_fs(T, p) - y
    res = least_squares(resid, p0, bounds=(lower, upper),
                        max_nfev=max_nfev, xtol=1e-12, ftol=1e-12)
    yhat = multi_fs(T, res.x)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rms = np.sqrt(np.mean((y - yhat) ** 2))
    return res.x, r2, rms


# ----------------------------------------------------------------------
# Solid-state reaction models (conversion functions)
# ----------------------------------------------------------------------
def build_models():
    def F1_f(x): return (1 - x)
    def F1_g(x): return -np.log(1 - x)
    def F2_f(x): return (1 - x) ** 2
    def F2_g(x): return 1 / (1 - x) - 1
    def F3_f(x): return (1 - x) ** 3
    def F3_g(x): return 0.5 * ((1 / (1 - x) ** 2) - 1)
    def D1_f(x): return 1 / (2 * x)
    def D1_g(x): return x ** 2
    def D2_f(x): return 1 / (-np.log(1 - x))
    def D2_g(x): return (1 - x) * np.log(1 - x) + x
    def D3_f(x): return (3 / 2) * ((1 - x) ** (2 / 3)) / (1 - (1 - x) ** (1 / 3))
    def D3_g(x): return (1 - (1 - x) ** (1 / 3)) ** 2
    def D4_f(x): return (3 / 2) * ((1 - x) ** (1 / 3)) / (1 - (1 - x) ** (1 / 3))
    def D4_g(x): return 1 - (2 * x) / 3 - (1 - x) ** (2 / 3)
    def R2_f(x): return 2 * (1 - x) ** 0.5
    def R2_g(x): return 1 - (1 - x) ** 0.5
    def R3_f(x): return 3 * (1 - x) ** (2 / 3)
    def R3_g(x): return 1 - (1 - x) ** (1 / 3)
    def P2_f(x): return 2 * (x ** 0.5)
    def P2_g(x): return x ** 0.5
    def P3_f(x): return 3 * (x ** (2 / 3))
    def P3_g(x): return x ** (1 / 3)
    def P4_f(x): return 4 * (x ** (3 / 4))
    def P4_g(x): return x ** (1 / 4)
    def A2_f(x): return 2 * (1 - x) * (-np.log(1 - x)) ** 0.5
    def A2_g(x): return (-np.log(1 - x)) ** 0.5
    def A3_f(x): return 3 * (1 - x) * (-np.log(1 - x)) ** (2 / 3)
    def A3_g(x): return (-np.log(1 - x)) ** (1 / 3)
    def A4_f(x): return 4 * (1 - x) * (-np.log(1 - x)) ** (3 / 4)
    def A4_g(x): return (-np.log(1 - x)) ** (1 / 4)
    return {
        "F1": (F1_f, F1_g, "First-order"),
        "F2": (F2_f, F2_g, "Second-order"),
        "F3": (F3_f, F3_g, "Third-order"),
        "D1": (D1_f, D1_g, "1-D diffusion"),
        "D2": (D2_f, D2_g, "2-D diffusion"),
        "D3": (D3_f, D3_g, "3-D diffusion (Jander)"),
        "D4": (D4_f, D4_g, "Ginstling-Brounshtein"),
        "R2": (R2_f, R2_g, "Contracting area (R2)"),
        "R3": (R3_f, R3_g, "Contracting volume (R3)"),
        "P2": (P2_f, P2_g, "Power law (P2)"),
        "P3": (P3_f, P3_g, "Power law (P3)"),
        "P4": (P4_f, P4_g, "Power law (P4)"),
        "A2": (A2_f, A2_g, "Avrami-Erofeev (A2)"),
        "A3": (A3_f, A3_g, "Avrami-Erofeev (A3)"),
        "A4": (A4_f, A4_g, "Avrami-Erofeev (A4)"),
    }


MODELS = build_models()


# ----------------------------------------------------------------------
# Conversion / rate extraction from an isolated single-process peak
# ----------------------------------------------------------------------
def make_alpha(T_K, tsec, sig):
    """
    From an isolated, baseline-free single-process peak signal sig(t) (>=0),
    build conversion alpha and rate dalpha/dt (1/s) by integrating over TIME.
      alpha(t) = cumulative_area(sig, tsec) / total_area
      rate(t)  = sig / total_area     [1/s]
    Returns a DataFrame sorted by time with strictly increasing alpha helpers.
    """
    sig = np.clip(np.asarray(sig, float), 0, None)
    area = cumulative_trapezoid(sig, tsec, initial=0.0)
    tot = area[-1]
    if tot <= 0:
        raise ValueError("zero peak area")
    alpha = area / tot
    rate = sig / tot
    return pd.DataFrame({"Tk": T_K, "tsec": tsec, "sig": sig,
                         "alpha": alpha, "rate": rate}), tot


def interp_alpha(df):
    """Monotonic interpolators T(alpha) and rate(alpha) from a processed peak."""
    tmp = df[["alpha", "Tk", "rate"]].copy()
    tmp["ar"] = tmp["alpha"].round(6)
    g = tmp.groupby("ar", as_index=False).agg(alpha=("alpha", "mean"),
                                              Tk=("Tk", "mean"), rate=("rate", "mean"))
    g = g.sort_values("alpha")
    a = g["alpha"].values
    mask = np.r_[True, np.diff(a) > 0]
    a = a[mask]; Tk = g["Tk"].values[mask]; rate = g["rate"].values[mask]
    fT = interp1d(a, Tk, bounds_error=False, fill_value=np.nan)
    fr = interp1d(a, rate, bounds_error=False, fill_value=np.nan)
    return fT, fr


# ----------------------------------------------------------------------
# Isoconversional methods  (peaks: list of dicts {beta, df})
# ----------------------------------------------------------------------
def _collect(peaks, a):
    """Return arrays (beta, T_K, rate) at fixed conversion a across heating rates."""
    betas, Ts, rates = [], [], []
    for pk in peaks:
        fT, fr = pk["interp"]
        T = float(fT(a)); r = float(fr(a))
        if np.isfinite(T) and np.isfinite(r) and r > 0:
            betas.append(pk["beta"]); Ts.append(T); rates.append(r)
    return np.array(betas), np.array(Ts), np.array(rates)


def isoconversional(peaks, alphas):
    """
    Compute Ea(alpha) by Friedman, KAS, FWO(Doyle), Starink for each conversion.
    Returns DataFrame with one row per alpha.
    Friedman intercept = ln(A f(alpha)) (used later for lnA / KCE).
    """
    rows = []
    for a in alphas:
        beta, T, rate = _collect(peaks, a)
        if len(T) < 3:
            continue
        invT = 1.0 / T
        row = {"alpha": float(a), "T_mean_C": float(np.mean(T) - 273.15)}
        # Friedman: ln(da/dt) = ln(A f) - E/RT   (rate already = da/dt)
        s, i, r, *_ = linregress(invT, np.log(rate))
        row["Fr_Ea"] = -s * R / 1000.0; row["Fr_int"] = i; row["Fr_R2"] = r**2
        # KAS: ln(beta/T^2) vs 1/T, slope -E/R
        s, i, r, *_ = linregress(invT, np.log(beta / T**2))
        row["KAS_Ea"] = -s * R / 1000.0; row["KAS_int"] = i; row["KAS_R2"] = r**2
        # FWO/OFW: ln(beta) vs 1/T, slope -1.0518 E/R  (Doyle)
        s, i, r, *_ = linregress(invT, np.log(beta))
        row["FWO_Ea"] = -s * R / 1.0518 / 1000.0; row["FWO_int"] = i; row["FWO_R2"] = r**2
        # Starink: ln(beta/T^1.92) vs 1/T, slope -1.0008 E/R
        s, i, r, *_ = linregress(invT, np.log(beta / T**1.92))
        row["St_Ea"] = -s * R / 1.0008 / 1000.0; row["St_int"] = i; row["St_R2"] = r**2
        rows.append(row)
    return pd.DataFrame(rows)


def compensation(iso, mech):
    """
    Kinetic compensation effect (ln A vs Ea) for EACH isoconversional method that
    yields a regression intercept. lnA(alpha) is recovered from each method's intercept
    using the chosen mechanism f(alpha)/g(alpha):
      Friedman : lnA = i_Fr - ln f
      KAS      : lnA = i_KAS + ln(Ea*g/R)
      OFW      : lnA = i_OFW + 5.331 + ln(R*g/Ea)
      Starink  : lnA = i_St  + ln(Ea*g/R)
    Returns dict method -> (lnA array, slope a, intercept b, R2).
    """
    f, g, _ = MODELS[mech]
    a = iso["alpha"].values
    fa = f(a); ga = g(a)
    out = {}
    defs = {
        "Friedman": iso["Fr_int"].values - np.log(fa),
        "KAS": iso["KAS_int"].values + np.log(iso["KAS_Ea"].values * 1000.0 * ga / R),
        "OFW": iso["FWO_int"].values + 5.331 + np.log(R * ga / (iso["FWO_Ea"].values * 1000.0)),
        "Starink": iso["St_int"].values + np.log(iso["St_Ea"].values * 1000.0 * ga / R),
    }
    Ea_method = {"Friedman": "Fr_Ea", "KAS": "KAS_Ea", "OFW": "FWO_Ea", "Starink": "St_Ea"}
    for name, lnA in defs.items():
        Ea = iso[Ea_method[name]].values
        m = np.isfinite(lnA) & np.isfinite(Ea)
        sl, ic, rr, *_ = linregress(Ea[m], lnA[m])
        out[name] = dict(Ea=Ea[m], lnA=lnA[m], a=float(sl), b=float(ic), r2=float(rr**2))
    return out


# ----------------------------------------------------------------------
# Criado master plot (y(alpha)) and mechanism selection
# ----------------------------------------------------------------------
def _temp_integral(E, T_lo, T_hi, n=40):
    """J(E) = int_{T_lo}^{T_hi} exp(-E/RT) dT  by trapezoid (advanced Vyazovkin)."""
    T = np.linspace(T_lo, T_hi, n)
    return np.trapezoid(np.exp(-E / (R * T)), T)


def vyazovkin_advanced(peaks, alphas):
    """
    Vyazovkin advanced (nonlinear) isoconversional method. For each conversion a,
    Ea minimizes  Phi(E) = sum_i sum_{j!=i} J(E,i) beta_j / (J(E,j) beta_i),
    with J integrated over the small interval [a-da, a] along each heating rate.
    Returns DataFrame(alpha, Vy_Ea_kJ).
    """
    from scipy.optimize import minimize_scalar
    rows = []
    da = alphas[1] - alphas[0]
    Tfun = []
    for pk in peaks:
        fT, _ = pk["interp"]
        Tfun.append((pk["beta"], fT))
    for a in alphas:
        segs = []
        ok = True
        for beta, fT in Tfun:
            T0 = float(fT(max(a - da, 1e-4))); T1 = float(fT(a))
            if not (np.isfinite(T0) and np.isfinite(T1) and T1 > T0):
                ok = False; break
            segs.append((beta, T0, T1))
        if not ok or len(segs) < 3:
            continue

        def phi(E):
            E = E * 1000.0
            J = [_temp_integral(E, T0, T1) for _, T0, T1 in segs]
            betas = [b for b, _, _ in segs]
            tot = 0.0
            for i in range(len(segs)):
                for j in range(len(segs)):
                    if i != j and J[j] > 0:
                        tot += (J[i] * betas[j]) / (J[j] * betas[i])
            return tot

        res = minimize_scalar(phi, bounds=(20, 600), method="bounded")
        rows.append({"alpha": float(a), "Vy_Ea": float(res.x)})
    return pd.DataFrame(rows)


def criado_y(peaks, grid):
    """Experimental y(a)=(T/T0.5)^2 (da/dt)/(da/dt)_0.5 for each heating rate."""
    out = []
    for pk in peaks:
        fT, fr = pk["interp"]
        T05 = float(fT(0.5)); r05 = float(fr(0.5))
        if not (np.isfinite(T05) and np.isfinite(r05) and r05 > 0):
            continue
        for a in grid:
            T = float(fT(a)); r = float(fr(a))
            if np.isfinite(T) and np.isfinite(r) and r > 0:
                out.append({"beta": pk["beta"], "alpha": float(a),
                            "y": (T / T05)**2 * (r / r05)})
    return pd.DataFrame(out)


def criado_error(ydf, code):
    f, g, _ = MODELS[code]
    a = ydf["alpha"].values; ye = ydf["y"].values
    yth = (f(a) * g(a)) / (f(0.5) * g(0.5))
    m = np.isfinite(ye) & np.isfinite(yth) & (ye > 0) & (yth > 0)
    if m.sum() < 8:
        return np.nan
    return float(np.mean((np.log(ye[m]) - np.log(yth[m]))**2))


def kce_r2(iso, code, method="Fr"):
    """R^2 of the compensation line lnA vs Ea using f(a) of `code` (Friedman route)."""
    f = MODELS[code][0]
    a = iso["alpha"].values
    Ea = iso[f"{method}_Ea"].values * 1000.0
    lnAf = iso[f"{method}_int"].values
    fv = f(a)
    m = np.isfinite(fv) & (fv > 0) & np.isfinite(Ea) & np.isfinite(lnAf)
    if m.sum() < 3:
        return np.nan
    lnA = lnAf[m] - np.log(fv[m])
    _, _, r, *_ = linregress(Ea[m], lnA)
    return float(r**2)


def rank_models(peaks, iso, grid, tie_tol=0.02):
    ydf = criado_y(peaks, grid)
    rows = [{"code": c, "err": criado_error(ydf, c), "kce_r2": kce_r2(iso, c)}
            for c in MODELS]
    rank = pd.DataFrame(rows).dropna(subset=["err"]).sort_values("err").reset_index(drop=True)
    best = rank["err"].iloc[0]
    cand = rank[rank["err"] <= best * (1 + tie_tol)].sort_values(
        ["kce_r2", "err"], ascending=[False, True])
    return cand["code"].iloc[0], rank, ydf


def lnA_from_friedman(iso, code):
    """lnA(a) = Friedman_intercept - ln f(a) for the chosen mechanism."""
    f = MODELS[code][0]
    a = iso["alpha"].values
    return iso["Fr_int"].values - np.log(f(a))


# ----------------------------------------------------------------------
# Forward (inverse-modelling) reconstruction:  da/dT = (A/beta) f(a) exp(-E/RT)
# ----------------------------------------------------------------------
def forward_alpha(beta_K_per_s, T_grid_K, E_Jmol, A_s, code):
    f = MODELS[code][0]

    def rhs(T, a):
        a = min(max(a[0], 0.0), 1 - 1e-9)
        return [(A_s / beta_K_per_s) * f(a) * np.exp(-E_Jmol / (R * T))]

    sol = solve_ivp(rhs, (T_grid_K[0], T_grid_K[-1]), [1e-9],
                    t_eval=T_grid_K, method="LSODA", rtol=1e-8, atol=1e-10, max_step=2.0)
    a = np.clip(sol.y[0], 0, 1)
    return a
