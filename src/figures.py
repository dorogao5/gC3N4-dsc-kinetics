"""Figures for the DSC kinetic analysis of g-C3N4 formation."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
import dsc_lib as L
import kinetics as K

OUT = L.FIG_DIR
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "savefig.dpi": 200})
CB = L.COLORS

TITLE = {"Mel": "Melamine (~550 C peak, melem->melon->g-C$_3$N$_4$)",
         "MelBar": "Melamine barbiturate (central ~350-370 C peak, polycondensation->g-C$_3$N$_4$)",
         "MelTbar": "Melamine thiobarbiturate (~300 C peak, polycondensation->g-C$_3$N$_4$)"}
FINAL_MECH = {"MelBar": "R3", "Mel": "F1", "MelTbar": "R3"}
RUS = L.SAMPLE_NAME


def fig_alpha_rate(res):
    s = res["sample"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for pk in res["peaks"]:
        b = pk["beta"]; proc = pk["pk"]["proc"]
        Tc = proc["Tk"].values - 273.15
        axs[0].plot(Tc, proc["alpha"], color=CB[b], lw=2, label=f"{b:.0f} K/min")
        axs[1].plot(Tc, proc["rate"] * 1e3, color=CB[b], lw=2, label=f"{b:.0f} K/min")
    axs[0].set_ylabel("conversion alpha"); axs[0].set_title("Conversion curves alpha(T)")
    axs[1].set_ylabel("dalpha/dt x10^3, 1/s"); axs[1].set_title("Reaction rate")
    for ax in axs:
        ax.set_xlabel("T, C"); ax.legend()
    axs[0].axhline(0.1, ls="--", color="0.6", lw=0.8); axs[0].axhline(0.9, ls="--", color="0.6", lw=0.8)
    fig.suptitle(TITLE[s]); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_alpha_rate_{s}.png"); plt.close(fig)


def fig_ea(res):
    s = res["sample"]; iso = res["iso"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = [("Fr_Ea", "Friedman", "o-", "#000000"),
              ("Vy_Ea", "Vyazovkin (nonlin.)", "s-", "#d62728"),
              ("KAS_Ea", "KAS", "^-", "#1f77b4"),
              ("FWO_Ea", "OFW (Doyle)", "v-", "#2ca02c"),
              ("St_Ea", "Starink", "d--", "#9467bd")]
    for col, lab, st, c in styles:
        if col in iso:
            ax.plot(iso["alpha"], iso[col], st, color=c, label=lab, ms=5, lw=1.4)
    ax.set_xlabel("conversion alpha"); ax.set_ylabel("E$_a$, kJ/mol")
    ax.set_title(f"Activation energy E$_a$(alpha) - {RUS[s]}")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_Ea_{s}.png"); plt.close(fig)


def fig_friedman_lines(res):
    s = res["sample"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    alphas = np.arange(0.1, 0.91, 0.1)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for a, col in zip(alphas, cmap):
        invT, lnr = [], []
        for pk in res["peaks"]:
            fT, fr = pk["interp"]
            T = float(fT(a)); r = float(fr(a))
            if np.isfinite(T) and np.isfinite(r) and r > 0:
                invT.append(1000.0 / T); lnr.append(np.log(r))
        if len(invT) >= 2:
            invT = np.array(invT); lnr = np.array(lnr)
            ax.plot(invT, lnr, "o", color=col, ms=6)
            sl, ic, *_ = linregress(invT, lnr)
            xx = np.array([invT.min(), invT.max()])
            ax.plot(xx, sl * xx + ic, "-", color=col, lw=1,
                    label=f"alpha={a:.1f}" if a in (0.1, 0.5, 0.9) else None)
    ax.set_xlabel("1000/T, 1/K"); ax.set_ylabel("ln(dalpha/dt)")
    ax.set_title(f"Friedman isoconversional lines - {RUS[s]}")
    ax.legend(title="labelled alpha = 0.1/0.5/0.9"); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_friedman_{s}.png"); plt.close(fig)


CRIADO_ORDER = ["F1", "F2", "F3", "D1", "D2", "D3", "D4", "R2", "R3",
                "P2", "P3", "P4", "A2", "A3", "A4"]


def fig_criado(res):
    s = res["sample"]; ydf = res["ydf"]; mech = FINAL_MECH[s]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    grid = np.arange(0.1, 0.901, 0.005)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(CRIADO_ORDER)))
    for code, c in zip(CRIADO_ORDER, cmap):       # all 15 theoretical curves
        f, g, _ = L.MODELS[code]
        yth = (f(grid) * g(grid)) / (f(0.5) * g(0.5))
        if code == mech:
            ax.plot(grid, yth, "-", color="k", lw=2.8, zorder=4, label=f"{code} (selected)")
        else:
            ax.plot(grid, yth, "-", color=c, lw=1.1, alpha=0.85, label=code)
    for b in sorted(ydf["beta"].unique()):         # experiment on top
        sub = ydf[ydf["beta"] == b].sort_values("alpha")
        ax.plot(sub["alpha"], sub["y"], "o", color=CB[b], ms=5.5, zorder=6,
                markeredgecolor="0.2", markeredgewidth=0.4, label=f"exp. {b:.0f} K/min")
    ax.set_xlabel("conversion alpha"); ax.set_ylabel("y(alpha) (normalized at alpha = 0.5)")
    ax.set_ylim(0, 2.0); ax.set_title(f"Criado master-plot - all 15 models and experiment ({RUS[s]})")
    ax.legend(ncol=1, fontsize=8, loc="center left", bbox_to_anchor=(1.005, 0.5))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_criado_{s}.png"); plt.close(fig)


def fig_forward(res):
    s = res["sample"]; code = FINAL_MECH[s]
    fr = K.forward_reconstruct(res, code, optimize_A=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for pk in res["peaks"]:
        b = pk["beta"]
        if b not in fr["per"]:
            continue
        d = fr["per"][b]
        Tc = d["Tg"] - 273.15
        ax.plot(Tc, d["aexp"], "o", color=CB[b], ms=3, alpha=0.5,
                label=f"exp. {b:.0f} K/min")
        ax.plot(Tc, d["amod"], "-", color=CB[b], lw=2,
                label=f"model {code} {b:.0f} K/min (RMSE={d['rmse']:.3f})")
    ax.set_xlabel("T, C"); ax.set_ylabel("conversion alpha")
    ax.set_title(f"Forward reconstruction (consistency check): alpha(T) exp. and model {code} ({RUS[s]})\n"
                 f"E={fr['E']:.0f} kJ/mol, lnA={fr['lnA']:.1f}")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_forward_{s}.png"); plt.close(fig)


def fig_kce(res):
    s = res["sample"]; mech = FINAL_MECH[s]
    comp = L.compensation(res["iso"], mech)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {"Friedman": ("o", "#000000"), "KAS": ("^", "#1f77b4"),
              "OFW": ("v", "#2ca02c"), "Starink": ("D", "#9467bd")}
    for name, d in comp.items():
        mk, c = styles[name]
        ax.scatter(d["Ea"], d["lnA"], marker=mk, color=c, s=26, zorder=3)
        xx = np.array([d["Ea"].min(), d["Ea"].max()])
        ax.plot(xx, d["a"] * xx + d["b"], "-", color=c, lw=1.3,
                label=f"{name}: ln A = {d['a']:.3f}*E$_a$ {d['b']:+.1f}; R2={d['r2']:.3f}")
    ax.set_xlabel("E$_a$, kJ/mol"); ax.set_ylabel("ln A")
    ax.set_title(f"Kinetic compensation effect by method\n{RUS[s]}, model {mech}", fontsize=11.5)
    ax.legend(fontsize=8.5); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_kce_{s}.png"); plt.close(fig)


CMP_LABEL = {"MelTbar": "Melamine thiobarbiturate (~300 C)",
             "MelBar": "Melamine barbiturate (~360 C)",
             "Mel": "Melamine (~540 C)"}
CMP_COLOR = {"MelTbar": "#1f77b4", "MelBar": "#e377c2", "Mel": "#8c564b"}


def fig_compare(results):
    """results: list of analyzed dicts (any order); plotted low->high T."""
    order = sorted(results, key=lambda r: float(r["peaks"][0]["pk"]["proc"]["Tk"].mean()))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    for res in order:
        s = res["sample"]; lab = CMP_LABEL[s]; c = CMP_COLOR[s]
        for pk in res["peaks"]:
            if pk["beta"] != 3.0:
                continue
            proc = pk["pk"]["proc"]
            axs[0].plot(proc["Tk"].values - 273.15, proc["alpha"], color=c, lw=2.2, label=lab)
        axs[1].plot(res["iso"]["alpha"], res["iso"]["Fr_Ea"], "o-", color=c, ms=4, label=lab)
    axs[0].set_xlabel("T, C"); axs[0].set_ylabel("conversion alpha"); axs[0].legend(fontsize=9)
    axs[0].set_title("g-C$_3$N$_4$ formation: alpha(T) at 3 K/min")
    axs[1].set_xlabel("conversion alpha"); axs[1].set_ylabel("E$_a$ (Friedman), kJ/mol")
    axs[1].legend(fontsize=9)
    axs[1].set_title("E$_a$(alpha): three precursors")
    fig.suptitle("Comparison of g-C$_3$N$_4$ formation routes from three precursors")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_compare.png"); plt.close(fig)


if __name__ == "__main__":
    results = {}
    for s in ["MelBar", "Mel", "MelTbar"]:
        res = K.analyze(s)
        results[s] = res
        fig_alpha_rate(res); fig_ea(res); fig_friedman_lines(res)
        fig_criado(res); fig_forward(res); fig_kce(res)
        print(f"{s}: figures done (mech={FINAL_MECH[s]})")
    fig_compare(list(results.values()))
    print("comparison figure done")
