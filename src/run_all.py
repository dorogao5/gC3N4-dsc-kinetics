"""
Run the full analysis pipeline end to end and write all results to results/.

  1. extract.py      - baseline correction + Fraser-Suzuki deconvolution; deconvolution figures
  2. kinetics.py     - isoconversional Ea(alpha), Criado master plot, compensation effect
  3. deviation.py    - model-discrimination table (all 15 models)
  4. figures.py      - kinetic figures
  5. export_data.py  - post-processed XY data (one CSV per figure)

Usage:  python src/run_all.py
"""
import os
import dsc_lib as L
import extract as X
import kinetics as K
import deviation as D
import figures as F
import export_data as E

SAMPLES = ["MelBar", "Mel", "MelTbar"]


def main():
    os.makedirs(L.FIG_DIR, exist_ok=True)
    os.makedirs(L.DATA_OUT_DIR, exist_ok=True)

    print(">> 1/5 extraction + deconvolution figures")
    extracted = {}
    for s in SAMPLES:
        peaks = X.extract_sample(s)
        extracted[s] = peaks
        X.figure_deconv(s, peaks, os.path.join(L.FIG_DIR, f"fig_deconv_{s}.png"))
        print(f"   {L.SAMPLE_NAME[s]}: {len(peaks)} heating rates")

    print(">> 2/5 isoconversional kinetics")
    results = {}
    for s in SAMPLES:
        res = K.analyze(s)
        results[s] = res
        res["iso"].to_csv(os.path.join(L.DATA_OUT_DIR, f"kinetics_{s}.csv"), index=False)
        res["rank"].to_csv(os.path.join(L.DATA_OUT_DIR, f"criado_rank_{s}.csv"), index=False)
        ea = res["iso"]["Fr_Ea"].mean()
        print(f"   {L.SAMPLE_NAME[s]}: mechanism {F.FINAL_MECH[s]}, mean Ea(Friedman) = {ea:.0f} kJ/mol")

    print(">> 3/5 model-discrimination table")
    for s in SAMPLES:
        D.table(s).to_csv(os.path.join(L.DATA_OUT_DIR, f"deviation_{s}.csv"), index=False)

    print(">> 4/5 figures")
    for s in SAMPLES:
        res = results[s]
        F.fig_alpha_rate(res); F.fig_ea(res); F.fig_friedman_lines(res)
        F.fig_criado(res); F.fig_forward(res); F.fig_kce(res)
    F.fig_compare(list(results.values()))

    print(">> 5/5 post-processed XY data")
    for s in SAMPLES:
        E.export_deconv(s, extracted[s])
        E.export_alpha_rate(s, results[s])
        E.export_Ea(s, results[s])
        E.export_friedman(s, results[s])
        E.export_criado(s, results[s])
        E.export_forward(s, results[s])
        E.export_kce(s, results[s])
    E.export_compare(results)

    nfig = len([f for f in os.listdir(L.FIG_DIR) if f.endswith(".png")])
    ndat = len([f for f in os.listdir(L.DATA_OUT_DIR) if f.endswith(".csv")])
    print(f"\nDone. results/figures: {nfig} PNG, results/data: {ndat} CSV")


if __name__ == "__main__":
    main()
