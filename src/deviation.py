"""
Model-discrimination table for all 15 solid-state models, per sample:
  - delta_Criado_logMSE : deviation of the experimental Criado master-plot from each
                          model (mean squared error in log space; the discriminating quantity)
  - RMSE_forward_recon  : RMSE of the forward alpha(T) reconstruction with E fixed and lnA
                          fitted (a consistency check; nearly non-discriminating)
"""
import os
import numpy as np
import pandas as pd
import dsc_lib as L
import kinetics as K

DATA_OUT_DIR = L.DATA_OUT_DIR
ORDER = ["F1", "F2", "F3", "D1", "D2", "D3", "D4", "R2", "R3",
         "P2", "P3", "P4", "A2", "A3", "A4"]


def table(sample):
    res = K.analyze(sample)
    ydf = res["ydf"]
    rows = []
    for c in ORDER:
        dev = L.criado_error(ydf, c)
        try:
            rmse = K.forward_reconstruct(res, c, optimize_A=True)["rmse"]
        except Exception:
            rmse = np.nan
        rows.append({"model": c, "delta_Criado_logMSE": dev, "RMSE_forward_recon": rmse})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    for s in ["MelBar", "Mel", "MelTbar"]:
        df = table(s)
        df.to_csv(os.path.join(DATA_OUT_DIR, f"deviation_{s}.csv"), index=False)
        print(f"=== {L.SAMPLE_NAME[s]} ({s}) ===")
        for _, r in df.iterrows():
            print(f"  {r['model']:>2}: delta_Criado={r['delta_Criado_logMSE']:.5f}  "
                  f"RMSE_recon={r['RMSE_forward_recon']:.4f}")
