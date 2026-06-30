# DSC kinetics of graphitic carbon nitride (g-C3N4) formation

Differential scanning calorimetry (DSC) and non-isothermal kinetic analysis of
graphitic carbon nitride (g-C3N4) formation from three precursors:

- **melamine** (Mel),
- **melamine barbiturate** (MelBar),
- **melamine thiobarbiturate** (MelTbar).

Each precursor was measured at three heating rates (1, 3 and 5 K/min) on a
NETZSCH DSC 204F1 Phoenix. The repository contains the raw instrument exports and
a self-contained Python pipeline that reproduces every processed curve, figure and
kinetic parameter from those raw files.

## What the pipeline does

1. **Loading** – reads the heating segment of each NETZSCH ASCII export.
2. **Baseline correction and deconvolution** (`extract.py`)
   - melamine and melamine thiobarbiturate: a single asymmetric Fraser–Suzuki peak
     on a linear (tangential) baseline;
   - melamine barbiturate: a constrained, simultaneous two-peak Fraser–Suzuki fit
     across all heating rates (shared asymmetry and central-peak area fraction,
     Perejón / Sánchez-Jiménez 2011), with the melt spike excluded by a post-melt
     valley anchor. The dominant central peak (polycondensation to g-C3N4) is kept.
3. **Conversion and rate** – the isolated, baseline-free peak is integrated over
   time to give conversion `alpha(T)` and rate `dalpha/dt`.
4. **Isoconversional activation energy** `Ea(alpha)` (`kinetics.py`) by the
   Friedman (differential), KAS, OFW/FWO (Doyle), Starink and Vyazovkin advanced
   (nonlinear) methods.
5. **Mechanism identification** via the Criado master plot over the full set of 15
   solid-state reaction models, combined with the kinetic compensation effect (KCE).
6. **Consistency check** – forward integration of `alpha(T)` with the selected
   model and the isoconversional `Ea`, fitting a single pre-exponential factor.
7. **Export** (`figures.py`, `export_data.py`) – publication figures plus the
   post-processed XY data behind every figure as plain CSV.

## Results

| Precursor | Main DSC peak | Effective model | ln A | Mean Ea (Friedman), kJ/mol |
|-----------|---------------|-----------------|------|----------------------------|
| Melamine | ~550 °C | F1 (first order) | ~44 | ~340 |
| Melamine barbiturate | ~360 °C | R3 (contracting volume) | ~94 | ~480 (a) |
| Melamine thiobarbiturate | ~300 °C | R3 (contracting volume) | ~58 (b) | ~480 (a) |

> Figures and CSV values are the authoritative numbers; the table is a summary.

The process is multi-step (`Ea` varies with `alpha`), so `f(alpha)` is an
**effective** description rather than a single elementary mechanism. Two points of
method are worth stating explicitly, because they determine the model choice:

- The Criado master plot is **degenerate**: the first-order and Avrami–Erofeev
  curves coincide (F1 ≡ A2 ≡ A3 ≡ A4), and the contracting-volume and 3-D Jander
  diffusion curves coincide (R3 ≡ D3). The master plot therefore identifies a
  *group* of equivalent shapes, not a unique model.
- Within each group the choice is made on physical grounds: an Avrami
  nucleation–growth law is not appropriate for the molecular polycondensation of
  these precursors and is not used; the diffusion form D3 is rejected in favour of
  the geometric contracting-volume form R3.

## Repository layout

```
data/                         raw NETZSCH DSC exports (ASCII), one folder per precursor
  metadata.csv                sample, heating rate, mass, program, instrument, file
  melamine/                   melamine_{1,3,5}K.txt
  melamine-barbiturate/       melamine-barbiturate_{1,3,5}K.txt
  melamine-thiobarbiturate/   melamine-thiobarbiturate_{1,3,5}K.txt
src/                          analysis pipeline
  dsc_lib.py                  loader, baseline, Fraser-Suzuki, models, isoconversional, Criado
  extract.py                  peak isolation / deconvolution + deconvolution figures
  kinetics.py                 Ea(alpha), Criado master plot, compensation effect, forward model
  deviation.py                model-discrimination table for all 15 models
  figures.py                  kinetic figures
  export_data.py              post-processed XY data (one CSV per figure)
  run_all.py                  runs the whole pipeline
results/
  figures/                    generated PNG figures
  data/                       generated CSV (XY data, kinetic tables, model deviations)
```

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
python src/run_all.py
```

`results/figures/` and `results/data/` are regenerated from `data/` on every run.

## Data format

Raw files are NETZSCH ASCII exports (CP1251, `;`-separated); only the heating
segment is used. The exported `results/data/*.csv` are UTF-8 with a `#` comment
line describing the file. Wide CSVs store adjacent X,Y column pairs per curve
(highlight two neighbouring columns to plot); shorter series are padded with empty
cells. Units: temperature in °C, conversion `alpha` as a fraction (0..1),
`dalpha/dt` in 1/s, `Ea` in kJ/mol, DSC signal in mW/mg.

## Methods and references

- Friedman, KAS, OFW/FWO, Starink and Vyazovkin advanced isoconversional methods,
  following the ICTAC Kinetics Committee recommendations.
- Criado master-plot analysis of the 15 standard solid-state reaction models.
- Fraser–Suzuki asymmetric peak deconvolution and the constrained multi-rate fit
  after Perejón, Sánchez-Jiménez et al. (2011).
