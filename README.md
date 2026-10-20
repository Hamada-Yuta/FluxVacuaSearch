# Data analysis of millions of Type IIB flux vacua

This repository contains the flux scan, the selected sample used in Chapters 4
and 5, and executable statistical and machine-learning analyses for the manuscript.

It is intended as a companion repository to the manuscript:

**Data analysis of millions of Type IIB flux vacua**  
Yuta Hamada, Yuma Uematsu, and Masahito Yamazaki  
An arXiv identifier will be added after submission.

The bundled model `i = 2` is the model studied in the manuscript: the
**degree-18 Calabi–Yau hypersurface in the weighted projective space
$\mathbb{CP}^{4}_{1,1,1,6,9}$**. The full hypersurface has Hodge numbers
$(h^{1,1}, h^{2,1}) = (2,272)$. The scan works in the
$\Gamma = \mathbb{Z}_6 \times \mathbb{Z}_{18}$-invariant sector, with two
effective complex-structure moduli, $Z_1$ and $Z_2$. Accordingly, the value
`2` in `Potential_data/h21_2.csv` counts the effective moduli used by the
scan, rather than the full hypersurface's $h^{2,1} = 272$.

## Reproduce Chapters 4 and 5

The dataset contains **1,296,633 reliable solutions** in 13 columns.
All analyses use this same sample. The numerical SUSY criterion
$V < 10^{-4.5}$ selects **127,241 SUSY solutions**; the remaining
**1,169,392 solutions** are non-SUSY. The selection criteria and the meaning
of numerical SUSY classification are defined in the manuscript.

### Install the analysis environment

The reference run used **Python 3.11.3**, **XGBoost 3.0.5** and
**scikit-learn 1.8.0**. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m analysis.prepare_data
```

The last command concatenates the two distributed Parquet parts in their
explicit file order and writes `data/selected_solutions.parquet`. Splitting the
file for distribution does not change any value, dtype or row position. The
complete adopted dataset is in Git; no scan or external data download is needed.
On macOS, XGBoost's CPU runtime may require the OpenMP library provided by
`brew install libomp`.

### Run the analyses

```bash
python -m analysis.chapter4 --data data/selected_solutions.parquet --output results/chapter4
python -m analysis.flux_statistics --data data/selected_solutions.parquet --output results/flux_statistics
python -m analysis.classification --data data/selected_solutions.parquet --output results/classification
python -m analysis.regression --data data/selected_solutions.parquet --output results/regression
```

Every model is fitted by these commands. They do not load previous predictions
or private working directories. Model fitting uses two CPU threads. Outputs
include plots, numerical tables, settings, software versions, and the sample
indices used for training, validation, testing and auxiliary subsampling.
Individual predictions are optional: add `--save-predictions` to the two ML
commands if they are needed. Generated results and fitted models are not
required to be committed to Git.

| Entry point | Coverage |
|---|---|
| `analysis.chapter4` | Full and SUSY sample summaries; Pearson and Spearman figures; numerical Kendall comparisons; restricted-tail comparisons; moduli densities; instanton corrections through degree 7. |
| `analysis.flux_statistics` | Logarithmic potential distribution; SUSY fraction at each flux value; invariant combinations; flux-orbit counts; shared-input diagnostics and examples with both SUSY and non-SUSY solutions. |
| `analysis.classification` | Five input-feature comparisons, the main classifier, ranking, stricter potential thresholds, TreeSHAP contributions, grouped-split comparisons and performance as the training set grows. |
| `analysis.regression` | Four potential regressions and twelve moduli regressions, using the full sample for every target. |

Both regression tasks use **972,474 training solutions and 324,159 test
solutions**, with the same split for all models. Each of the six moduli
coordinates is predicted separately using either the six flux parameters
$m_A,n_A$ or the nine invariants $G_{AB},\omega_{AB}$.
Imaginary coordinates are transformed with $\log_{10}x$; real coordinates
with $\mathrm{sign}(x)\ln(1+\lvert x\rvert)$.
The moduli figure contains one panel: predicted against true
$\log_{10}(\mathrm{Im}\,Z_2)$ using the invariant inputs.

### Manuscript figures and tables

The filenames below are stable identifiers; figure and table numbers refer
to the manuscript draft of 20 October 2026. Figures are generated in the
chosen output directories. Figure 1 is the manuscript's own TikZ flowchart,
and Table 2 compares earlier papers rather than a computed dataset summary.

| Manuscript item | Analysis output |
|---|---|
| Table 1; Table 3 | `chapter4/table_all_moduli.csv`; `chapter4/table_SUSY_moduli.csv` |
| Figure 2 | `chapter4/F_correction_histogram.pdf` |
| Figure 3 | `flux_statistics/ml_hist_logV.pdf` |
| Figures 4–5 | `chapter4/correlation_matrix_reliable.pdf`, `correlation_matrix_Z1i_Z2i_leq50.pdf`, `rank_correlation_spearman.pdf` |
| Figure 6 | `chapter4/Z1i_Z1r.pdf`, `Z2i_Z2r.pdf`, `Z2r_Z1r.pdf`, `Z2i_Z1i.pdf` |
| Figures 7–9 | `chapter4/correlation_matrix_SUSY.pdf`, `rank_correlation_SUSY.pdf` (Spearman), `SUSY_Z2r_Z1r.pdf`, `SUSY_Z2i_Z1i.pdf` |
| Table 4; Figure 10 | `flux_statistics/solutions_per_orbit.csv`; `ml_vacuum_rate.pdf` |
| Table 5 | `classification/feature_ablation.csv`, including the Brier score |
| Table 6 | `classification/budget.csv` |
| Figure 11 | `classification/ml_shap.pdf` and `shap.csv` |
| Tables 7–9 | `classification/threshold_robustness.csv`, `grouped_comparison.csv`, `learning_curve.csv` |
| Tables 10–11; Figure 12 | `regression/logV_scores.csv`, `moduli_scores.csv`; `ml_moduli_prediction.pdf` |

The classifier ROC, precision-recall, and ranking plots in
`classification/ml_classifier.pdf` are retained as auxiliary diagnostics;
they are not included as a figure in the manuscript.

Numerical Kendall correlations are retained to support the qualitative
comparison in Section 4.3; no Kendall figure is generated.

### Saved reference results and split identities

`reference_results/` records numerical summaries for the current manuscript,
compressed split indices, software versions and model configurations.
The scripts recreate the splits from the settings in the code and the
original dataset order; the saved indices also make each assignment directly
inspectable.

- A source row is the **zero-based physical row position** in the concatenated
  Parquet table. Preserve its order and values when rerunning the analyses.
- Classification and forward regression save their source-row splits in NPZ
  files. Auxiliary grouped, learning-curve and SHAP samples are also recorded.
- Numerical summaries are stored as JSON/CSV, with model settings and software
  versions alongside them. Individual predictions are regenerated on request.

The statistical unit is one reliable solution, including multiple solutions
with the same flux. The coordinates are used before fundamental-domain mapping.
Tail comparisons and plot windows are explicit in the scripts and metadata.

The selected sample suffices to reproduce the reported statistics and ML
analyses. It does not reconstruct the original random scan, or rerun the
selection/Hessian/deduplication procedure from all raw endpoints. In particular,
the number of scanned flux orbits used as the zero-discovery denominator is a
recorded scan constant, not inferred from the selected rows. Floating-point
results may differ slightly on different platforms or library builds.

## Repository layout

```text
FluxVacuaSearch/
├── README.md
├── requirements.txt                 # original scan dependencies
├── requirements-analysis.txt        # pinned analysis environment
├── gradient_P.py                    # original scan entry point
├── Potential_data/                  # seven symbolic inputs for model 2
├── utils/                           # scan utilities
├── data/selected_solutions/         # two ordered Parquet parts
├── analysis/                        # portable analysis entry points and helpers
└── reference_results/               # summaries, configurations and split indices
```

`Potential_data/*_2.csv` are symbolic expressions, not sampled solutions. Only
model 2 is supported by the bundled scan: its flux enumeration assumes three
components in each of `m` and `n`. The original scan and symbolic inputs are
retained separately from the analysis code below.

## Running a fresh scan

### Set up Python

Use Python 3.9 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python gradient_P.py --help
```

Dependencies are NumPy, SciPy, SymPy, joblib, tqdm, pandas, and PyArrow.
PyArrow provides the Parquet engine.

### Choose the assigned flux pair

The scan option is `--index` (default: `0`). There are **234 valid indices,
0 through 233**. Each index selects one fixed pair `(m1, n1)`, ordered by
increasing `m1` and then increasing `n1`, from:

```text
0 <= m1, n1 <= 16
m1**2 + n1**2 <= 276
```

Index 0 selects `(0, 0)`; index 233 selects `(16, 4)`. Inspect the full
assignment without running an optimization:

```bash
python - <<'PY'
from gradient_P import valid_m1n1_pairs
for index, (m1, n1, sumsq) in enumerate(valid_m1n1_pairs()):
    print(index, m1, n1, sumsq)
PY
```

For the assigned pair, the script exhaustively enumerates the remaining
four components `m2, n2, m3, n3` in `[-16, 16]`, subject to:

```text
m1**2 + m2**2 + m3**2 + n1**2 + n2**2 + n3**2 <= 276
```

Across all 234 indices this enumerates 30,770,178 distinct flux tuples,
including the all-zero tuple. The latter is not removed by the scan.
This is enumeration within the stated component ranges; no duality
identification is performed.

| Setting | Value |
|---|---|
| Model | `i_start = i_end = 2` |
| Assignment | One fixed `(m1, n1)` per `--index` |
| Flux sum-of-squares bound | 276 |
| Fluxes per processing block | `BLOCK = 2500` |
| Parallel workers per running script | `N_JOBS = 8` |
| Initial guesses per flux | 10 |
| Rows per complete Parquet part | `CHUNK_SIZE = 50_000` |

`BLOCK` controls how many flux tuples are passed to joblib at once.
It does not limit the total number of fluxes explored for an index.
The largest assignment, index 0, contains 377,657 flux tuples.

### Run an assignment

Run from the repository root so relative input paths resolve correctly:

```bash
python gradient_P.py --index 233
```

This assignment contains 89 flux tuples and attempts ten starts per tuple.
A full scan requires running every index from 0 through 233 once; assignments
have different sizes. When using a scheduler, pass its array index to
`--index` and account for eight worker processes per script.

The previous CSV version used `index * 1500` as a flux-list offset.
Those old index assignments do not correspond to the fixed-pair indices
used here. There are no `--chunk-id`, `--chunk-size`, or restart options.
To limit a local trial to fewer fluxes, explicitly truncate the generator
in a local copy; reducing `BLOCK` alone does not shorten the scan.

The output directory is created automatically:

```text
output/parquet/i=2/index=<index:06d>/ts=<YYYYMMDD_HHMM>/part-00000.parquet
output/parquet/i=2/index=<index:06d>/ts=<YYYYMMDD_HHMM>/part-00001.parquet
...
run.log
```

The final part may contain fewer than 50,000 rows. If no endpoints are
accepted, no part is written. An interrupted run can lose the unwritten
buffer while leaving earlier parts on disk. Re-running an index restarts
its scan and does not resume from existing parts.

Use a distinct output location or minute for repeated runs of the same
index: the timestamp has minute precision, and matching part names can be
overwritten. All runs in the same working directory append to `run.log`.

### Read generated results

Read an individual part without loading the entire dataset:

```python
from pathlib import Path
import pandas as pd

parts = sorted(Path("output/parquet").glob("i=2/index=*/ts=*/part-*.parquet"))
if parts:
    df = pd.read_parquet(parts[0])
    print(df.dtypes)
    print(df.head())
else:
    print("No result files found.")
```

When combining parts, select the intended run for each index so repeated
runs do not double-count the same flux assignments.

## Data formats and optimization

### Symbolic inputs

The seven CSV files use Mathematica-style expressions parsed with SymPy.
They specify the exponential Kähler factor, covariant derivatives of the
superpotential, inverse complex-structure metric factor, and number of
complex-structure moduli. The `_2` suffix identifies the bundled model.

### Generated Parquet parts

The columns are:

```text
FunctionValue, taur, taui, Z1r, Z1i, Z2r, Z2i,
m1, m2, m3, n1, n2, n3
```

The first seven columns are `float64`; the six flux columns are `int16`.
`FunctionValue` is the absolute value returned by the objective wrapper.
The `r` and `i` suffixes denote real and imaginary parts.

Each flux choice receives ten independent random initial guesses, with
real parts uniform in `[-2, 2]` and imaginary parts uniform in `[0.1, 2]`.
The worker uses SLSQP with `tol=1e-5` and imaginary-part lower bounds of
`1e-5`. It accepts endpoints when the solver reports success and all
components of its gradient wrapper are at most 0.1 in absolute value.
Within each flux choice, accepted endpoints are deduplicated by rounding
the potential and coordinates to two significant digits. Stored values
retain their unrounded precision.

The scan does not apply the later LCS cut `Im Z1, Im Z2 >= 1`, classify
SUSY solutions, map the axio-dilaton to the modular fundamental domain,
or test the Hessian for local-minimum stability.

### Known numerical limitation

The copied worker returns `1e200` when objective evaluation fails or is
nonfinite, and returns an entire zero gradient vector if any gradient
component is nonfinite or an exception occurs. Consequently, solver success
and the gradient-wrapper check can accept invalid endpoints. An objective
value different from `1e200` does not exclude this failure mode.

Reevaluate saved endpoints with explicit failure reporting and check finite
potential, finite gradient, and the gradient threshold before interpreting
them as solutions. The scan output requires the subsequent selection,
Hessian check and duplicate removal described in the manuscript. The
distributed reliable-solution sample has already passed that selection.

## Validation status

The distributed Parquet parts reproduce the full sample exactly, including
values, types and order. Chapter 4 statistics and figures were regenerated
from that sample. The retained classification and full-sample regression
scores were checked against predictions from the fitted models, and the
training/test splits were reconstructed. Python syntax, imports and CLI
entry points were also checked. A fresh full production flux scan was not
run as part of this update.

Initial guesses in the scan are random and are not seeded. The original scan
requirements are unpinned; the separate analysis environment is pinned. The numerical fallback behavior
described above is preserved in this snapshot.

## Citation

Please cite the companion manuscript when using the data or code. Its official
bibliographic entry will be added after an arXiv or journal record is available.
