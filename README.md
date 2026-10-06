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

The adopted dataset contains **1,296,633 selected numerical candidates** in
13 columns. All analyses use this same sample. The operational SUSY label is
$V < 10^{-4.5}$, which selects **127,241 rows**; the remaining **1,169,392 rows**
form its complement. These numerical labels and the recorded selection criteria
have the scope stated in the manuscript.

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
python -m analysis.conditional_regression --data data/selected_solutions.parquet --output results/conditional_regression
```

Every model is fitted by these commands. They do not load previous predictions
or private working directories. Model fitting uses two CPU threads. Outputs
include plots, numerical tables, settings, software versions, and the sample
indices used for training, validation, testing and auxiliary subsampling.
Individual predictions are optional: add `--save-predictions` to the three ML
commands if they are needed. Generated results and fitted models are not
required to be committed to Git.

| Entry point | Coverage |
|---|---|
| `analysis.chapter4` | Full and SUSY sample summaries; Pearson, Spearman and Kendall correlations; restricted-tail comparisons; moduli densities; direct instanton corrections through degree 7. |
| `analysis.flux_statistics` | Logarithmic potential distribution; low-potential rates per flux; invariant multiplicities; orbit counts; shared-input diagnostics and mixed-label examples. |
| `analysis.classification` | Feature comparisons, the main classifier, ranking and budget metrics, learning curves, threshold robustness, SHAP summaries and grouped-split comparisons. |
| `analysis.regression` | Four log-potential regressions and twelve raw-flux/invariant moduli regressions. |
| `analysis.conditional_regression` | Thirteen conditional random-forest models each for the full and SUSY samples, feature importances, tail diagnostics and an independent check of the $W_0$ formula. |

### Saved reference results and split identities

`reference_results/` records the numerical summaries from a complete rerun of
these scripts against the distributed dataset. It also includes compressed
split indices and model configurations. The scripts recreate these indices
from the recorded seeds and dataset row order; the saved indices make the
original assignment directly inspectable.

- A source row is the **zero-based physical row position** in the concatenated
  Parquet table. Preserve its order and values when rerunning the analyses.
- Classification and forward regression save their source-row splits in NPZ
  files. Auxiliary grouped, learning-curve and SHAP samples are also recorded.
- Conditional regression first sorts each population stably by `FunctionValue`.
  Its NPZ files contain both positions within that sorted population and the
  corresponding original source-row mappings.
- The main random seeds are 0. The matched grouped-comparison test subsample
  uses seed 1. Per-analysis settings record these choices explicitly.
- Numerical summaries are stored as JSON/CSV, with model settings and software
  versions alongside them. Individual predictions are regenerated on request.

The statistical unit is one retained solution row, including multiple rows
with the same flux. The raw saved coordinates are used; the distribution file
has not been remapped into a fundamental domain. Exactly-zero fluxes are
excluded only where the corresponding analysis specifies this. Tail and plot
windows are explicit in the scripts and metadata.

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
them as solutions. In the audited 46,388,212-row dataset, 83,959 anomalous
LCS endpoints coincided with the analysis notebook's `V >= 1e10` exclusion;
that dataset-specific observation does not make this cutoff a general
replacement for endpoint validation. The raw Parquet files produced here
do not apply that exclusion.

## Validation status

This update checks source/input equality to the StringVacua snapshot,
Python syntax and imports, the CLI help, symbolic-input construction,
flux assignments and enumeration counts, and Parquet write/read behavior.
A bounded integration run checks the scan with a reduced flux iterator.
A fresh full production scan was not run as part of packaging this repository.
The selected-sample analyses above were rerun and checked against the manuscript's
existing numerical results.

Initial guesses in the scan are random and are not seeded. The original scan
requirements are unpinned; the separate analysis environment is pinned. The numerical fallback behavior
described above is preserved in this snapshot.

## Citation

Please cite the companion manuscript when using the data or code. Its official
bibliographic entry will be added after an arXiv or journal record is available.
