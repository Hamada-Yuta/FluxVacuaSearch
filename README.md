# Data analysis of millions of Type IIB flux vacua

This repository contains the `gradient_P.py` scan and the supporting Python
modules and symbolic inputs from the StringVacua version used for the
fixed-`(m1, n1)` flux enumeration and Parquet output.

It is intended as a companion repository to the manuscript:

**Data analysis of millions of Type IIB flux vacua**  
Yuta Hamada, Yuma Uematsu, and Masahito Yamazaki  
**arXiv:** `26XX.XXXXX` (placeholder; to be replaced upon submission).

The bundled model `i = 2` is the model studied in the manuscript: the
**degree-18 Calabi–Yau hypersurface in the weighted projective space
$\mathbb{CP}^{4}_{1,1,1,6,9}$**. The full hypersurface has Hodge numbers
$(h^{1,1}, h^{2,1}) = (2,272)$. The scan works in the
$\Gamma = \mathbb{Z}_6 \times \mathbb{Z}_{18}$-invariant sector, with two
effective complex-structure moduli, $Z_1$ and $Z_2$. Accordingly, the value
`2` in `Potential_data/h21_2.csv` counts the effective moduli used by the
scan, rather than the full hypersurface's $h^{2,1} = 272$.

## Repository layout

```text
FluxVacuaSearch/
├── README.md
├── requirements.txt
├── gradient_P.py
├── Potential_data/
│   ├── h21_2.csv
│   ├── EK_2.csv
│   ├── DtauW_2.csv
│   ├── DtaubW_2.csv
│   ├── DZW_2.csv
│   ├── DZbW_2.csv
│   └── KZZb_2.csv
└── utils/
    ├── __init__.py
    ├── import_helpers.py
    ├── import_wrappers.py
    ├── logger_setup.py
    ├── optimization_worker.py
    ├── variable_generator.py
    └── ... (other StringVacua utility modules)
```

| Path | Purpose |
|---|---|
| `gradient_P.py` | Construct the scalar potential, enumerate fluxes for one fixed pair, minimize it with SLSQP, and write Parquet parts. |
| `utils/optimization_worker.py` | Perform ten starts per flux choice and return distinct accepted endpoints. |
| `utils/import_helpers.py`, `utils/import_wrappers.py` | Read the symbolic input expressions. |
| `utils/variable_generator.py`, `utils/logger_setup.py` | Construct symbols and configure logging. |
| `Potential_data/*_2.csv` | Seven symbolic input files for model 2; these are not sampled vacuum solutions. |

All fourteen Python files in the source `utils/` directory are included,
including the SUSY helpers and legacy workers. The scan uses
`optimization_worker.py`; it does not use `optimization_worker_old.py` or
`sampling.py`. Flux enumeration for this scan is defined directly in
`gradient_P.py`. Separate SUSY input files and entry-point scripts are not
bundled.

The main script, utility sources, requirements, and seven input CSV files
are copied without changes from the StringVacua snapshot. Python caches,
logs, and generated datasets are excluded. Only model 2 is supported by the
bundled scan: its flux enumeration explicitly assumes three components in
each of `m` and `n`.

## Quick start

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
Full production scans and reproduction of the manuscript's results are
not performed as part of this update.

Initial guesses are random; dependency versions are not pinned to a
publication reproduction environment. The numerical fallback behavior
described above is preserved in this snapshot.

## Citation

If you use this repository, please cite the companion manuscript.
Replace the placeholder arXiv identifier with the assigned identifier
before using this entry in a publication.

```bibtex
@article{Hamada:2026FluxVacua,
    author = "Hamada, Yuta and Uematsu, Yuma and Yamazaki, Masahito",
    title = "{Data analysis of millions of Type IIB flux vacua}",
    eprint = "26XX.XXXXX",
    archivePrefix = "arXiv",
    year = "2026",
    note = "arXiv identifier is a placeholder"
}
```
