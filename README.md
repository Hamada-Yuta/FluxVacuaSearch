# FluxVacuaSearch

This repository contains Python code and symbolic input data for numerical
searches of string flux vacua. It provides routines for minimizing the scalar
potential and for searching for solutions of supersymmetry conditions.

The source tree is organized around two parts:

- Potential-minimization scripts, including indexed and chunked flux scans.
- A supersymmetric-vacuum search script and its supporting equation data.

The bundled inputs describe the default model `i = 2`, with two
complex-structure moduli. Large generated solution datasets and the full
Calabi–Yau catalogue are not included. The repository does not currently
contain machine-learning analysis code.

## Repository layout

```text
FluxVacuaSearch/
├── README.md
├── requirements.txt
├── gradient_P.py
├── gradient_P_index.py
├── gradient_P_index_chunk.py
├── SUSY_P.py
├── Potential_data/
│   ├── h21_2.csv
│   ├── EK_2.csv
│   ├── DtauW_2.csv
│   ├── DtaubW_2.csv
│   ├── DZW_2.csv
│   ├── DZbW_2.csv
│   └── KZZb_2.csv
├── SUSY_data/
│   └── DW_2.csv
└── utils/
```

### Search scripts

| Path | Purpose |
|---|---|
| `gradient_P_index_chunk.py` | Divide an indexed flux scan into chunks and write solutions as Parquet files. |
| `gradient_P_index.py` | Run an indexed flux scan without chunk slicing; write Parquet files. |
| `gradient_P.py` | Run an alternative potential-minimization scan; write CSV files. |
| `SUSY_P.py` | Search for solutions of supersymmetry equations; write CSV files. |
| `utils/` | Input parsing, variable generation, flux sampling, numerical solvers, and logging. |

### Symbolic inputs

`Potential_data/` contains the symbolic expressions used to construct the
scalar potential: the exponential Kähler factor, covariant derivatives of the
superpotential, and the complex-structure metric factor, together with the
number of complex-structure moduli.

`SUSY_data/DW_2.csv` supplies the equation matrix read by the supersymmetry
search. These CSV files are **inputs to the calculation**, not tables of
sampled vacuum solutions.

Only the default model's inputs are bundled. Changing the model requires the
corresponding input expressions as well as a review of the script's
model-specific assumptions.

## Quick start

### Set up Python

Use Python 3.9 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The code uses NumPy, SciPy, SymPy, joblib, and tqdm. The Parquet-output scripts
also use pandas and pyarrow.

### Inspect the scan options

```bash
python gradient_P_index_chunk.py --help
```

The chunked scan accepts:

| Option | Meaning |
|---|---|
| `--index` | Zero-based index of the fixed `(m1, n1)` pair. |
| `--chunk-id` | Zero-based chunk number within that pair's remaining flux grid. |
| `--chunk-size` | Number of flux points assigned to each chunk. |

The current enumeration uses `0 <= m1, n1 <= 16`, remaining flux components
between `-16` and `16`, and a total sum-of-squares bound of `276`.
These settings are defined in the script.

### Run a small scan

Run from the repository root so that relative input paths resolve correctly:

```bash
python gradient_P_index_chunk.py --index 0 --chunk-id 0 --chunk-size 1
```

This assigns one flux point; it does not guarantee that a vacuum will be
found. Each point can involve multiple optimization starts, so even a small
job may take time. The chunked script uses eight workers by default, controlled
by `N_JOBS`.

Results are written beneath:

```text
output/parquet/i=2/index=<index>/chunk=<chunk>/ts=<timestamp>/part-*.parquet
```

Logging is written to `run.log`. Generated outputs and logs are excluded from
Git tracking. Before using either CSV-output script, create the output
directory with `mkdir -p output`.

Review flux bounds, model settings, and worker counts before production runs.
The unrestricted scans can be computationally expensive.

### Read generated results

After a scan has produced a Parquet file:

```python
from pathlib import Path
import pandas as pd

files = sorted(Path("output/parquet").rglob("part-*.parquet"))
if files:
    solutions = pd.read_parquet(files[0])
    print(solutions.head())
else:
    print("No result files found.")
```

## Data formats

### Symbolic input CSV files

Input expressions use Mathematica-style notation. The loaders in
`utils/import_helpers.py` and the search scripts translate the variable names
and parse the expressions with SymPy. The suffix `_2` identifies the bundled
model; `h21_2.csv` records its number of complex-structure moduli.

### Generated solution tables

For the default model, the potential-minimization Parquet output has the columns:

```text
FunctionValue, taur, taui, Z1r, Z1i, Z2r, Z2i,
m1, m2, m3, n1, n2, n3
```

`FunctionValue` is the minimized scalar-potential value. The `r` and `i`
suffixes denote real and imaginary parts; `m*` and `n*` label integer flux
components. Multiple solutions may be recorded for the same flux choice.

The SUSY CSV output records moduli and flux components without the
`FunctionValue` column. Large generated catalogues are not distributed in
this repository, and no external dataset archive is specified here.

## Validation status

This is a release-preparation snapshot. Checks performed so far cover Python
syntax, the chunked script's help command, local imports, and parsing of the
bundled potential inputs. Full scans, the SUSY workflow, and reproduction of
paper results have not yet been validated for this distribution.

Numerical routines were copied without intentional algorithmic changes.
Random initialization is used, and dependency versions are not pinned to a
publication reproduction environment.

## Citation

A paper citation and BibTeX entry have not yet been specified for this
repository. They should be added once the authors confirm the appropriate
reference.

## License

No license has been selected for this release-preparation snapshot.
