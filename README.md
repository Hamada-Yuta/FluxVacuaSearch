# Data analysis of millions of Type IIB flux vacua

This repository contains the potential-minimization script `gradient_P.py`
and the supporting modules and symbolic inputs needed to run it. The script
searches for local minima of the Type IIB flux scalar potential using SLSQP.

It is intended as a companion repository to the manuscript:

**Data analysis of millions of Type IIB flux vacua**  
Yuta Hamada, Yuma Uematsu, and Masahito Yamazaki  
**arXiv:** `26XX.XXXXX` (placeholder; to be replaced upon submission).

The bundled inputs describe the default model `i = 2`, with two
complex-structure moduli. Large solution datasets, other search scripts,
and machine-learning analysis code are not included.

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
    ├── csv_helper.py
    ├── import_helpers.py
    ├── import_wrappers.py
    ├── logger_setup.py
    ├── optimization_worker.py
    ├── sampling.py
    └── variable_generator.py
```

### Code and inputs

| Path | Purpose |
|---|---|
| `gradient_P.py` | Construct the scalar potential, select a batch of flux choices, minimize it, and write CSV results. |
| `utils/` | Input parsing, variable generation, flux enumeration, optimization, CSV output, and logging. |
| `Potential_data/*_2.csv` | Seven symbolic input files for the default model, not sampled vacuum solutions. |

Only model `i = 2` inputs are bundled. Changing the model requires the
corresponding input expressions and a review of model-specific assumptions.

## Quick start

### Set up Python

Use Python 3.9 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are NumPy, SciPy, SymPy, joblib, and tqdm.
No pandas or Parquet engine is required.

### Inspect the command-line option

```bash
python gradient_P.py --help
```

The only scan option is `--index` (default: `0`). It selects a batch
from the ordered flux enumeration, with a starting offset of
`index * sample_size`. It is not a random-number seed or an index
of a fixed `(m1, n1)` pair.

Current settings in `gradient_P.py`:

| Setting | Value |
|---|---|
| Model range | `i_start = i_end = 2` |
| Flux choices per batch | `sample_size = 1500` |
| Enumeration range parameter | `flux_range = 17` |
| Flux sum-of-squares bound | `max_sum_of_squares = 276` |
| Parallel workers | `N_JOBS = 4` |

The first components of both flux vectors are nonnegative, and the
remaining components may have either sign, subject to the bound.
Optimization uses ten random initial guesses per flux choice.

### Run a batch

Run from the repository root so that relative input paths resolve correctly:

```bash
mkdir -p output
python gradient_P.py --index 0
```

This can be a substantial calculation: it requests up to 1,500 flux choices
with multiple optimization starts per choice. For a small trial, first reduce
`sample_size` in the script (for example, to `1`). Changing
`sample_size` also changes the batch offsets selected by `--index`.
There are no `--chunk-id` or `--chunk-size` options.

The script writes:

```text
output/i_2_gradient_<index:06>_<YYYYMMDD_HHMM>.csv
run.log
```

Create `output/` before running; the script does not create it automatically.
Avoid concurrent runs with the same index in the same minute, since they
would target the same CSV filename. Generated outputs and logs are excluded
from Git tracking.

### Read generated results

```python
import csv
from pathlib import Path

files = sorted(Path("output").glob("i_2_gradient_*.csv"))
if files:
    with files[0].open(newline="") as stream:
        reader = csv.DictReader(stream)
        print(reader.fieldnames)
        print(next(reader, None))
else:
    print("No result files found.")
```

## Data formats

### Symbolic input CSV files

Input expressions use Mathematica-style notation and are parsed with SymPy.
They contain the exponential Kähler factor, covariant derivatives of the
superpotential, the inverse complex-structure metric factor, and the number
of complex-structure moduli. The suffix `_2` identifies the bundled model.

### Generated solution CSV files

For the default model, the columns are:

```text
FunctionValue, taur, taui, Z1r, Z1i, Z2r, Z2i,
m1, m2, m3, n1, n2, n3
```

`FunctionValue` is the absolute potential value returned by the optimizer's
objective. The `r` and `i` suffixes denote real and imaginary parts;
`m*` and `n*` label flux components. Multiple solutions may be recorded
for the same flux choice. Check `run.log` for errors before interpreting
the output; interrupted runs may contain a trailing status message.

## Validation status

This is a release-preparation snapshot. Checks cover Python syntax, imports
in the reduced file tree, `gradient_P.py --help`, symbolic-input loading
and potential construction, and a small flux-enumeration check.
Full production scans and reproduction of paper results have not been
validated for this distribution.

The numerical logic of `gradient_P.py` and its optimization worker is
unchanged. Random initialization is used, and dependency versions are not
pinned to a publication reproduction environment.

## Citation

If you use this repository, please cite the companion manuscript.
The arXiv identifier below is a placeholder and must be replaced with the
assigned identifier before using this entry in a publication.

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
