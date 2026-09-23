# String Vacua — release preparation

Python scripts for numerical searches of string flux vacua.

This is a **private release-preparation snapshot**, not a reviewed public release.
It starts with a new Git history. Internal discussion notes, large result
datasets, notebooks, Mathematica notebooks, cluster submission scripts and
obsolete worker implementations are not included.

## Contents

- `gradient_P_index_chunk.py`: indexed, chunked flux scan; writes Parquet results.
- `gradient_P_index.py`: indexed flux scan without chunk slicing.
- `gradient_P.py`: alternative potential-minimization scan; writes CSV results.
- `SUSY_P.py`: search based on supersymmetry equations.
- `utils/`: supporting Python modules used by these scripts.
- `Potential_data/*_2.csv` and `SUSY_data/DW_2.csv`: small symbolic
  input expressions for the default model `i = 2`, **not** sampled solution datasets.

Only model `i = 2` inputs are bundled. Other models require their corresponding
input expressions. The large Calabi–Yau catalogue is not included.

## Setup

Use Python 3.9 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python gradient_P_index_chunk.py --help
```

The indexed scripts use `pandas` and a Parquet engine; `pyarrow` is included in
the dependency list. Package versions are not yet pinned to a publication
reproduction environment.

## Running a small scan

Review the model and numerical settings before running. All relative input
paths are resolved from the current directory, so run from the repository root.

```sh
python gradient_P_index_chunk.py --index 0 --chunk-id 0 --chunk-size 1
```

Even one flux point can take time because each point uses multiple optimization
starts. The chunked script uses eight workers by default (`N_JOBS` in the
script). Results go to `output/parquet/`; logging goes to `run.log`.
Do not launch the unrestricted scans inadvertently: they can be expensive.
For the CSV-based scripts, create `output/` before execution.

## Validation and limitations

Preparation checks completed:
- Python syntax compilation of the 15 included Python files.
- `gradient_P_index_chunk.py --help`.
- Local imports and parsing of the seven bundled potential input files.

A full scan, the SUSY workflow, and reproduction of paper figures/results
have **not** been validated here. Numerical algorithms, tolerances and random
initialization were copied without intentional changes; no reproducibility
guarantee is implied. No machine-learning analysis code is included in this snapshot.

## Before making this repository public

The authors should confirm the selected code and symbolic inputs are the
versions intended for release, choose a license, and add the final paper
citation and reproducibility information. No license has been chosen on their
behalf.
