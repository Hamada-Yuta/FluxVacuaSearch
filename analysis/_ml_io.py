"""CLI and output utilities for the statistical and forward-ML analyses."""
from pathlib import Path
import argparse
import json
import platform

SELECTED_ROWS = 1_296_633
SUSY_ROWS = 127_241
THRESHOLD = 10.0 ** -4.5


def parse_cli(description, predictions=True):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, required=True,
                        help="Selected-solution Parquet, in its original row order.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for figures, summaries, settings and split indices.")
    if predictions:
        parser.add_argument("--save-predictions", action="store_true",
                            help="Also save individual held-out predictions (not needed for the tables/figures).")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Input Parquet does not exist: {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def environment(modules):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {module.__name__: module.__version__ for module in modules}}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_sample(frame, columns):
    """Check the supplied sample without selecting, sorting or rewriting its rows."""
    import numpy as np
    if len(frame) != SELECTED_ROWS:
        raise ValueError(f"Expected {SELECTED_ROWS:,} selected rows; got {len(frame):,}.")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError("The supplied sample contains non-finite values.")
    if int((frame.FunctionValue.to_numpy() < THRESHOLD).sum()) != SUSY_ROWS:
        raise ValueError(f"Expected {SUSY_ROWS:,} rows below V=10**(-4.5).")
