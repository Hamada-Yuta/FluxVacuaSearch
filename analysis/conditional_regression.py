r"""Reproduce conditional random-forest regressions in Chapters 4--5.

Run from the repository root::

    python -m analysis.conditional_regression \
        --data data/selected_solutions.parquet \
        --output results/conditional_regression

Every model is fitted from the input data. There are no dependencies on prior
results, manuscript sources, or local audit directories. Targets and features
remain in their original units. These regressions use recorded endpoint data;
in particular, predicting a flux uses the other five fluxes and is not a
reconstruction from moduli alone. Individual predictions are saved only when
``--save-predictions`` is supplied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import pyarrow
import scipy
from scipy.special import zeta
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_info, threadpool_limits


INPUTS = [
    "FunctionValue", "taur", "taui", "Z1r", "Z1i", "Z2r", "Z2i",
    "m1", "m2", "m3", "n1", "n2", "n3",
]
SUSY_THRESHOLD = 10 ** (-4.5)
RF_PARAMETERS = dict(
    n_estimators=30, max_depth=10, min_samples_leaf=100,
    max_samples=20000, n_jobs=2, random_state=0,
)
SPLIT_PARAMETERS = dict(test_size=0.2, random_state=0, shuffle=True)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def current_w_and_ek(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Period-vector W and e^K with the Kahler-volume factor omitted.

    The degree-18 model's polynomial prepotential includes the zeta(3) term;
    worldsheet instantons are omitted, as in the scan. The overall
    Kahler-moduli-dependent factor is suppressed.
    """
    t = df.taur.to_numpy() + 1j * df.taui.to_numpy()
    z1 = df.Z1r.to_numpy() + 1j * df.Z1i.to_numpy()
    z2 = df.Z2r.to_numpy() + 1j * df.Z2i.to_numpy()
    m1, m2, m3, n1, n2, n3 = df[INPUTS[-6:]].to_numpy(float).T
    xi = 67.5 * zeta(3, 1) / np.pi**3
    p0 = 1.5*z1**3 + 1.5*z1*z1*z2 + 0.5*z1*z2*z2 + 4.25*z1 + 1.5*z2 - 1j*xi
    p1 = -4.5*z1*z1 - 3*z1*z2 - 0.5*z2*z2 + 4.5*z1 + 1.5*z2 + 4.25
    p2 = -1.5*z1*z1 - z1*z2 + 1.5*z1 + 1.5
    W = -((m1-t*n1) + (m2-t*n2)*z1 + (m3-t*n3)*z2
          + (n1+t*m1)*p0 + (n2+t*m2)*p1 + (n3+t*m3)*p2)
    y1, y2, s = df.Z1i.to_numpy(), df.Z2i.to_numpy(), df.taui.to_numpy()
    ek = 1 / (8*s*y1*(3*y1*y1 + 3*y1*y2 + y2*y2) + 4*s*xi)
    return W, ek


def independent_expanded_w_and_ek(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Independent expanded W polynomial and complex-coordinate e^K.

    This does not call the period-vector construction above. Its e^K expression
    uses coordinate differences rather than the real-coordinate denominator.
    """
    t = df.taur.to_numpy() + 1j * df.taui.to_numpy()
    z1 = df.Z1r.to_numpy() + 1j * df.Z1i.to_numpy()
    z2 = df.Z2r.to_numpy() + 1j * df.Z2i.to_numpy()
    m1, m2, m3, n1, n2, n3 = df[INPUTS[-6:]].to_numpy(float).T
    c = 270j * zeta(3, 1) / np.pi**3
    P = 3*z1**2 + 3*z1*z2 + z2**2
    W = (-0.5*m1*P*z1*t
         +0.5*(-n1*P*z1 + (m2*(3*z1+z2)**2 + m3*(3*z1+2*z2)*z1)*t)
         +0.25*(-(17*m1+6*(3*m2+m3)-4*n2)*z1*t
                +2*(2*n3-3*(m1+m2))*z2*t
                +2*n2*(3*z1+z2)**2 +2*n3*(3*z1**2+2*z1*z2))
         -0.25*((4*m2+17*n1+18*n2+6*n3)*z1
                +2*(2*m3+3*n1+3*n2)*z2+(17*m2+6*m3-4*n1-c*m1)*t)
         -0.25*(4*m1+17*n2+6*n3-c*n1))
    d1, d2, dt = z1-z1.conjugate(), z2-z2.conjugate(), t-t.conjugate()
    ek = 2 / (dt*(d1*(3*d1**2+d2**2+3*d1*d2)-c))
    return W, ek


def prepare_susy(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    susy = frame.loc[frame.FunctionValue < SUSY_THRESHOLD].copy()
    if len(susy) == 0:
        raise ValueError("The input has no rows satisfying the numerical SUSY criterion.")
    W, ek = current_w_and_ek(susy)
    if not (np.isfinite(W).all() and np.isfinite(ek).all() and (ek > 0).all()):
        raise ValueError("Nonfinite W or nonpositive/nonfinite e^K in the SUSY subset.")
    susy["W0_abs"] = np.sqrt(2 / np.pi) * np.sqrt(ek) * np.abs(W)
    if not np.isfinite(susy.W0_abs.to_numpy()).all():
        raise ValueError("Nonfinite |W0| in the SUSY subset.")
    random_spots = np.random.default_rng(0).choice(len(susy), min(1024, len(susy)), replace=False)
    spots = np.unique(np.concatenate([
        random_spots,
        [np.argmin(susy.W0_abs), np.argmax(susy.W0_abs),
         np.argmax(susy.taui), np.argmax(susy.Z1i), np.argmax(susy.Z2i)],
    ]))
    independent_W, independent_ek = independent_expanded_w_and_ek(susy.iloc[spots])
    independent_w0 = np.sqrt(2 / np.pi) * np.sqrt(independent_ek.real) * np.abs(independent_W)
    np.testing.assert_allclose(independent_W, W[spots], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(independent_ek, ek[spots], rtol=1e-12, atol=0)
    np.testing.assert_allclose(independent_w0, susy.W0_abs.to_numpy()[spots], rtol=1e-10, atol=1e-10)
    validation = {
        "passed": True, "n_spots": len(spots),
        "source_rows": susy.index.to_numpy()[spots].tolist(),
        "positions_in_stable_sorted_susy": spots.tolist(),
        "max_W_scaled_difference": float(np.max(np.abs(independent_W-W[spots])/(1+np.abs(W[spots])))),
        "max_eK_relative_difference": float(np.max(np.abs(independent_ek-ek[spots])/ek[spots])),
        "max_W0_scaled_difference": float(np.max(np.abs(independent_w0-susy.W0_abs.to_numpy()[spots])/(1+susy.W0_abs.to_numpy()[spots]))),
        "formulas": "Period-vector W and real e^K versus expanded W and complex-coordinate e^K.",
        "tolerances": {"W": {"rtol": 1e-10, "atol": 1e-8},
                       "eK": {"rtol": 1e-12, "atol": 0},
                       "W0_abs": {"rtol": 1e-10, "atol": 1e-10}},
    }
    return susy, validation


def scale_diagnostics(population: str, target: str, partition: str,
                      truth: np.ndarray, rmse: float) -> dict:
    centered = (truth - truth.mean()) ** 2
    std = float(truth.std())
    quantiles = np.quantile(truth, [0.01, 0.99, 0.999])
    return {
        "population": population, "target": target, "partition": partition,
        "n": len(truth), "min": float(truth.min()), "max": float(truth.max()),
        "mean": float(truth.mean()), "std": std, "median": float(np.median(truth)),
        "q01": float(quantiles[0]), "q99": float(quantiles[1]), "q999": float(quantiles[2]),
        "max_abs": float(np.abs(truth).max()),
        "top1_fraction_of_centered_ss": float(centered.max()/centered.sum()) if centered.sum() > 0 else None,
        "rmse_over_target_std": rmse / std if std > 0 else None,
    }


def run(args: argparse.Namespace) -> None:
    began = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    frame = pd.read_parquet(args.data, columns=INPUTS)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Input columns must be finite.")
    if not ((frame.FunctionValue > 0).all() and (frame.taui > 0).all()
            and (frame.Z1i >= 1).all() and (frame.Z2i >= 1).all()):
        raise ValueError("Expected positive saved potential, Im(tau)>0, and Im(Z1),Im(Z2)>=1.")
    # Ignore any stored pandas index: source_row always means physical input row.
    frame.index = pd.RangeIndex(len(frame), name="source_row")
    frame = frame.sort_values("FunctionValue", kind="stable")
    susy, W0_validation = prepare_susy(frame)
    populations = [
        ("all", frame, INPUTS, INPUTS),
        ("susy", susy, ["W0_abs"] + INPUTS[1:], INPUTS + ["W0_abs"]),
    ]
    prepared_splits = {}
    for population, data, _, _ in populations:
        train, test = train_test_split(np.arange(len(data)), **SPLIT_PARAMETERS)
        if len(train) < RF_PARAMETERS["max_samples"]:
            raise ValueError(f"Population {population} needs at least 20,000 training rows for the published forest settings.")
        prepared_splits[population] = (train, test)

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "W0_validation.json", W0_validation)
    records, importances, diagnostics = [], [], []
    sample_info, input_columns = {}, {}
    max_metric_error = 0.0
    parameters = {**RF_PARAMETERS, "n_jobs": args.threads}
    print(f"Read {len(frame):,} rows; numerical SUSY subset: {len(susy):,}; W0 check passed.", flush=True)
    for population, data, targets, features in populations:
        train, test = prepared_splits[population]
        source_rows = data.index.to_numpy()
        sample_info[population] = {"total": len(data), "train": len(train), "test": len(test)}
        is_test = np.zeros(len(data), dtype=bool)
        is_test[test] = True
        # train/test are positions in the stable-V-sorted population. Both the
        # full source-row map and the ordered partition source rows are saved.
        np.savez_compressed(args.output / f"{population}_split_indices.npz",
                            train=train, test=test, source_row=source_rows,
                            train_source_row=source_rows[train], test_source_row=source_rows[test])
        pd.DataFrame({"source_row": source_rows, "is_test": is_test}).to_parquet(
            args.output / f"{population}_split.parquet", index=False)
        input_columns[population] = {}
        for target in targets:
            fit_start = time.monotonic()
            cols = [column for column in features if column != target]
            if (target in cols or set(cols) & {"eK", "W0_real", "W0_imag", "log10_W0_abs"}
                    or (("W0_abs" in cols) != (population == "susy" and target != "W0_abs"))):
                raise RuntimeError("Unexpected leakage feature configuration.")
            input_columns[population][target] = cols
            X, y = data[cols], data[target].to_numpy()
            model = RandomForestRegressor(**parameters)
            model.fit(X.iloc[train], y[train])
            prediction = model.predict(X)
            record = {"population": population, "target": target, "n_features": len(cols)}
            for partition, positions in [("train", train), ("test", test)]:
                truth, estimate = y[positions], prediction[positions]
                record[f"r2_{partition}"] = float(r2_score(truth, estimate))
                record[f"rmse_{partition}"] = float(np.sqrt(mean_squared_error(truth, estimate)))
                record[f"mae_{partition}"] = float(mean_absolute_error(truth, estimate))
                denominator = np.sum((truth - truth.mean()) ** 2)
                if denominator == 0:
                    raise ValueError(f"R2 is undefined for constant target {population}/{target}/{partition}.")
                direct_r2 = 1 - np.sum((truth - estimate) ** 2) / denominator
                error = abs(direct_r2 - record[f"r2_{partition}"])
                max_metric_error = max(max_metric_error, error)
                np.testing.assert_allclose(direct_r2, record[f"r2_{partition}"], rtol=1e-13, atol=1e-13)
                diagnostics.append(scale_diagnostics(population, target, partition,
                                                     truth, record[f"rmse_{partition}"]))
            record["elapsed_seconds"] = time.monotonic() - fit_start
            records.append(record)
            importances.extend({"population": population, "target": target,
                                "feature": feature, "importance": float(importance)}
                               for feature, importance in zip(cols, model.feature_importances_))
            if args.save_predictions:
                pd.DataFrame({"source_row": source_rows, "true": y,
                              "predicted": prediction, "is_test": is_test}).to_parquet(
                    args.output / f"{population}_{target}_predictions.parquet", index=False)
            print(f"{population:5s} {target:13s} train={record['r2_train']:.9f} "
                  f"test={record['r2_test']:.9f} ({record['elapsed_seconds']:.1f}s)", flush=True)

    if len(records) != 26 or "FunctionValue" in input_columns["susy"]:
        raise RuntimeError("Expected all 13 full-sample and 13 SUSY regression tasks.")
    scores = pd.DataFrame(records)
    scores.to_csv(args.output / "scores.csv", index=False)
    pd.DataFrame(importances).to_csv(args.output / "feature_importances.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / "target_scale_diagnostics.csv", index=False)
    write_json(args.output / "input_columns.json", input_columns)
    numeric_summary = {
        "flux_test_r2_range": {}, "targets_with_train_and_test_r2_above_0_5": {},
    }
    for population in ["all", "susy"]:
        subset = scores.loc[scores.population == population]
        flux = subset.loc[subset.target.isin(INPUTS[-6:])]
        numeric_summary["flux_test_r2_range"][population] = [float(flux.r2_test.min()), float(flux.r2_test.max())]
        numeric_summary["targets_with_train_and_test_r2_above_0_5"][population] = subset.loc[
            (subset.r2_train > 0.5) & (subset.r2_test > 0.5), "target"].tolist()
    w0 = scores.loc[(scores.population == "susy") & (scores.target == "W0_abs")].iloc[0]
    numeric_summary["W0_abs_r2"] = {"train": float(w0.r2_train), "test": float(w0.r2_test)}
    results = {
        "source_file": args.data.name, "full_rows": len(frame), "SUSY_rows": len(susy),
        "SUSY_criterion": "FunctionValue < 10**(-4.5)", "threshold": SUSY_THRESHOLD,
        "populations": sample_info, "parameters": parameters, "scores": records,
        "summary": numeric_summary, "all_models_fitted_from_input": True,
        "all_targets": INPUTS, "SUSY_targets": ["W0_abs"] + INPUTS[1:],
        "W0_validation": W0_validation, "maximum_independent_R2_error": max_metric_error,
    }
    write_json(args.output / "results.json", results)
    manifest = {
        "analysis": "Conditional prediction among recorded variables", "started_at_utc": started_at,
        "source_file": args.data.name, "populations": sample_info,
        "selection": "All input rows, plus the numerical SUSY subset FunctionValue < 10**(-4.5); no further filtering.",
        "source_row_definition": "Zero-based physical row of the input Parquet, ignoring any stored pandas index.",
        "sort": "Stable ascending saved FunctionValue; ties preserve physical input row order.",
        "split": SPLIT_PARAMETERS, "split_indices_definition": "Positions within the stable-V-sorted population; each NPZ also contains source-row mappings.",
        "weighting": "Each retained solution row has equal weight; multiple rows may share a flux choice.",
        "coordinates": "Saved representative coordinates, fluxes and potential; no remapping or minimization.",
        "target_transform": "None; original numerical units.",
        "resolved_estimator_parameters": RandomForestRegressor(**parameters).get_params(),
        "input_columns": input_columns, "all_models_fitted_from_input": True,
        "W0_definition": "sqrt(2/pi)*sqrt(eK)*abs(W); polynomial prepotential including zeta(3), no instantons, Kahler-volume factor suppressed.",
        "W0_rules": "Only for SUSY: target or input; V remains an input and is not a SUSY target. No eK, real/imaginary W0 or transformed target copies.",
        "validation": {"finite_inputs": True, "target_excluded_from_features": True,
                       "no_auxiliary_W0_leakage_features": True, "W0_independent_formula_check": True,
                       "all_R2_independently_checked": True,
                       "reference_population_counts_match": len(frame) == 1296633 and len(susy) == 127241},
        "predictions_saved": args.save_predictions,
        "environment": {
            "python": sys.version, "implementation": platform.python_implementation(),
            "system": platform.system(), "machine": platform.machine(),
            "packages": {module.__name__: module.__version__ for module in [np, pd, pyarrow, scipy, sklearn]},
            "threads": args.threads,
            "threadpools": [{key: pool.get(key) for key in ["user_api", "internal_api", "prefix", "version", "num_threads"]}
                            for pool in threadpool_info()],
        },
        "elapsed_seconds": time.monotonic() - began,
    }
    write_json(args.output / "manifest.json", manifest)
    print(f"Completed all 26 regressions in {manifest['elapsed_seconds']:.1f}s.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Selected-solution Parquet (13 recorded input columns).")
    parser.add_argument("--output", type=Path, required=True, help="Directory for scores, splits and reproducibility metadata.")
    parser.add_argument("--threads", type=int, default=2, help="Worker and numeric-library thread limit (default: 2).")
    parser.add_argument("--save-predictions", action="store_true", help="Also save individual train and test predictions for all 26 tasks.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    pyarrow.set_cpu_count(args.threads)
    pyarrow.set_io_thread_count(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
