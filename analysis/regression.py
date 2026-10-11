"""Reproduce potential and moduli regression on the selected solution sample.

Run: python -m analysis.regression --data data/selected_solutions.parquet
     --output results/regression [--save-predictions]
     [--reuse-logv-models results/previous_regression]
No quantile trimming, row averaging, or further physical cut is applied.
All targets use the same full-sample training and test solution rows.
"""
from pathlib import Path
import argparse
import os
import sys
import json
import shutil
import time
import platform

HERE = None
SOURCE = None
SAVE_PREDICTIONS = False
from ._ml_io import check_sample, write_json
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLUX = ["m1", "m2", "m3", "n1", "n2", "n3"]
MODULI = ["taur", "taui", "Z1r", "Z1i", "Z2r", "Z2i"]
INVARIANTS = ["omega12", "omega13", "omega23", "G11", "G12", "G13", "G22", "G23", "G33"]
BASE_PARAMS = dict(max_depth=8, min_child_weight=5, subsample=0.8,
                   colsample_bytree=0.8, random_state=0, n_jobs=2,
                   tree_method="hist", objective="reg:squarederror")


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Selected-solution Parquet, in its original row order.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for figures, summaries, models and split indices.")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save all test predictions for independent verification.")
    parser.add_argument("--reuse-logv-models", type=Path,
                        help="Reuse unchanged potential models after checking settings, splits and reproduced scores.")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Input Parquet does not exist: {args.data}")
    if args.reuse_logv_models is not None and args.output.resolve() == args.reuse_logv_models.resolve():
        parser.error("The new output must differ from the directory of reused models.")
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def validate_reused_logv(directory, results, train_rows, test_rows):
    previous = json.loads((directory / "metrics.json").read_text())
    for key in ["source_rows", "zero_flux_rows_excluded", "logV_rows", "logV_train_rows",
                "logV_test_rows", "split", "feature_order", "base_parameters", "logV_parameters", "versions"]:
        assert previous[key] == results[key], f"Reused potential metadata differ: {key}"
    with np.load(directory / "splits.npz") as saved:
        assert np.array_equal(saved["logV_train_source_rows"], train_rows)
        assert np.array_equal(saved["logV_test_source_rows"], test_rows)
    return previous


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def feature_arrays(frame):
    raw64 = frame[FLUX].to_numpy(dtype=np.float64)
    m, n = raw64[:, :3], raw64[:, 3:]
    inv64 = np.column_stack([
        m[:, 0] * n[:, 1] - m[:, 1] * n[:, 0],
        m[:, 0] * n[:, 2] - m[:, 2] * n[:, 0],
        m[:, 1] * n[:, 2] - m[:, 2] * n[:, 1],
        m[:, 0] ** 2 + n[:, 0] ** 2,
        m[:, 0] * m[:, 1] + n[:, 0] * n[:, 1],
        m[:, 0] * m[:, 2] + n[:, 0] * n[:, 2],
        m[:, 1] ** 2 + n[:, 1] ** 2,
        m[:, 1] * m[:, 2] + n[:, 1] * n[:, 2],
        m[:, 2] ** 2 + n[:, 2] ** 2,
    ])
    coords = frame[MODULI].to_numpy(dtype=np.float64)
    mods64 = np.column_stack([coords, signed_log(coords)])
    arrays = {"raw6": raw64, "invariants9": inv64, "moduli12": mods64}
    cast_report = {}
    for name, a in arrays.items():
        assert np.isfinite(a).all(), name
        assert np.max(np.abs(a)) < np.finfo(np.float32).max, name
        cast = a.astype(np.float32)
        assert np.isfinite(cast).all(), name
        cast_report[name] = {"max_absolute_float64": float(np.max(np.abs(a))),
                             "max_cast_absolute_difference": float(np.max(np.abs(a - cast))),
                             "finite_after_float32_cast": True}
        arrays[name] = cast
    return arrays, cast_report


def metrics(y, pred):
    return {"r2": float(r2_score(y, pred)),
            "mae": float(mean_absolute_error(y, pred))}


def persist_metrics(results):
    (HERE / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


def make_figure(results, predictions):
    labels = [r"$\mathrm{Re}\,\tau$", r"$\mathrm{Im}\,\tau$", r"$\mathrm{Re}\,Z_1$",
              r"$\mathrm{Im}\,Z_1$", r"$\mathrm{Re}\,Z_2$", r"$\mathrm{Im}\,Z_2$"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
    target = max(MODULI, key=lambda c: results["moduli"][c]["invariants9"]["r2"])
    xx = predictions[f"{target}_true_transformed"].to_numpy()
    yy = predictions[f"{target}_pred_invariants9_transformed"].to_numpy()
    hb = ax.hexbin(xx, yy, gridsize=75, mincnt=1, bins="log", cmap="viridis", linewidths=0)
    plotted_count = int(hb.get_array().sum())
    assert plotted_count == len(xx)
    lo, hi = min(xx.min(), yy.min()), max(xx.max(), yy.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#333333", linewidth=1)
    padding = (hi - lo) * 0.025
    ax.set_xlim(lo - padding, hi + padding)
    ax.set_ylim(lo - padding, hi + padding)
    coordinate = labels[MODULI.index(target)].strip("$")
    transformed_label = (rf"$\log_{{10}}({coordinate})$" if target.endswith("i") else
                         rf"$\mathrm{{sign}}({coordinate})\ln(1+|{coordinate}|)$")
    ax.set_xlabel("True " + transformed_label)
    ax.set_ylabel("Predicted " + transformed_label)
    ax.set_title(r"$G_{AB},\,\omega_{AB}$: " + labels[MODULI.index(target)])
    fig.colorbar(hb, ax=ax, label="Test solutions per bin")
    fig.savefig(HERE / "ml_moduli_prediction.pdf")
    fig.savefig(HERE / "ml_moduli_prediction.png", dpi=180)
    plt.close(fig)
    results["figure"] = {"target": target, "panels": 1,
                         "all_test_rows_plotted": len(xx), "hexbin_count": plotted_count}


def main():
    global HERE, SOURCE, SAVE_PREDICTIONS
    args = parse_cli()
    HERE, SOURCE, SAVE_PREDICTIONS = args.output, args.data, args.save_predictions
    import pyarrow
    pyarrow.set_cpu_count(2)
    pyarrow.set_io_thread_count(2)
    started = time.time()
    data = pd.read_parquet(SOURCE, columns=["FunctionValue"] + MODULI + FLUX)
    check_sample(data, ["FunctionValue"] + MODULI + FLUX)
    nonzero = np.any(data[FLUX].to_numpy() != 0, axis=1)
    rows = np.flatnonzero(nonzero)
    assert nonzero.all(), "The full reliable sample must not contain zero-flux solutions"
    assert (data.FunctionValue.iloc[rows] > 0).all(), "No log clipping allowed; inspect nonpositive V"
    assert (data[["taui", "Z1i", "Z2i"]].to_numpy() > 0).all()
    train, test = train_test_split(np.arange(len(rows)), test_size=0.25, random_state=0)
    np.savez_compressed(HERE / "splits.npz", logV_train_source_rows=rows[train],
                        logV_test_source_rows=rows[test], moduli_train_source_rows=rows[train],
                        moduli_test_source_rows=rows[test])
    arrays, cast_report = feature_arrays(data)
    y = np.log10(data.FunctionValue.to_numpy()[rows])
    results = {"source": str(SOURCE),
               "save_predictions": SAVE_PREDICTIONS,
               "source_row_convention": "Zero-based physical Parquet row position; no reordering before splitting",
               "source_rows": len(data), "zero_flux_rows_excluded": int((~nonzero).sum()),
               "logV_rows": len(rows), "logV_train_rows": len(train), "logV_test_rows": len(test),
               "moduli_rows": len(rows), "moduli_train_rows": len(train), "moduli_test_rows": len(test),
               "moduli_sample": "All reliable solution rows; no potential-value cut",
               "split": {"test_size": 0.25, "random_state": 0,
               "stratify": None, "unit": "solution row", "same_rows_across_models": True},
               "interpretation": "Solution-weighted interpolation; multiple solutions per flux retained; row split can share fluxes and invariant tuples",
               "feature_order": {"raw6": FLUX, "invariants9": INVARIANTS,
                                 "moduli12": MODULI + ["signed_log_" + c for c in MODULI]},
               "float32_checks": cast_report, "base_parameters": BASE_PARAMS,
               "logV_parameters": {"n_estimators": 300, "learning_rate": 0.15},
               "moduli_parameters": {"n_estimators": 400, "learning_rate": 0.1},
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "matplotlib": matplotlib.__version__},
               "logV": {}, "moduli": {}}
    previous = None
    if args.reuse_logv_models is not None:
        previous = validate_reused_logv(args.reuse_logv_models, results, rows[train], rows[test])
        results["logV_model_reuse"] = {
            "directory": str(args.reuse_logv_models.resolve()),
            "validation": "Identical sample size, settings, package versions and source-row splits; all test scores recomputed",
            "models": {}}
    persist_metrics(results)
    log_predictions = pd.DataFrame({"source_row": rows[test], "V_true": data.FunctionValue.to_numpy()[rows[test]], "log10V_true": y[test]})
    specifications = {"invariants9": ["invariants9"], "invariants9_moduli12": ["invariants9", "moduli12"],
                      "raw6_moduli12": ["raw6", "moduli12"], "raw6_invariants9_moduli12": ["raw6", "invariants9", "moduli12"]}
    for name, keys in specifications.items():
        start = time.time()
        X = np.column_stack([arrays[k] for k in keys])[rows]
        if previous is None:
            model = XGBRegressor(**BASE_PARAMS, n_estimators=300, learning_rate=0.15)
            print(f"FIT logV {name} {X.shape}", flush=True)
            model.fit(X[train], y[train])
            pred = model.predict(X[test])
            # Native booster serialization avoids the sklearn-1.8 removal of the
            # legacy _estimator_type attribute; it does not change fit/predict.
            model.get_booster().save_model(HERE / f"model_logV_{name}.ubj")
            write_json(HERE / f"model_logV_{name}.configuration.json",
                       json.loads(model.get_booster().save_config()))
        else:
            print(f"REUSE logV {name} {X.shape}", flush=True)
            original = args.reuse_logv_models / f"model_logV_{name}.ubj"
            model = xgboost.Booster(params={"nthread": 2})
            model.load_model(original)
            pred = model.predict(xgboost.DMatrix(X[test], nthread=2))
            for score, value in metrics(y[test], pred).items():
                assert np.isclose(value, previous["logV"][name][score], rtol=1e-12, atol=1e-12), (name, score)
            for extension in ["ubj", "configuration.json"]:
                filename = f"model_logV_{name}.{extension}"
                shutil.copy2(args.reuse_logv_models / filename, HERE / filename)
            results["logV_model_reuse"]["models"][name] = {
                "original_fit_seconds": previous["logV"][name]["seconds"],
                "recomputed_scores_match_original": True}
        results["logV"][name] = dict(metrics(y[test], pred), feature_count=X.shape[1], seconds=time.time() - start)
        log_predictions[f"log10V_pred_{name}"] = pred
        if SAVE_PREDICTIONS:
            log_predictions.to_parquet(HERE / "logV_test_predictions.parquet", index=False)
        persist_metrics(results)
        print(name, results["logV"][name], flush=True)
        del X, model
    mod_predictions = pd.DataFrame({"source_row": rows[test]})
    for target in MODULI:
        physical = data[target].to_numpy()[rows]
        yt = np.log10(physical) if target.endswith("i") else signed_log(physical)
        results["moduli"][target] = {}
        mod_predictions[f"{target}_true_physical"] = physical[test]
        mod_predictions[f"{target}_true_transformed"] = yt[test]
        for name in ["raw6", "invariants9"]:
            start = time.time()
            X = arrays[name][rows]
            model = XGBRegressor(**BASE_PARAMS, n_estimators=400, learning_rate=0.1)
            print(f"FIT {target} {name} {X.shape}", flush=True)
            model.fit(X[train], yt[train])
            pred = model.predict(X[test]).astype(np.float64)
            physical_pred = np.power(10.0, pred) if target.endswith("i") else np.sign(pred) * np.expm1(np.abs(pred))
            model.get_booster().save_model(HERE / f"model_{target}_{name}.ubj")
            write_json(HERE / f"model_{target}_{name}.configuration.json",
                       json.loads(model.get_booster().save_config()))
            out = dict(metrics(yt[test], pred),
                       transform="log10" if target.endswith("i") else "sign(x)*ln(1+abs(x))",
                       seconds=time.time() - start)
            results["moduli"][target][name] = out
            mod_predictions[f"{target}_pred_{name}_transformed"] = pred
            mod_predictions[f"{target}_pred_{name}_physical"] = physical_pred
            if SAVE_PREDICTIONS:
                mod_predictions.to_parquet(HERE / "moduli_test_predictions.parquet", index=False)
            persist_metrics(results)
            print(target, name, out, flush=True)
            del X, model
    make_figure(results, mod_predictions)
    results["elapsed_seconds"] = time.time() - started
    _, flux_id = np.unique(data[FLUX].to_numpy(dtype=np.int32), axis=0, return_inverse=True)
    flux_id = np.asarray(flux_id).reshape(-1)
    results["exact_flux_overlap"] = {}
    for task, train_rows, test_rows in [("logV", rows[train], rows[test]),
                                       ("moduli", rows[train], rows[test])]:
        seen = np.zeros(int(flux_id.max()) + 1, dtype=bool)
        seen[flux_id[train_rows]] = True
        shared = seen[flux_id[test_rows]]
        results["exact_flux_overlap"][task] = dict(test_rows=len(test_rows),
            test_rows_sharing_exact_flux_with_training=int(shared.sum()),
            test_shared_flux_fraction=float(shared.mean()))
    pd.DataFrame([{"input": name, **{k: v for k, v in row.items() if k != "seconds"}}
                  for name, row in results["logV"].items()]).to_csv(HERE / "logV_scores.csv", index=False)
    pd.DataFrame([{"target": target, "input": name,
                   **{k: v for k, v in row.items() if k != "seconds"}}
                  for target, models in results["moduli"].items()
                  for name, row in models.items()]).to_csv(HERE / "moduli_scores.csv", index=False)
    write_json(HERE / "settings.json", {key: results[key] for key in
        ["source", "save_predictions", "source_row_convention",
         "source_rows", "moduli_sample", "moduli_rows", "moduli_train_rows", "moduli_test_rows",
         "split", "feature_order", "base_parameters", "logV_parameters", "moduli_parameters", "versions"]})
    persist_metrics(results)
    print("DONE", results["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
