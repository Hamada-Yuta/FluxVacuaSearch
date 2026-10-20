"""Reproduce SUSY classification, ranking and input-representation comparisons.

All models use the paper settings. Boundary-score ties are averaged uniformly.
Run: python -m analysis.classification --data data/selected_solutions.parquet
     --output results/classification [--save-predictions]
"""
from pathlib import Path
import os
import sys

HERE = None
SAVE_PREDICTIONS = False
from ._ml_io import parse_cli, environment, check_sample
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import csv
import json
import platform
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb

BUDGETS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
THRESHOLD = 10.0 ** -4.5
PARAMS = dict(max_depth=8, learning_rate=0.15, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, random_state=0,
              tree_method="hist", n_jobs=2, objective="binary:logistic",
              eval_metric="logloss")


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def save_csv(name, rows):
    if rows:
        pd.DataFrame(rows).to_csv(HERE / name, index=False)


def json_safe(value):
    if isinstance(value,dict):return {key:json_safe(val) for key,val in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(val) for val in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    return value


def budget_metrics(y, p, fractions=BUDGETS):
    """Expected positives with exactly floor(b*n) choices, averaging ties."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p)
    n, positives = len(y), int(y.sum())
    order = np.argsort(-p, kind="stable")
    scores, labels = p[order], y[order]
    cumulative = np.concatenate(([0], np.cumsum(labels, dtype=np.int64)))
    rows = []
    for fraction in fractions:
        k = min(n, max(1, int(np.floor(fraction * n))))
        boundary = scores[k - 1]
        # Search on ascending -scores avoids full-array scans for each budget.
        left = int(np.searchsorted(-scores, -boundary, side="left"))
        right = int(np.searchsorted(-scores, -boundary, side="right"))
        tie_positives = int(cumulative[right] - cumulative[left])
        expected = float(cumulative[left] + (k - left) * tie_positives / (right - left))
        rows.append(dict(budget=fraction, selected_rows=k,
                         actual_budget=k / n, expected_low_V=expected,
                         recall=expected / positives if positives else None,
                         ideal_recall_upper_bound=min(1.0,k / positives) if positives else None,
                         precision=expected / k,
                         threshold_score=float(boundary),
                         boundary_tie_rows=right-left,
                         boundary_tie_selected=k-left,
                         boundary_tie_positives=tie_positives))
    return rows


def metrics(y, p):
    y = np.asarray(y, dtype=np.int8)
    n, positive = len(y), int(y.sum())
    base = positive / n
    return dict(rows=n, positives=positive, base_rate=base,
                roc_auc=float(roc_auc_score(y, p)) if 0 < positive < n else None,
                average_precision=float(average_precision_score(y, p)) if positive else None,
                brier_score=float(brier_score_loss(y, p)),
                constant_test_prevalence_brier=base*(1-base),
                recall_at_5percent=budget_metrics(y, p, [0.05])[0]["recall"])


def fit_predict(tag, features, names, y, train, test, trees):
    started = time.time()
    model_path = HERE / "models" / f"{tag}.ubj"
    model_path.parent.mkdir(exist_ok=True)
    configuration_path = HERE / "models" / f"{tag}.configuration.json"
    print(f"Fitting {tag}: {len(train):,} train / {len(test):,} test / {len(names)} features", flush=True)
    model = xgb.XGBClassifier(n_estimators=trees, **PARAMS)
    model.fit(features[train], y[train])
    predictions = model.predict_proba(features[test])[:, 1]
    assert np.isfinite(predictions).all()
    configuration = dict(feature_names=names, n_estimators=trees,
                         train_rows=len(train), test_rows=len(test),
                         training_positives=int(y[train].sum()),
                         evaluation_positives=int(y[test].sum()),
                         sklearn_get_params=json_safe(model.get_params(deep=True)),
                         xgb_get_params=json_safe(model.get_xgb_params()),
                         booster_save_config=json.loads(model.get_booster().save_config()))
    configuration_path.write_text(json.dumps(configuration, indent=2, allow_nan=False)+"\n")
    # Native Booster serialization also works with scikit-learn 1.8.
    model.get_booster().save_model(model_path)
    if SAVE_PREDICTIONS:
        prediction_path = HERE / "predictions" / f"{tag}.npz"
        prediction_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(prediction_path, row_index=test,
                            probability=predictions, label=y[test])
    print(f"Completed {tag}: {time.time()-started:.1f}s; {metrics(y[test], predictions)}", flush=True)
    return model, predictions


def feature_sets(raw, inv, inv_names):
    m, n = raw[:, :3], raw[:, 3:]
    omega = np.column_stack([m[:,a]*n[:,b]-m[:,b]*n[:,a]
                             for a,b in ((0,1),(0,2),(1,2))]).astype(np.float32)
    norm_m, norm_n = (m*m).sum(axis=1), (n*n).sum(axis=1)
    norm = norm_m + norm_n
    raw_names = ["m1","m2","m3","n1","n2","n3"]
    reduced_names = ["omega12", "omega13", "omega23", "norm2"]
    reduced = np.column_stack((omega, norm)).astype(np.float32)
    return {
        "raw": (raw, raw_names),
        "reduced_invariants": (reduced, reduced_names),
        "invariants": (inv, inv_names),
        "raw_reduced_invariants": (np.column_stack((raw,reduced)),raw_names+reduced_names),
        "raw_invariants": (np.column_stack((raw,inv)),raw_names+inv_names),
    }


def matched_subsample(indices, n, labels, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == n:
        return indices
    kept, _ = train_test_split(indices, train_size=n, random_state=seed,
                              stratify=labels[indices])
    return np.asarray(kept, dtype=np.int64)


def split_overlap(group_id, training, evaluation, y):
    seen=np.zeros(int(group_id.max())+1,dtype=bool)
    seen[group_id[training]]=True
    shared=seen[group_id[evaluation]]
    return dict(shared_test_rows=int(shared.sum()),test_rows=len(evaluation),
                shared_test_fraction=float(shared.mean()),
                shared_test_positive_rows=int((shared & (y[evaluation]==1)).sum()),
                test_positive_rows=int(y[evaluation].sum()))


def make_nested_samples(train, y, sizes):
    """Random order within labels; every successive sample contains the former."""
    rng = np.random.default_rng(0)
    positive = rng.permutation(train[y[train] == 1])
    negative = rng.permutation(train[y[train] == 0])
    result = []
    previous = set()
    for size in sizes:
        count_positive = int(round(size * len(positive) / len(train)))
        chosen = np.concatenate((positive[:count_positive], negative[:size-count_positive]))
        chosen = rng.permutation(chosen)
        assert len(chosen) == size
        assert previous.issubset(set(chosen.tolist()))
        previous = set(chosen.tolist())
        result.append(chosen)
    return result


def shap_attributions(model, features, names, test):
    sample = np.random.default_rng(0).choice(test,size=min(8000,len(test)),replace=False)
    data = features[sample]
    matrix=xgb.DMatrix(data,nthread=2)
    contributions=model.get_booster().predict(matrix,pred_contribs=True,approx_contribs=False)
    margin=model.get_booster().predict(matrix,output_margin=True)
    error=np.abs(contributions.sum(axis=1)-margin)
    assert np.allclose(contributions.sum(axis=1),margin,rtol=1e-5,atol=1e-4)
    means=np.abs(contributions[:,:-1]).mean(axis=0)
    rows=[dict(feature=name,mean_abs_shap=float(value)) for name,value in zip(names,means)]
    save_csv("shap.csv",rows)
    np.savez_compressed(HERE/"shap_sample_rows.npz", row_index=sample)
    if SAVE_PREDICTIONS:
        np.savez_compressed(HERE/"shap_values.npz",row_index=sample,features=data,contributions=contributions,margin=margin)
    order=np.argsort(means)
    fig,ax=plt.subplots(figsize=(7.6,4.9),constrained_layout=True)
    display={"omega12":r"$\omega_{12}$","omega13":r"$\omega_{13}$","omega23":r"$\omega_{23}$"}
    display.update({f"G{a}{b}":rf"$G_{{{a}{b}}}$" for a,b in ((1,1),(1,2),(1,3),(2,2),(2,3),(3,3))})
    ax.barh([display.get(names[i],names[i]) for i in order],means[order],color="#2466a8")
    ax.set_xlabel("Mean absolute TreeSHAP contribution (log-odds)")
    ax.set_title(f"{len(sample):,} solutions from the test set")
    ax.grid(axis="x",alpha=.2)
    fig.savefig(HERE/"ml_shap.pdf");fig.savefig(HERE/"ml_shap.png",dpi=160);plt.close(fig)
    return dict(sample_rows=len(sample),seed=0,method="XGBoost exact TreeSHAP pred_contribs=True approx_contribs=False",
                additive_reconstruction_max_abs_error=float(error.max()),attributions=rows)


def run_analysis(raw, inv, inv_names, potential, train, test, group_train, group_test,
                 source_path, main_choice="invariants"):
    started=time.time()
    y=(potential < THRESHOLD).astype(np.int8)
    train=np.asarray(train,dtype=np.int64);test=np.asarray(test,dtype=np.int64)
    assert len(np.intersect1d(train,test)) == 0
    assert len(train)+len(test) == len(y) == 1296633
    assert np.array_equal(np.sort(np.concatenate((train,test))),np.arange(len(y)))
    _,raw_id=np.unique(raw,axis=0,return_inverse=True)
    _,inv_id=np.unique(inv,axis=0,return_inverse=True)
    raw_id=raw_id.reshape(-1);inv_id=inv_id.reshape(-1)
    integrity={}
    for tag,(training,evaluation) in {"main_row_split":(train,test),"whole_tuple_split":(group_train,group_test)}.items():
        assert len(np.intersect1d(training,evaluation))==0
        integrity[tag]={"rows_disjoint":True,"raw_flux":split_overlap(raw_id,training,evaluation,y),
                        "invariants":split_overlap(inv_id,training,evaluation,y)}
        if tag=="whole_tuple_split":
            assert integrity[tag]["raw_flux"]["shared_test_rows"]==0
            assert integrity[tag]["invariants"]["shared_test_rows"]==0
    dump("split_integrity.json",integrity)
    inner_train,validation=train_test_split(train,test_size=.25,random_state=0,stratify=y[train])
    np.savez_compressed(HERE/"splits.npz",train=train,test=test,inner_train=inner_train,validation=validation)
    feature_map=feature_sets(np.asarray(raw,dtype=np.float32),np.asarray(inv,dtype=np.float32),inv_names)
    results=dict(source=str(source_path),
                 target="V < 10**(-4.5)",threshold=THRESHOLD,
                 rows=len(y),positives=int(y.sum()),base_rate=float(y.mean()),
                 main_split=dict(train_rows=len(train),test_rows=len(test),
                                 training_positives=int(y[train].sum()),test_positives=int(y[test].sum()),
                                 split="75/25 endpoint-row stratified random_state=0"),
                 validation_split=dict(train_rows=len(inner_train),validation_rows=len(validation),
                                       train_positives=int(y[inner_train].sum()),validation_positives=int(y[validation].sum()),
                                       split="75/25 stratified inside main training set only, random_state=0"),
                 budget_rule="floor(b*n) rows, expected positives under uniform ordering inside boundary score ties",
                 parameters=PARAMS)
    ablation=[]
    for tag,(features,names) in feature_map.items():
        _,p=fit_predict(f"ablation_{tag}",features,names,y,inner_train,validation,300)
        ablation.append(dict(feature_set=tag,features=len(names),**metrics(y[validation],p)))
        save_csv("feature_ablation.csv",ablation)
    winner=max(ablation,key=lambda row:row["average_precision"])["feature_set"]
    selected=winner if main_choice=="validation_best" else main_choice
    assert selected in feature_map
    results.update(feature_ablation=ablation,validation_AP_winner=winner,main_feature_set=selected,
                   main_choice_rule=main_choice)
    dump("results.partial.json",results)
    features,names=feature_map[selected]
    model,p=fit_predict("main",features,names,y,train,test,400)
    results["headline_metrics"]=metrics(y[test],p)
    results["headline_metrics"]["training_prevalence_constant_brier"]=float(np.mean((y[test]-y[train].mean())**2))
    budgets=budget_metrics(y[test],p)
    save_csv("budget.csv",budgets);results["budgets"]=budgets
    robustness=[]
    for exponent in [-4.5,-5,-6,-7,-8]:
        target=(potential[test] < 10.0**exponent).astype(np.int8)
        robustness.append(dict(log10_threshold=exponent,threshold=10.0**exponent,**metrics(target,p)))
    save_csv("threshold_robustness.csv",robustness);results["threshold_robustness"]=robustness
    results["shap"]=shap_attributions(model,features,names,test)
    dump("results.partial.json",results)
    # Fix the main test set and features while enlarging nested training subsets.
    sizes=sorted(set(n for n in [10000,40000,160000,len(train)] if n<=len(train)))
    samples=make_nested_samples(train,y,sizes)
    np.savez_compressed(HERE/"learning_curve_splits.npz",**{f"train_{n}":s for n,s in zip(sizes,samples)},test=test)
    learning=[]
    for n,subset in zip(sizes,samples):
        _,prediction=fit_predict(f"learning_{n}",features,names,y,subset,test,300)
        learning.append(dict(training_rows=n,training_positives=int(y[subset].sum()),**metrics(y[test],prediction)))
        save_csv("learning_curve.csv",learning)
    results["learning_curve"]=learning
    # Equal row counts in the row-random and whole-invariant-tuple split.
    n_train=min(len(train),len(group_train));n_test=min(len(test),len(group_test))
    split_sets={"random_rows":(matched_subsample(train,n_train,y,0),matched_subsample(test,n_test,y,1)),
                "invariant_groups":(matched_subsample(group_train,n_train,y,0),matched_subsample(group_test,n_test,y,1))}
    for tag,(training,evaluation) in split_sets.items():
        assert len(np.intersect1d(training,evaluation))==0
        integrity[f"matched_{tag}"]={"rows_disjoint":True,"raw_flux":split_overlap(raw_id,training,evaluation,y),
                                     "invariants":split_overlap(inv_id,training,evaluation,y)}
        if tag=="invariant_groups":
            assert integrity[f"matched_{tag}"]["raw_flux"]["shared_test_rows"]==0
            assert integrity[f"matched_{tag}"]["invariants"]["shared_test_rows"]==0
    dump("split_integrity.json",integrity)
    np.savez_compressed(HERE/"matched_group_splits.npz",**{f"{tag}_{part}":idx for tag,pair in split_sets.items() for part,idx in zip(["train","test"],pair)})
    grouped=[]
    for split,(training,evaluation) in split_sets.items():
        for tag in ["raw","invariants"]:
            x,labels=feature_map[tag]
            _,prediction=fit_predict(f"group_comparison_{split}_{tag}",x,labels,y,training,evaluation,300)
            grouped.append(dict(split=split,feature_set=tag,training_rows=len(training),training_positives=int(y[training].sum()),**metrics(y[evaluation],prediction)))
            save_csv("grouped_comparison.csv",grouped)
    results["grouped_comparison"]=grouped
    results["group_split_unsubsampled"]=dict(train_rows=len(group_train),test_rows=len(group_test),
                                             design="75/25 invariant tuples stratified by any-positive tuple, seed=0; endpoints kept together")
    results["elapsed_seconds"]=time.time()-started
    dump("results.json",results)
    provenance=dict(source=str(source_path),
                    environment=environment([np,pd,matplotlib,sklearn,xgb]),
                    n_jobs=2, model_parameters=PARAMS,
                    seeds={"model":0,"main_row_split":0,"validation_split":0,
                           "whole_tuple_split":0,"matched_train_subsample":0,
                           "matched_test_subsample":1,"learning_subsamples":0,"shap_sample":0},
                    source_row_convention="Zero-based physical Parquet row position; no reordering before splitting",
                    save_predictions=SAVE_PREDICTIONS,
                    statistical_unit="equally weighted selected solution rows",
                    interpretation="Retrospective classification conditional on the selected sample; not new-scan success or compute savings")
    dump("settings.json",provenance)
    print("Classification analysis finished",flush=True)


def main():
    global HERE, SAVE_PREDICTIONS
    args = parse_cli(__doc__)
    HERE = args.output
    SAVE_PREDICTIONS = args.save_predictions
    import pyarrow
    import pyarrow.parquet as pq
    pyarrow.set_cpu_count(2)
    pyarrow.set_io_thread_count(2)
    source = args.data
    columns=["m1","m2","m3","n1","n2","n3","FunctionValue"]
    frame=pq.ParquetFile(source).read(columns=columns).to_pandas()
    check_sample(frame, columns)
    raw=frame[columns[:6]].to_numpy(dtype=np.int32)
    potential=frame.FunctionValue.to_numpy(dtype=np.float64)
    m,n=raw[:,:3],raw[:,3:]
    inv=np.column_stack([m[:,a]*n[:,b]-m[:,b]*n[:,a] for a,b in ((0,1),(0,2),(1,2))]+
                        [m[:,a]*m[:,b]+n[:,a]*n[:,b] for a,b in ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))])
    names=["omega12","omega13","omega23","G11","G12","G13","G22","G23","G33"]
    y=(potential < THRESHOLD).astype(np.int8)
    indices=np.arange(len(y),dtype=np.int64)
    train,test=train_test_split(indices,test_size=.25,stratify=y,random_state=0)
    _,group_id=np.unique(inv,axis=0,return_inverse=True)
    group_id=group_id.reshape(-1)
    positive=np.bincount(group_id,weights=y)>0
    group_train_ids,group_test_ids=train_test_split(np.arange(len(positive)),test_size=.25,
                                                   stratify=positive,random_state=0)
    in_training=np.zeros(len(positive),dtype=bool);in_training[group_train_ids]=True
    group_train=indices[in_training[group_id]];group_test=indices[~in_training[group_id]]
    np.savez_compressed(HERE/"independent_group_splits.npz",group_id=group_id,train_groups=group_train_ids,
                        test_groups=group_test_ids,train=group_train,test=group_test)
    run_analysis(raw,inv,names,potential,train,test,group_train,group_test,source)


if __name__ == "__main__":
    main()
