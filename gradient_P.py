#!/usr/bin/env python
# coding: utf-8
import os

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP (Intel MKL など)
os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Apple Accelerate
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import argparse
import csv
import joblib
import logging
import numpy as np
import re
import sympy as sp
import sys
import time

from contextlib import contextmanager
from functools import lru_cache
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from pathlib import Path
from sympy.parsing.mathematica import parse_mathematica
from tqdm import tqdm
from utils.csv_helper import initialize_csv
from utils.import_wrappers import (
    read_h21,
    read_EK,
    read_DtauW,
    read_DtaubW,
    read_KZZb,
    read_DZW,
    read_DZbW,
)
from utils.logger_setup import start_logging
from utils.optimization_worker import optimize_single_flux_factory
from utils.sampling import sample_m_n, ordered_sample_m_n
from utils.variable_generator import generate_variables

csv.field_size_limit(sys.maxsize)  # フィールドサイズ制限を増加

# 正規表現で X[数字] を X_数字 に置き換える関数
# def replace_parentheses(expression):
#     expression = re.sub(r'tau1', r'taur', expression)
#     expression = re.sub(r'tau2', r'taui', expression)
#     expression = re.sub(r'Z\[(\d+)\]', r'Z\1', expression)
#     expression = re.sub(r'Zb\[(\d+)\]', r'Zb\1', expression)
#     expression = re.sub(r'Zre\[(\d+)\]', r'Z\1r', expression)
#     expression = re.sub(r'Zim\[(\d+)\]', r'Z\1i', expression)
#     expression = re.sub(r'm\[(\d+)\]', r'm\1', expression)
#     expression = re.sub(r'n\[(\d+)\]', r'n\1', expression)
#     return expression

_RX = [
    (re.compile(r"tau1"), "taur"),
    (re.compile(r"tau2"), "taui"),
    (re.compile(r"Z\[(\d+)\]"), r"Z\1"),
    (re.compile(r"Zb\[(\d+)\]"), r"Zb\1"),
    (re.compile(r"Zre\[(\d+)\]"), r"Z\1r"),
    (re.compile(r"Zim\[(\d+)\]"), r"Z\1i"),
    (re.compile(r"m\[(\d+)\]"), r"m\1"),
    (re.compile(r"n\[(\d+)\]"), r"n\1"),
]

# PROCESS LIMIT of KEKCC batch queues: 20
# safety margin required for other processes
# parallel jobs are also limited with reserved CPU/Memory Cores (max 12)
# PROCESS_LIMIT = min(cpu_count(), 20)
# SAFETY_MARGIN = 10
# N_JOBS = max(1, PROCESS_LIMIT - SAFETY_MARGIN)
N_JOBS = 4
print(f"[INFO] No. of parallel process is limited to {N_JOBS}.")


def replace_parentheses(expr):
    for pat, rep in _RX:  # 事前コンパイル済みパターンを順に適用
        expr = pat.sub(rep, expr)
    return expr


@lru_cache(maxsize=None)  # 1 度読んだらメモ化
def read_KZZb_file(i):  # i は int → ハッシュ可能
    fn = Path(f"./Potential_data/KZZb_{i}.csv")  # ★Path オブジェクト
    if not fn.exists():
        logging.error("%s not found", fn)  # ファイルが無ければエラー
        return None

    KZZb = []
    with fn.open() as csvfile:  # Path の open() を使用
        for row in csv.reader(csvfile):
            replaced = [replace_parentheses(x) for x in row]
            parsed = [parse_mathematica(x) for x in replaced]
            KZZb.append(
                [
                    item
                    for elem in parsed
                    for item in (elem if isinstance(elem, tuple) else [elem])
                ]
            )
    KZZb = [list(elem) for sublist in KZZb for elem in sublist]  # tupleを除去
    return KZZb


# # 行列KZZbを読み込む関数
# def read_KZZb_file(i):
#     # ファイル名を動的に生成
#     filename = f'./Potential_data/KZZb_{i}.csv'

#     KZZb = []

#     try:
#         # 指定されたファイルを読み込む
#         with open(filename, newline='') as csvfile:
#             reader = csv.reader(csvfile)
#             for row in reader:
#                 # Convert each element in the row to a sympy expression
#                 replaced_row = [replace_parentheses(expr) for expr in row]
#                 parsed_row = [parse_mathematica(expr) for expr in replaced_row]
#                 KZZb.append([item for elem in parsed_row for item in (elem if isinstance(elem, tuple) else [elem])])
#         KZZb = [list(elem) for sublist in KZZb for elem in sublist] #tupleを除去
#         return KZZb

#     except FileNotFoundError:
#         print(f"File {filename} not found.")
#         return None


@contextmanager
def tqdm_joblib(tqdm_object):
    """tqdm context manager for joblib.Parallel"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# i の範囲を設定
i_start = 2
i_end = 2

# def is_duplicate(row, existing, atol=1e-4, rtol=0.0):
#     return any(np.allclose(row, ex, atol=atol, rtol=rtol) for ex in existing)

# Zeta(3)を数値化
Zeta = sp.Function("Zeta")
z3   = sp.zeta(3).evalf(50)
subs_zeta = {Zeta(3): z3}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="sample index", type=int, default=0)
    args = parser.parse_args()

    log_q, listener = start_logging()
    logging.info("=== script started ===")
    # i のループ
    for i in range(i_start, i_end + 1):
        logging.info(f"--- i={i} started ---")
        try:
            h21_data = read_h21(i)
            EK_data = read_EK(i)
            DtauW_data = read_DtauW(i)
            DtaubW_data = read_DtaubW(i)
            KZZb_data = read_KZZb_file(i)
            DZW_data = read_DZW(i)
            DZbW_data = read_DZbW(i)

            # 変数生成
            taur, taui, Z_vars, m_vars, m_varsAll, n_vars, all_vars = (
                generate_variables(h21_data)
            )
            
            # 数値化したZeta(3)を代入
            EK_data  = EK_data.subs(Zeta(3), z3)
            DtauW_data  = DtauW_data.subs(Zeta(3), z3)
            DtaubW_data = DtaubW_data.subs(Zeta(3), z3)
            DZW_data = [
                [e.subs(Zeta(3), z3).evalf() for e in t]
                for t in DZW_data
            ]
            DZbW_data = [
                [e.subs(Zeta(3), z3).evalf() for e in t]
                for t in DZbW_data
            ]
            KZZb_data = [
                [e.subs(Zeta(3), z3).evalf() for e in t]
                for t in KZZb_data
            ]

            # ポテンシャルVを計算
            K_tau_taub = 4 * taui**2
            Vtau = K_tau_taub * DtauW_data * DtaubW_data
            KZZb_matrix = sp.Matrix(KZZb_data)
            DZW_vector = sp.Matrix(DZW_data).T
            DZbW_vector = sp.Matrix(DZbW_data)
            Vcs_matrix = DZbW_vector * KZZb_matrix * DZW_vector
            Vcs = Vcs_matrix[
                0
            ]  # 結果が 1×1 行列なので、[0, 0] 要素を取得してスカラー値にする
            V = EK_data * (Vtau + Vcs)
            dV_exprs = [sp.diff(V, v) for v in all_vars]

            # CSVファイル設定
            timestamp = time.strftime("%Y%m%d_%H%M")
            csv_filename = f"./output/i_{i}_gradient_{args.index:06}_{timestamp}.csv"
            header = (
                ["FunctionValue"]
                + [str(var) for var in (taur, taui) + Z_vars]
                + [str(var) for var in m_varsAll]
                + [str(var) for var in n_vars]
            )
            initialize_csv(csv_filename, header)

            # サンプリング
            # flux_samples = sample_m_n(
            #     # max_sum_of_squares=1500, num_samples=200, k=h21_data + 1, flux_range=10
            #     max_sum_of_squares=276, num_samples=10, k=h21_data + 1, flux_range=17
            sample_size = 1500
            # sample_size = 10 # test with --index 1500000
            flux_samples = ordered_sample_m_n(
                max_sum_of_squares=276, num_samples=sample_size, sample_seed=args.index*sample_size, k=h21_data + 1, flux_range=17
            )
            logging.info(f"flux samples generated: {len(flux_samples)}")

            # 実行開始時刻を記録
            start_time = time.time()

            wrapped_opt = optimize_single_flux_factory(V, dV_exprs, all_vars, h21_data)
            with tqdm_joblib(
                tqdm(desc="Optimizing fluxes", total=len(flux_samples))
            ) as progress_bar:
                results = Parallel(n_jobs=N_JOBS, prefer="processes", batch_size=10)(
                    delayed(wrapped_opt)((m, n)) for m, n in flux_samples
                )

            # 重複除去 & 書き出し
            # dup_cache = set()
            # with open(csv_filename, "a", newline="") as outfile:
            #     writer = csv.writer(outfile)
            #     unique_cnt = 0
            #     for res in results:
            #         if res is not None:
            #             h = tuple(np.round(res, 4))  # 重複チェック用ハッシュ
            #             if h in dup_cache:
            #                 continue
            #             dup_cache.add(h)  # 重複チェック用ハッシュに追加
            #             writer.writerow(res)
            #             unique_cnt += 1
            # logging.info("finished i=%d  unique=%d", i, unique_cnt)
            
            # 複数解対応版
            dup_cache = set()
            with open(csv_filename, "a", newline="") as outfile:
                writer = csv.writer(outfile)
                unique_cnt = 0
                for res_list in results:
                    if res_list is None:
                        continue
                    # res_list は「その flux で得られた解のリスト」
                    for res in res_list:
                        # res は 1 解：[Vmin, taur, taui, Z..., m..., n...]
                        writer.writerow(res)
                        unique_cnt += 1

            logging.info("finished i=%d  unique=%d", i, unique_cnt)


        except Exception as e:
            # 経過秒数をCSVに保存
            elapsed_time = time.time() - start_time
            with open(csv_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([])
                writer.writerow(
                    [f"Execution stopped after {elapsed_time:.2f} seconds."]
                )
            logging.exception("abort i=%d : %s", i, e)  # 例外内容をログに残す
            continue
    listener.stop()

if __name__ == "__main__":
    main()
