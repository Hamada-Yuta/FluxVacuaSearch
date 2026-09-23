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
from itertools import islice
#from utils.csv_helper import initialize_csv
from utils.import_wrappers import (
    read_h21,
    read_EK,
    read_DtauW,
    read_DtaubW,
    #read_KZZb,
    read_DZW,
    read_DZbW,
)
from utils.logger_setup import start_logging
from utils.optimization_worker import optimize_single_flux_factory
#from utils.sampling import sample_m_n, ordered_sample_m_n
from utils.variable_generator import generate_variables

csv.field_size_limit(sys.maxsize)  # フィールドサイズ制限を増加

# for Parquet
import pandas as pd
#import pyarrow as pa
#import pyarrow.parquet as pq


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


# parallel jobs are also limited with reserved CPU/Memory Cores (max 18)
# PROCESS_LIMIT = min(cpu_count(), 20)
# SAFETY_MARGIN = 10
# N_JOBS = max(1, PROCESS_LIMIT - SAFETY_MARGIN)
N_JOBS = 8
print(f"[INFO] No. of parallel process is limited to {N_JOBS}.")


def replace_parentheses(expr):
    for pat, rep in _RX:  # 事前コンパイル済みパターンを順に適用
        expr = pat.sub(rep, expr)
    return expr

def valid_m1n1_pairs(max_sumsq=276):
    r01 = range(0, 17)  # 0..16
    pairs = []
    for m1 in r01:
        m1s = m1*m1
        for n1 in r01:
            s1 = m1s + n1*n1
            if s1 <= max_sumsq:
                pairs.append((m1, n1, s1))
    # 固定順（m1昇順, n1昇順）で確定
    return pairs


def flux_iter_for_fixed_m1n1(m1, n1, s1, max_sumsq=276):
    r = range(-16, 17)  # -16..16
    rem = max_sumsq - s1
    if rem < 0:
        return  # 何も出さない

    for m2 in r:
        s2 = m2*m2
        if s2 > rem:
            continue
        for n2 in r:
            s3 = s2 + n2*n2
            if s3 > rem:
                continue
            for m3 in r:
                s4 = s3 + m3*m3
                if s4 > rem:
                    continue
                for n3 in r:
                    if s4 + n3*n3 > rem:
                        continue
                    m = (m1, m2, m3)
                    n = (n1, n2, n3)
                    yield (m, n)



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


def flush_chunk(rows, columns, float_cols, int_cols, out_path):
    df = pd.DataFrame(rows, columns=columns)
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in int_cols:
        # m,n は range 次第で int16 で足りないなら int32 に
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int16")

    df.to_parquet(out_path, index=False)

# i の範囲を設定
i_start = 2
i_end = 2

# Zeta(3)を数値化
Zeta = sp.Function("Zeta")
z3   = sp.zeta(3).evalf(50)
subs_zeta = {Zeta(3): z3}

def main():
    timestamp = time.strftime("%Y%m%d_%H%M")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="sample index", type=int, default=0)
    parser.add_argument("--chunk-id", type=int, default=0,
                    help="which chunk of the (m2,n2,m3,n3) grid to process")
    parser.add_argument("--chunk-size", type=int, default=300000,
                    help="how many flux points to process in this job (within one index)")
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
            EK_data  = EK_data.subs(subs_zeta)
            DtauW_data  = DtauW_data.subs(subs_zeta)
            DtaubW_data = DtaubW_data.subs(subs_zeta)
            DZW_data = [
                [e.subs(subs_zeta).evalf() for e in t]
                for t in DZW_data
            ]
            DZbW_data = [
                [e.subs(subs_zeta).evalf() for e in t]
                for t in DZbW_data
            ]
            KZZb_data = [
                [e.subs(subs_zeta).evalf() for e in t]
                for t in KZZb_data
            ]
            if KZZb_data is None:
                logging.error("KZZb missing for i=%d", i)
                continue

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
            
            # Parquetファイル設定
            # 出力ディレクトリ設定
            #timestamp = time.strftime("%Y%m%d_%H%M")
            #out_dir = Path(f"./output/parquet/i={i}/index={args.index:06d}/ts={timestamp}")
            out_dir = Path(f"./output/parquet/i={i}/index={args.index:06d}/chunk={args.chunk_id:04d}/ts={timestamp}") # 変更点
            out_dir.mkdir(parents=True, exist_ok=True)
            # ヘッダー
            columns = (
                ["FunctionValue"]
                + [str(var) for var in (taur, taui) + Z_vars]
                + [str(var) for var in m_varsAll]
                + [str(var) for var in n_vars]
            )
            float_cols = ["FunctionValue"] + [str(var) for var in (taur, taui) + Z_vars]
            int_cols   = [str(var) for var in m_varsAll] + [str(var) for var in n_vars]
            # Parquet chunk 書き出し用
            chunk_rows = []
            chunk_id = 0
            CHUNK_SIZE = 50_000
            ###


            # --- (m1,n1) を index で割り当て ---
            pairs = valid_m1n1_pairs(max_sumsq=276)
            if args.index < 0 or args.index >= len(pairs):
                logging.error("index out of range: %d (valid: 0..%d)", args.index, len(pairs)-1)
                continue

            m1, n1, s1 = pairs[args.index]
            logging.info("assigned (m1,n1)=(%d,%d), s1=%d, index=%d/%d",
                        m1, n1, s1, args.index, len(pairs)-1)

            grid = flux_iter_for_fixed_m1n1(m1, n1, s1, max_sumsq=276)

            # ---- chunk slicing (NEW) ----
            chunk_start = args.chunk_id * args.chunk_size
            chunk_end   = (args.chunk_id + 1) * args.chunk_size

            # 先に chunk_start 個スキップし、chunk_size 個だけ処理する
            grid = islice(grid, chunk_start, chunk_end)

            logging.info("chunk slice: chunk_id=%d chunk_size=%d range=[%d,%d)",
                        args.chunk_id, args.chunk_size, chunk_start, chunk_end)
            # -----------------------------

            BLOCK = 2500   # 1回にjoblibへ投げる flux 数

            # Parquetファイル設定
            unique_cnt = 0
            wrapped_opt = optimize_single_flux_factory(V, dV_exprs, all_vars, h21_data)
            while True:
                flux_samples = list(islice(grid, BLOCK))
                if not flux_samples:
                    break

                logging.info("flux block size=%d", len(flux_samples))

                with tqdm_joblib(tqdm(desc="Optimizing fluxes", total=len(flux_samples))) as _:
                    results = Parallel(n_jobs=N_JOBS, prefer="processes", batch_size=10)(
                        delayed(wrapped_opt)((m, n)) for m, n in flux_samples
                    )
                # 複数解バージョン    
                for res_list in results:
                    if res_list is None:
                       continue
                    for res in res_list:
                        chunk_rows.append(res)
                        unique_cnt += 1

                        if len(chunk_rows) >= CHUNK_SIZE:
                            part_path = out_dir / f"part-{chunk_id:05d}.parquet"
                            flush_chunk(chunk_rows, columns, float_cols, int_cols, part_path)
                            chunk_rows.clear()
                            chunk_id += 1
                    
                    # 単一解バージョン
                # for res in results:
                #     if res is None:
                #         continue
                #     chunk_rows.append(res)
                #     unique_cnt += 1
                #     if len(chunk_rows) >= CHUNK_SIZE:
                #         part_path = out_dir / f"part-{chunk_id:05d}.parquet"
                #         flush_chunk(chunk_rows, columns, float_cols, int_cols, part_path)
                #         chunk_rows.clear()
                #         chunk_id += 1
            # 余りを書き出し
            if chunk_rows:
                part_path = out_dir / f"part-{chunk_id:05d}.parquet"
                flush_chunk(chunk_rows, columns, float_cols, int_cols, part_path)
                chunk_rows.clear()
            
            logging.info("finished i=%d  rows=%d  out=%s", i, unique_cnt, str(out_dir))
        except Exception as e:
            logging.exception("abort i=%d : %s", i, e)
            continue

    listener.stop()

if __name__ == "__main__":
    main()
