#!/usr/bin/env python
# coding: utf-8
import os
os.environ['OMP_NUM_THREADS']   = '1'   # OpenMP (Intel MKL など)
os.environ['MKL_NUM_THREADS']   = '1'   # Intel MKL
os.environ['OPENBLAS_NUM_THREADS'] = '1' # OpenBLAS
os.environ['VECLIB_MAXIMUM_THREADS'] = '1' # Apple Accelerate
os.environ['NUMEXPR_MAX_THREADS']   = '1'

import csv, time
import re
import sys
from itertools import product
import multiprocessing as mp
import sympy as sp
from sympy.parsing.mathematica import parse_mathematica
from pathlib import Path  
import numpy as np
from functools import lru_cache
from utils.csv_helper import initialize_csv
from utils.variable_generator import generate_variables
from utils.equation_generator import generate_equations
from utils.taur_solver import find_tau_roots, substitute_and_evaluate
from utils.SUSYsolver import solve_flux, filter_flux_solutions
from utils.logger_setup import start_logging
from utils.import_wrappers import (
    read_h21, read_DZbW
)
from tqdm import tqdm
import logging

csv.field_size_limit(sys.maxsize)   # フィールドサイズ制限を増加

# 正規表現で X[数字] を X_数字 に置き換える関数
def replace_parentheses(expression):
    expression = re.sub(r'tau1', r'taur', expression)
    expression = re.sub(r'tau2', r'taui', expression)
    expression = re.sub(r'Z\[(\d+)\]', r'Z\1', expression)
    expression = re.sub(r'Zb\[(\d+)\]', r'Zb\1', expression)
    expression = re.sub(r'Zre\[(\d+)\]', r'Z\1r', expression)
    expression = re.sub(r'Zim\[(\d+)\]', r'Z\1i', expression)
    expression = re.sub(r'm\[(\d+)\]', r'm\1', expression)
    expression = re.sub(r'n\[(\d+)\]', r'n\1', expression)
    return expression  
    
# 行列DWを読み込む関数
@lru_cache(maxsize=None)
def read_DW_file(i):
    # ファイル名を動的に生成
    filename = f'./SUSY_data/DW_{i}.csv'
    
    KZZb = []
    
    try:
        # 指定されたファイルを読み込む
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Convert each element in the row to a sympy expression
                replaced_row = [replace_parentheses(expr) for expr in row]
                parsed_row = [parse_mathematica(expr) for expr in replaced_row]
                KZZb.append([item for elem in parsed_row for item in (elem if isinstance(elem, tuple) else [elem])])   
        KZZb = [list(elem) for sublist in KZZb for elem in sublist] #tupleを除去
        return KZZb
    
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None    
    
# 時間制限を秒単位で設定（デフォルト2時間）
time_limit = 2 * 60 * 60

def process_flux_solution(args):
    equationsAll, flux_params, all_vars, initial_guess, m_varsAll, n_vars = args 
    SUSY_solution = solve_flux(equationsAll, flux_params, all_vars, initial_guess)
    if SUSY_solution is not None:
        result_row = list(SUSY_solution) + \
                     [flux_params.get(var, None) for var in m_varsAll] + \
                     [flux_params.get(var, None) for var in n_vars]
        return result_row
    return None


# パラメータ設定
Z_values = np.arange(0.6, 1.1 + 0.1, 0.5)  # Zr, Zi用 
tau_values = np.arange(0.6, 1.1 + 0.1, 0.5)  # taui用
m1_vals = np.arange(0.01, 10.01, 0.05) #m1用


# i の範囲を設定
i_start = 2
i_end = 2

start_time = time.time()

if __name__ == '__main__':
    log_q, listener = start_logging() 
    logging.info('=== script started ===')
    for i in range(i_start, i_end + 1):
        logging.info(f'--- i={i} started ---')
        try:
            h21_data = read_h21(i)
            DW_data = read_DW_file(i)
            # 変数生成
            taur, taui, Z_vars, m_vars, m_varsAll, n_vars, all_vars = generate_variables(h21_data)
            # CSVファイル設定
            timestamp = time.strftime("%Y%m%d_%H%M")
            csv_filename = f'./output/i_{i}_SUSY_{timestamp}.csv'
            header = [str(var) for var in (taur, taui) + Z_vars] + [str(var) for var in m_varsAll] + [str(var) for var in n_vars]
            initialize_csv(csv_filename, header)
            # SUSY方程式生成
            equations, equationsAll, FluxMat = generate_equations(DW_data, m_vars, n_vars)

            total_combinations = len(tau_values) * (len(Z_values) ** (h21_data * 2)) 

            procs = mp.cpu_count()                                   # ★追加: 使用プロセス数を取得
            logging.info("    creating pool with %d processes", procs)
            with mp.get_context('spawn').Pool(processes=procs, initializer=start_logging, initargs=(log_q,)) as pool, \
                open(csv_filename, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                outer_loop = tqdm(product(Z_values, repeat=h21_data*2), desc="Outer loop")
                        
                for value_combination in outer_loop:
                    logging.info('--- (%s) started ---', ', '.join(f'{float(v):g}' for v in value_combination))
                    for tau_value in tau_values:
                        logging.info(f'taui={tau_value:.3g} started')
                        # 代入用の辞書を作成
                        Z_assignment = {
                            Z_vars[idx]: value_combination[idx] for idx in range(len(Z_vars))
                        }
                        assignment = {taui: tau_value, **Z_assignment}
                            
                        # taur の根を求める
                        t_root = time.perf_counter()
                        extended_roots = find_tau_roots(FluxMat, taur, assignment)
                        logging.info("    %d taur roots found in %.2fs",len(extended_roots), time.perf_counter() - t_root)

                        # 各 taur の解についてfluxの値を求める
                        for taur_val in extended_roots:
                        
                            # 実行時間のチェック
                            elapsed_time = time.time() - start_time
                            if elapsed_time > time_limit:
                                raise SystemExit(f"Execution stopped after {elapsed_time:.2f} seconds.")
                            
                            solutions_for_flux = substitute_and_evaluate(equations, taur_val, assignment, m_vars, n_vars)
                            flux_solutions = filter_flux_solutions(solutions_for_flux, m_vars, n_vars, m1_vals)
                            
                            initial_guess = [taur_val] + list(assignment.values())
                            all_vars = (taur, taui) + Z_vars

                            # 並列処理用タスク準備
                            task_args = [
                                (equationsAll, flux_params, all_vars, initial_guess, m_varsAll, n_vars)
                                for flux_params in flux_solutions
                            ]
                            logging.info("launching pool with %d flux candidates", len(task_args))
                            valid = 0
                            for row in tqdm(                             # ★ imap_unordered + chunksize
                                    pool.imap_unordered(process_flux_solution,task_args,chunksize=32),
                                    total=len(task_args),     
                                    desc=f"Solving flux eqs for taur={taur_val:.3g}, taui={tau_value:.3g}",
                                    leave=False):
                                if row is not None:                      # ★ 1 行ずつ即書込み
                                    writer.writerow(row)
                                    valid += 1
                            logging.info("solved %d / %d fluxes", valid, len(task_args))


        except SystemExit as e:
            # 経過秒数をCSVに保存
            elapsed_time = time.time() - start_time
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([])
                writer.writerow([f"Execution stopped after {elapsed_time:.2f} seconds."])
            raise
    
    listener.stop()
    logging.info('=== script finished ===')
    
    
    
# # 主ループ
# if __name__ == '__main__':
#     log_q, listener = start_logging() 
#     logging.info('=== script started ===')
#     for i in range(i_start, i_end + 1):
#         try:
#             h21_data = read_h21(i)
#             DW_data = read_DW_file(i)
#             # 変数生成
#             taur, taui, Z_vars, m_vars, m_varsAll, n_vars, all_vars = generate_variables(h21_data)
#             # CSVファイル設定
#             timestamp = time.strftime("%Y%m%d_%H%M")
#             csv_filename = f'./output/i_{i}_SUSY_{timestamp}.csv'
#             header = [str(var) for var in (taur, taui) + Z_vars] + [str(var) for var in m_varsAll] + [str(var) for var in n_vars]
#             initialize_csv(csv_filename, header)
#             # SUSY方程式生成
#             equations, equationsAll, FluxMat = generate_equations(DW_data, m_vars, n_vars)

#             total_combinations = len(tau_values) * (len(Z_values) ** (h21_data * 2)) 
#             outer_loop = tqdm(product(Z_values, repeat=h21_data * 2), desc="Outer loop")
#             for value_combination in outer_loop:
#                 logging.info('--- (%s) started ---', ', '.join(f'{float(v):g}' for v in value_combination))
#                 for tau_value in tau_values:

#                     # 代入用の辞書を作成
#                     Z_assignment = {
#                         Z_vars[idx]: value_combination[idx] for idx in range(len(Z_vars))
#                     }
#                     assignment = {taui: tau_value, **Z_assignment}
                    
#                     # taur の根を求める
#                     extended_roots = find_tau_roots(FluxMat, taur, assignment)

#                     # 各 taur の解についてfluxの値を求める
#                     for taur_val in extended_roots:
                        
#                         # 実行時間のチェック
#                         elapsed_time = time.time() - start_time
#                         if elapsed_time > time_limit:
#                             raise SystemExit(f"Execution stopped after {elapsed_time:.2f} seconds.")
                        
#                         solutions_for_flux = substitute_and_evaluate(equations, taur_val, assignment, m_vars, n_vars)
#                         flux_solutions = filter_flux_solutions(solutions_for_flux, m_vars, n_vars, m1_vals)
                        
#                         initial_guess = [taur_val] + list(assignment.values())
#                         all_vars = (taur, taui) + Z_vars

#                         # 並列処理用タスク準備
#                         task_args = [
#                             (equationsAll, flux_params, all_vars, initial_guess, m_varsAll, n_vars)
#                             for flux_params in flux_solutions
#                         ]
#                         with mp.get_context('spawn').Pool(initializer=start_logging, initargs=(log_q,)) as pool:
#                             tqdm_desc = f"Solving flux eqs for taur={taur_val:.3g}, taui={tau_value:.3g}"
#                             results = list(tqdm( 
#                                 pool.imap(process_flux_solution, task_args),
#                                 total=len(task_args),
#                                 desc=tqdm_desc,
#                                 leave=True
#                             ))
#                         logging.info(f'finished taur={taur_val:.3g} taui={tau_value:.3g} results={len(results)}')

#                         # 結果をCSVへ
#                         with open(csv_filename, 'a', newline='') as file:
#                             writer = csv.writer(file)
#                             for row in results:
#                                 if row is not None:
#                                     writer.writerow(row)

#         except SystemExit as e:
#             # 経過秒数をCSVに保存
#             elapsed_time = time.time() - start_time
#             with open(csv_filename, 'a', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow([])
#                 writer.writerow([f"Execution stopped after {elapsed_time:.2f} seconds."])
#             raise
    
#     listener.stop()
#     logging.info('=== script finished ===')    
