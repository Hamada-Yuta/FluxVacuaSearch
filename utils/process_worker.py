import sympy as sp
import numpy as np
from scipy.optimize import minimize
import time
import csv

def process_sample(args):
    m_values, n_values, V, all_vars, h21_data, csv_filename, start_time, time_limit = args

    successful_results = []

    # 実行時間のチェック
    elapsed_time = time.time() - start_time
    if elapsed_time > time_limit:
        #return None  # タイムアウト時
        raise TimeoutError(f"Execution stopped after {elapsed_time:.2f} seconds.")

    # フラックスパラメータの設定
    msubstitution = {f'm{j}': m_values[j - 1] for j in range(1, h21_data + 2)}
    nsubstitution = {f'n{j}': n_values[j - 1] for j in range(1, h21_data + 2)}
    substitutions = {**msubstitution, **nsubstitution}

    # フラックスパラメータをVに代入
    V_flux = V.subs(substitutions)
    V_func = sp.lambdify(all_vars, V_flux, 'numpy')

    # 最小化する関数
    def objective(params):
        taur, taui = params[0], params[1]
        Z_values = params[2:]
        return np.abs(V_func(taur, taui, *Z_values))

    # 勾配の関数
    grad_funcs = {var: sp.lambdify(all_vars, sp.diff(V_flux, var), 'numpy') for var in all_vars}

    def gradient(params):
        taur, taui = params[0], params[1]
        Z_values = params[2:]
        vars_values = (taur, taui) + tuple(Z_values)
        grads = [np.real(grad_funcs[var](*vars_values)) for var in all_vars]
        return np.array(grads)

    # 最適化実行 (10回試行)
    for _ in range(10):
        initial_guess = np.empty(2 * h21_data + 2)
        initial_guess[::2] = np.random.uniform(low=-2.0, high=2.0, size=h21_data + 1)  # 奇数番目の変数
        initial_guess[1::2] = np.random.uniform(low=0.1, high=2.0, size=h21_data + 1)  # 偶数番目の変数
        try:
            result = minimize(objective, initial_guess, method='SLSQP', jac=gradient)
        except Exception:
            continue

        # 結果の出力条件
        if result.success:
            grad_at_min = gradient(result.x)
            if np.all(np.abs(grad_at_min) <= 0.1):  # 勾配の条件をチェック
                func_value = objective(result.x)  # 最小値での関数の値
                result_row = [func_value] + list(result.x) + list(m_values) + list(n_values)
                successful_results.append(result_row)


    # 成功した結果をCSVに保存
    if successful_results:
        with open(csv_filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(successful_results)

    return len(successful_results)  # 成功したサンプル数を返す

#    return successful_results