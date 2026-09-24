# optimize_single_flux.py

import numpy as np
import sympy as sp
from scipy.optimize import minimize
import logging, os
from functools import lru_cache, partial
#logging.info(f"worker {os.getpid()} alive")


def optimize_single_flux(args):
    """
    単一の(m,n)フラックスサンプルに対するポテンシャル最小化を実行する。

    Parameters
    ----------
    args : tuple
        (m_values, n_values, V_expr, all_vars, h21)

    Returns
    -------
    list or None
        [Vmin, taur, taui, Z1r, Z1i, ..., m1, ..., n1, ...] 形式のリスト。
        条件を満たす最適化が成功すれば返し、そうでなければ None。
    """
    m_values, n_values, V_expr, all_vars, h21 = args

    log = logging.getLogger(__name__)          # ← 何度呼んでも軽量
    log.debug(f'[PID {os.getpid()}] m={m_values} n={n_values} start')

    # m,n の代入辞書
    substitutions = {f'm{j}': m_values[j - 1] for j in range(1, h21 + 2)}
    substitutions.update({f'n{j}': n_values[j - 1] for j in range(1, h21 + 2)})

    # ポテンシャルとその微分に代入
    V_subs = V_expr.subs(substitutions)


    # lambdify
    V_func = sp.lambdify(all_vars, V_subs, 'numpy')
    grad_funcs = {
        var: sp.lambdify(all_vars, sp.diff(V_subs, var), 'numpy')
        for var in all_vars
    }

    def objective(params):
        taur, taui = params[0], params[1]
        Z_values = params[2:]
        return np.abs(V_func(taur, taui, *Z_values))

    def gradient(params):
        taur, taui = params[0], params[1]
        Z_values = params[2:]
        vals = (taur, taui) + tuple(Z_values)
        return np.array([np.real(grad_funcs[var](*vals)) for var in all_vars])

    # taui, Z_i^imag > 0 制約
    constraints = [{'type': 'ineq', 'fun': lambda x, idx=i: x[idx] - 1e-5} for i in range(1, len(all_vars), 2)]

    for _ in range(10):
        x0 = np.empty(2 * h21 + 2)
        x0[::2] = np.random.uniform(-2.0, 2.0, h21 + 1)   # Re parts
        x0[1::2] = np.random.uniform(0.1, 2.0, h21 + 1)   # Im parts

        result = minimize(objective, x0, method='SLSQP', jac=gradient, constraints=constraints)

        if result.success:
            grad_val = gradient(result.x)
            if np.all(np.abs(grad_val) <= 0.1):
                Vmin = objective(result.x)
                log.info(f'[PID {os.getpid()}] success V={Vmin:.3e}'
                     f' m={m_values} n={n_values}')
                return [Vmin] + list(result.x) + list(m_values) + list(n_values)
    log.debug(f'[PID {os.getpid()}] m={m_values} n={n_values} failed')
    return None