# optimize_single_flux.py

import numpy as np, sympy as sp, logging, os
from scipy.optimize import minimize
from functools import lru_cache
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


EPS  = 1e-12
HUGE = 1e200

# ---------- λcache はモジュールスコープ ----------
@lru_cache(maxsize=128)
def _get_funcs_cached(v_repr: str,
                      d_repr_tuple: Tuple[str, ...],
                      all_vars_repr: Tuple[str, ...]):
    """repr 文字列をキーに lambdify 結果をキャッシュ"""
    vars_sym = tuple(map(sp.symbols, all_vars_repr))
    V_expr   = sp.sympify(v_repr)
    d_exprs  = [sp.sympify(r) for r in d_repr_tuple]

    V_func   = sp.lambdify(vars_sym, V_expr, 'numpy')
    grad_map = {str(v): sp.lambdify(vars_sym, d, 'numpy')
                for v, d in zip(vars_sym, d_exprs)}
    return V_func, grad_map
# -----------------------------------------------

# 有効数字 n 桁で丸める関数
import numpy as np

def round_sig_np(x, n):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    # 0 / 非有限値のマスク
    mask = (x != 0) & np.isfinite(x)
    out[~mask] = x[~mask]  # 0, inf, nan はそのまま

    if np.any(mask):
        xm = x[mask]
        # 桁 d = floor(log10(|x|))
        mags = np.floor(np.log10(np.abs(xm)))
        # シフト量 10^(n-1-d)
        factor = np.power(10.0, n - 1 - mags)
        out[mask] = np.round(xm * factor) / factor

    return out




# 複数解を返すバージョン
def optimize_single_flux_factory(V_tmpl, dV_exprs, all_vars, h21, ndigits=2):
    def wrapped(args):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        m_vals, n_vals = args
        log = logging.getLogger(__name__)

        # 1) 係数代入
        subs = {f'm{j}': m_vals[j-1] for j in range(1, h21+2)}
        subs |= {f'n{j}': n_vals[j-1] for j in range(1, h21+2)}
    
        V_sub  = V_tmpl.subs(subs)
        dV_sub = [d.subs(subs) for d in dV_exprs]
    
        # 2) キャッシュキーは repr でハッシュ可能に
        key_V  = sp.srepr(V_sub)
        key_dV = tuple(map(sp.srepr, dV_sub))
        key_vs = tuple(map(str, all_vars))
    
        V_func, grad_map = _get_funcs_cached(key_V, key_dV, key_vs)

        def objective(params):
            try:
                params = np.asarray(params, dtype=np.float64)
                taur, taui = params[:2]
                Z_vals     = params[2:]
                v = V_func(taur, taui, *Z_vals)
                v = np.asarray(v).reshape(-1)[0]
                a = float(np.abs(v))
                return a if np.isfinite(a) else HUGE
            except Exception:
                return HUGE            
            # taur, taui = params[:2]
            # Z_vals     = params[2:]
            # with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            #     return np.abs(V_func(taur, taui, *Z_vals))

        # def gradient(params):
        #     taur, taui = params[:2]
        #     Z_vals     = params[2:]
        #     vals = (taur, taui) + tuple(Z_vals)
        #     with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        #         return np.array([np.real(grad_map[str(v)](*vals)) for v in all_vars])

        def gradient(params):
            try:
                params = np.asarray(params, dtype=np.float64)
                taur, taui = params[:2]
                Z_vals     = params[2:]
                vals = (taur, taui) + tuple(Z_vals.tolist())
                g = np.array([np.real(grad_map[str(v)](*vals)) for v in all_vars], dtype=np.float64)
                return g if np.all(np.isfinite(g)) else np.zeros(len(all_vars), dtype=np.float64)
            except Exception:
                return np.zeros(len(all_vars), dtype=np.float64)

        def _ineq(x, idx):
            try:
                x = np.asarray(x, dtype=np.float64)
                v = float(x[idx] - 1e-5)
                return v if np.isfinite(v) else -1.0
            except Exception:
                return -1.0

        # tau_i, Z_imag > 0 制約
        #constraints = [{'type': 'ineq', 'fun': lambda x, idx=k: x[idx]-1e-5}
        #               for k in range(1, len(all_vars), 2)]
        constraints = [{'type':'ineq', 'fun': (lambda x, idx=k: _ineq(x, idx))}
               for k in range(1, len(all_vars), 2)]

        # try:
        #     res = minimize(objective, x0, jac=gradient, method='SLSQP',
        #                 constraints=constraints, tol=1e-5)
        # except Exception as e:
        #     log.warning(f"SLSQP failed (trial={trial}) m={m_vals} n={n_vals}: {e}")
        #     return None
        
        # 全部の解を保持する入れ物
        solutions = []      # full precision の解を格納
        seen_keys = set()   # 丸めたキーで重複チェック

        for _ in range(10):
        #for _ in range(3): # 試行回数を3回に削減
            x0 = np.empty(2*h21+2, dtype=np.float64)
            x0[::2]  = np.random.uniform(-2, 2,  h21+1)
            x0[1::2] = np.random.uniform(0.1, 2, h21+1)

            try:
                res = minimize(objective, x0, jac=gradient, method='SLSQP',
                               constraints=constraints, tol=1e-5)
            except Exception as e:
                # debug だと見えないことが多いので warning 推奨
                log.warning(f"SLSQP failed (trial={_}) m={m_vals} n={n_vals}: {e}")
                continue

            if res.success and np.all(np.abs(gradient(res.x)) <= 0.1):
                Vmin = objective(res.x)
                # ★ここで即 return（1解見つけたら終わり,単一解version）
                # sol = [Vmin] + list(res.x) + list(m_vals) + list(n_vals)
                # return sol

                # 以下は複数解バージョンのコード
                # 有効数字（ndigits桁）で丸めたキーを作る
                vec = np.concatenate(([Vmin], res.x))
                rounded_vec = round_sig_np(vec, ndigits)
                key = tuple(rounded_vec.tolist())

                # まだ見ていない解だけ追加
                if key not in seen_keys:
                    seen_keys.add(key)
                    # 元のフォーマットに倣ってフル精度で保存
                    sol = [Vmin] + list(res.x) + list(m_vals) + list(n_vals)
                    solutions.append(sol)

        # 試行回数すべて終わってから、重複なし全解を返す
        if solutions:
            # ここで solutions は「解のリスト」
            # 各要素は [Vmin, taur, taui, Z..., m..., n...] という1解
            return solutions

        log.debug(f'[PID {os.getpid()}] m={m_vals} n={n_vals} failed')
        return None

    return wrapped