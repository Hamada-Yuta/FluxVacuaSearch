import numpy as np
import sympy as sp
from scipy.optimize import root

def solve_flux(equations, flux_params, all_vars, initial_guess):
    """方程式系を解いてSUSY解を探す"""
    numeric_equations = [sp.lambdify(all_vars, eq.lhs.subs(flux_params), 'numpy') for eq in equations]
    #print(1)
    def equations_system(vars):
        return np.array([f(*vars) for f in numeric_equations])

    solution = root(equations_system, initial_guess, method='hybr', tol=1e-5)
    #print(solution)
    
    if solution.success:
        result = solution.x
        # 条件を満たす解のみ返す
        #if all(result[i] >= 0.01 for i in range(len(result)) if (i + 1) % 2 == 0):
        if all(result[i] > 0 for i in range(len(result)) if (i + 1) % 2 == 0):
            return result
    return None


def filter_flux_solutions(solutions, m_vars, n_vars, m1_vals):
    """Flux解をフィルタリング"""
    filtered = []
    for value, eqs in solutions:
        flux_solutions = []
        for m1val in m1_vals:
            substituted_eqs = {sp.Symbol('m1'): round(m1val)}
            try:
                substituted_eqs.update({
                    key: round(expr.subs(sp.Symbol('m1'), m1val).evalf())
                    if expr.subs(sp.Symbol('m1'), m1val).is_number else None
                    for key, expr in eqs.items()
                })
                if None not in substituted_eqs.values():
                    flux_solutions.append(substituted_eqs)
            except:
                continue
        # 条件フィルタリング
        flux_solutions = [dict(t) for t in {tuple(d.items()) for d in flux_solutions}]# 重複を削除
        flux_solutions = [
            item for item in flux_solutions
            if any(value != 0 for value in item.values()) #and
            #item[sp.Symbol('m1')]**2 + sum(item[m]**2 for m in m_vars) + sum(item[n]**2 for n in n_vars) <= 1000
        ]
        filtered.extend(flux_solutions)
    return filtered