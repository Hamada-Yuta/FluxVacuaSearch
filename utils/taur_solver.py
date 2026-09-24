import numpy as np
import sympy as sp
from scipy.optimize import root

def find_tau_roots(FluxMat, taur, assignment):
#def find_tau_roots(FluxMat, assignment):    
    """taur の解を求める"""
    numeric_fluxmat_func = sp.lambdify([taur], FluxMat.subs(assignment), 'numpy')
    
    def fluxmatrix(variable):
            tau1_val = float(variable) if np.isscalar(variable) else float(variable[0])
            return numeric_fluxmat_func(tau1_val).astype(np.float64)

    def fluxdet(x):
        return np.linalg.det(fluxmatrix(x))
    
    initial_guesses = np.linspace(0, 10, 10)
    roots = []
    for guess in initial_guesses:
        solution = root(fluxdet, guess)
        if solution.success:
            roots.append(solution.x[0])

    unique_roots = np.unique(np.round(roots, decimals=8))
    extended_roots = unique_roots.tolist()

    for ele in unique_roots:
        if -ele not in extended_roots:
            extended_roots.append(-ele)

    return extended_roots


def substitute_and_evaluate(equations, taur_val, assignment, m_vars, n_vars):
    """Fluxの値を決める方程式を解く"""
    substituted_equations = [eq.subs({**assignment, sp.Symbol('taur'): taur_val}) for eq in equations]
    return [(taur_val, sp.solve(substituted_equations, list(m_vars) + list(n_vars)))]