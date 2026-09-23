import numpy as np
import sympy as sp
from sympy import Eq, symbols, Matrix, lambdify

def generate_variables(h21_data):
    """必要な変数を動的に生成"""
    num_vars = h21_data
    Z_vars = [sp.symbols(f"Z{i}r Z{i}i") for i in range(1, num_vars + 1)]
    Z_vars = sum(Z_vars, ())  # Flatten the list of tuples
    taur, taui = sp.symbols('taur taui')
    k = h21_data + 1
    m_vars = sp.symbols([f"m{i}" for i in range(2, k + 1)])
    m_varsAll = sp.symbols([f"m{i}" for i in range(1, k + 1)])
    n_vars = sp.symbols([f"n{i}" for i in range(1, k + 1)])
    all_vars = (taur, taui) + tuple(Z_vars)
    return taur, taui, Z_vars, m_vars, m_varsAll, n_vars, all_vars