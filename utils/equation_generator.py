import numpy as np
import sympy as sp
from sympy import Eq, symbols, Matrix, lambdify

def generate_equations(DW_data, m_vars, n_vars):
    """ベクトル方程式と行列の生成"""
    m = sp.symbols(f"m1:{len(m_vars) + 2}")
    n = sp.symbols(f"n1:{len(n_vars) + 1}")
    vector = sp.Matrix(list(m) + list(n))
    result = sp.Matrix(DW_data) * vector
    equations = [sp.Eq(comp, 0) for comp in result[:-1]]
    equationsAll = [sp.Eq(comp, 0) for comp in result]
    return equations, equationsAll, sp.Matrix(DW_data)