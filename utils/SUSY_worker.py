import csv, time, itertools as it, multiprocessing as mp
from utils.taur_solver import find_tau_roots, substitute_and_evaluate
from utils.SUSYsolver import solve_flux, filter_flux_solutions
from utils.equation_generator import generate_equations

# ---------------- ワーカー ----------------
def worker(task):
    value_combination, tau_value = task
    rows = []

    # 1) 代入辞書
    Z_assignment = {Z_vars[i]: value_combination[i] for i in range(len(Z_vars))}
    assignment   = {taui: tau_value, **Z_assignment}
    equations, equationsAll, FluxMat = generate_equations(DW_data, m_vars, n_vars)

    # 2) taur の候補
    for taur_val in find_tau_roots(FluxMat, taur, assignment):

        # 3) flux を決定
        sols = substitute_and_evaluate(equations, taur_val,
                                       assignment, m_vars, n_vars)
        flux_sols = filter_flux_solutions(sols, m_vars, n_vars, m1_vals)

        init_guess = [taur_val] + list(assignment.values())

        # 4) F-flat 条件を解く
        for flux_params in flux_sols:
            sol = solve_flux(equationsAll, flux_params,
                             (taur_val, tau_value)+Z_vars, init_guess)
            if sol is None:
                continue

            row = (list(sol) +
                   [flux_params.get(v) for v in m_varsAll] +
                   [flux_params.get(v) for v in n_vars])
            rows.append(row)

    return rows

