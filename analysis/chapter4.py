"""Reproduce the Chapter 4 statistics and figures from the selected dataset.

Run from the repository root::

    python -m analysis.chapter4 --data data/selected_solutions.parquet --output results/chapter4

The input already contains the final representatives. This script makes no
new Hessian selection, duplicate removal, minimization, coordinate mapping or
averaging by flux. Only the explicitly named plotting/correlation subsets are
formed. The common numerical SUSY threshold is |V| < 10**(-4.5).

Flux multiplicity, log-V and per-flux label-fraction plots are reproduced by
analysis.flux_statistics. This module requires no previous results, local
manuscript files, external symbolic inputs or archived raw scan data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
from scipy.stats import kendalltau, rankdata, spearmanr

from . import _chapter4_figures as figures
from ._instanton import coefficient_records, correction_ratio

COLS = ['FunctionValue', 'taur', 'taui', 'Z1r', 'Z1i', 'Z2r', 'Z2i',
        'm1', 'm2', 'm3', 'n1', 'n2', 'n3']
THRESHOLD = 10**(-4.5)
KEY_PAIRS = [('Z1r', 'Z2r'), ('Z1i', 'Z2i'), ('taui', 'Z1i'),
             ('taui', 'Z2i'), ('Z1r', 'Z1i'), ('Z2r', 'Z2i'),
             ('FunctionValue', 'Z1i'), ('FunctionValue', 'Z2i')]


def read_data(path):
    # ParquetFile also supports parent directories with '=' in their names.
    data = pq.ParquetFile(path).read().to_pandas()
    if list(data.columns) != COLS:
        raise ValueError(f'Expected columns, in order: {COLS}')
    if len(data) < 2 or not np.isfinite(data.to_numpy()).all():
        raise ValueError('Expected a finite selected dataset with at least two rows')
    if not ((data.Z1i >= 1) & (data.Z2i >= 1) & (data.taui > 0)
            & (data.FunctionValue > 0) & (data.FunctionValue < 1e10)).all():
        raise ValueError('Input does not meet the documented final-sample coordinate/potential bounds')
    flux = data[COLS[7:]].to_numpy()
    if not np.equal(flux, np.round(flux)).all():
        raise ValueError('Flux columns must contain integers')
    return data.sort_values('FunctionValue')


def correlations(frame, threads):
    """All rows, average ties for Spearman and tau-b ties for Kendall."""
    values = np.asfortranarray(frame[COLS].to_numpy(dtype=float))
    pearson = frame[COLS].corr().to_numpy()
    ranks = np.empty_like(values)
    for j in range(len(COLS)):
        ranks[:, j] = rankdata(values[:, j], method='average')
    spearman = np.corrcoef(ranks, rowvar=False)
    del ranks
    kendall = np.eye(len(COLS))
    pairs = [(i, j) for i in range(len(COLS)) for j in range(i+1, len(COLS))]

    def pair(pair_indices):
        i, j = pair_indices
        result = kendalltau(values[:, i], values[:, j], variant='b',
                            method='asymptotic', nan_policy='raise')
        return i, j, float(result.statistic)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i, j, value in pool.map(pair, pairs):
            kendall[i, j] = kendall[j, i] = value
    methods = {'pearson': pearson, 'spearman': spearman, 'kendall_b': kendall}
    for matrix in methods.values():
        if not np.isfinite(matrix).all():
            raise ValueError('A correlation is undefined; check for constant input columns')
        assert np.allclose(matrix, matrix.T, atol=1e-14, rtol=0)
        assert np.allclose(np.diag(matrix), 1, atol=1e-14, rtol=0)
        assert np.max(abs(matrix)) <= 1+1e-12
    independent = []
    for a, b in KEY_PAIRS:
        i, j = COLS.index(a), COLS.index(b)
        reference = float(spearmanr(values[:, i], values[:, j]).statistic)
        assert abs(reference-spearman[i, j]) < 1e-12
        independent.append({'x': a, 'y': b, 'spearman_difference': reference-spearman[i,j]})
    return methods, {
        'all_rows_used': len(frame), 'symmetric_unit_diagonal': True,
        'numpy_pearson_max_difference': float(np.max(abs(np.corrcoef(values, rowvar=False)-pearson))),
        'independent_spearman_checks': independent,
    }


def pair_coefficients(frame, x, y):
    return {'pearson': float(frame[x].corr(frame[y])),
            'spearman': float(spearmanr(frame[x], frame[y]).statistic),
            'kendall_b': float(kendalltau(frame[x], frame[y], variant='b', method='asymptotic').statistic)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=Path('data/selected_solutions.parquet'))
    parser.add_argument('--output', type=Path, default=Path('results/chapter4'))
    parser.add_argument('--threads', type=int, default=2, help='Concurrent Kendall pairs (default: 2)')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data = read_data(args.data)
    susy = data.loc[data.FunctionValue.abs() < THRESHOLD]
    bounded = data.loc[(data.Z1i <= 50) & (data.Z2i <= 50)]
    susy_bounded = susy.loc[(susy.Z1i <= 50) & (susy.Z2i <= 50)]
    if min(len(susy), len(bounded), len(susy_bounded)) < 2:
        raise ValueError('The manuscript subsets must each contain at least two rows')
    summary = {
        'input': str(args.data), 'columns': COLS,
        'counts': {'all': len(data), 'SUSY': len(susy), 'nonSUSY': len(data)-len(susy),
                   'bounded': len(bounded), 'SUSY_bounded': len(susy_bounded),
                   'SUSY_percent': 100*len(susy)/len(data)},
        'classification': '|V| < 10**(-4.5)', 'threshold': THRESHOLD,
        'coordinate_convention': 'Original saved coordinates before fundamental-domain mapping',
        'weighting': 'One equal weight per retained solution row, including multiple rows at the same flux',
        'additional_physical_selection': False, 'additional_duplicate_removal': False,
        'statistics': {}, 'headlines': {}, 'correlations': {}, 'validation': {},
        'tail_cut_sensitivity': [], 'density_windows': {}, 'figures': {},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'pyarrow': pyarrow.__version__, 'matplotlib': figures.matplotlib.__version__},
    }
    for group, frame in [('all', data), ('SUSY', susy)]:
        desc = frame[COLS].describe()
        desc.to_csv(args.output/f'{group}_statistics.csv', float_format='%.17g')
        summary['statistics'][group] = desc.to_dict()
        independent_quantiles = np.quantile(frame[COLS].to_numpy(), [0, .25, .5, .75, 1], axis=0)
        assert np.allclose(independent_quantiles, desc.loc[['min', '25%', '50%', '75%', 'max']].to_numpy(), rtol=1e-14, atol=1e-14)
        for name, columns in [('moduli', COLS[:7]), ('flux', COLS[7:])]:
            desc[columns].loc[['min', '25%', '50%', '75%', 'max']].to_csv(
                args.output/f'table_{group}_{name}.csv', float_format='%.17g')
    matrix_arrays = {}
    for group, frame in [('all', data), ('bounded', bounded), ('SUSY', susy)]:
        print(f'Computing correlations: {group}, {len(frame):,} rows', flush=True)
        matrices, validation = correlations(frame, args.threads)
        matrix_arrays[group] = matrices
        summary['validation'][group] = validation
        summary['correlations'][group] = {method: matrix.tolist() for method, matrix in matrices.items()}
        for method, matrix in matrices.items():
            pd.DataFrame(matrix, index=COLS, columns=COLS).to_csv(
                args.output/f'{group}_{method}.csv', float_format='%.17g')
        summary['headlines'][group] = {f'{a}_{b}': {
            method: float(matrix[COLS.index(a), COLS.index(b)]) for method, matrix in matrices.items()}
            for a, b in KEY_PAIRS}
    for group, full, subset in [('all', data, bounded), ('SUSY', susy, susy_bounded)]:
        for x, y in [('Z1r', 'Z2r'), ('Z1i', 'Z2i')]:
            summary['tail_cut_sensitivity'].append({
                'group': group, 'x': x, 'y': y, 'cap': 50, 'rows': len(subset),
                'excluded': len(full)-len(subset), 'excluded_percent': 100*(len(full)-len(subset))/len(full),
                **pair_coefficients(subset, x, y)})
    pd.DataFrame(summary['tail_cut_sensitivity']).to_csv(args.output/'tail_cut_sensitivity.csv', index=False)
    summary['linear_histograms'] = figures.potential_histograms(data, args.output)
    for stem, group, title, colorbar in [
        ('correlation_matrix_reliable', 'all', 'Correlation matrix (reliable solutions)', False),
        ('correlation_matrix_Z1i_Z2i_leq50', 'bounded', r'Correlation matrix for $1 \leq \mathrm{Im}\,Z_1,\,\mathrm{Im}\,Z_2 \leq 50$', True),
        ('correlation_matrix_SUSY', 'SUSY', 'Correlation matrix (SUSY solutions)', True),
    ]:
        figures.pearson(matrix_arrays[group]['pearson'], args.output, stem, title, colorbar)
    for method, stem, label in [('spearman', 'rank_correlation_spearman', r'Spearman $\rho$'),
                                ('kendall_b', 'rank_correlation_kendall', r'Kendall $\tau_b$')]:
        summary['figures'][stem] = figures.rank_pair(args.output, stem, [
            (matrix_arrays['all'][method], '(a) All reliable solutions'),
            (matrix_arrays['bounded'][method], r'(b) $\mathrm{Im}\,Z_1,\,\mathrm{Im}\,Z_2\leq 50$')], label)
    summary['figures']['rank_correlation_SUSY'] = figures.rank_pair(args.output, 'rank_correlation_SUSY', [
        (matrix_arrays['SUSY']['spearman'], '(a) Spearman'),
        (matrix_arrays['SUSY']['kendall_b'], r'(b) Kendall $\tau_b$')], 'Correlation')
    windows = [('Z1r', 'Z1i', (-10,10), (0,50)), ('Z2r', 'Z2i', (-10,10), (0,50)),
               ('Z1r', 'Z2r', (-10,10), (-10,10)), ('Z1i', 'Z2i', (0,50), (0,50))]
    labelmap = dict(zip(COLS, figures.LABELS))
    for prefix, frame, active in [('', data, windows), ('SUSY_', susy, windows[2:])]:
        for x, y, xlim, ylim in active:
            stem = f'{prefix}{y}_{x}'
            summary['density_windows'][stem] = figures.density(
                frame, args.output, stem, x, y, xlim, ylim, labelmap, bool(prefix))
    pd.DataFrame.from_dict(summary['density_windows'], orient='index').rename_axis('figure').to_csv(
        args.output/'density_windows.csv')
    print('Computing instanton correction directly from all saved coordinates', flush=True)
    ratios = correction_ratio(data)
    summary['instanton'] = figures.instanton_histogram(ratios, args.output)
    pd.DataFrame(coefficient_records()).to_csv(args.output/'instanton_coefficients.csv', index=False)
    summary['generated_pdfs'] = sorted(p.name for p in args.output.glob('*.pdf'))
    summary['elapsed_seconds'] = time.monotonic()-start
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'counts': summary['counts'], 'instanton': summary['instanton'],
                      'elapsed_seconds': summary['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
