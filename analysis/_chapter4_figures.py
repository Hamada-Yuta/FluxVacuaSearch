"""Matplotlib layouts used by Chapter 4; all data are passed by the caller."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

LABELS = [r'$V$', r'$\mathrm{Re}\,\tau$', r'$\mathrm{Im}\,\tau$',
          r'$\mathrm{Re}\,Z_1$', r'$\mathrm{Im}\,Z_1$',
          r'$\mathrm{Re}\,Z_2$', r'$\mathrm{Im}\,Z_2$',
          r'$m_1$', r'$m_2$', r'$m_3$', r'$n_1$', r'$n_2$', r'$n_3$']


def save(fig, output, stem, tight=True):
    kwargs = {'bbox_inches': 'tight'} if tight else {}
    fig.savefig(output/f'{stem}.pdf', metadata={'CreationDate': None, 'ModDate': None}, **kwargs)
    fig.savefig(output/f'{stem}.png', dpi=160, **kwargs)
    plt.close(fig)


def pearson(matrix, output, stem, title, colorbar=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    if colorbar:
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('correlation', rotation=270, labelpad=18, fontsize=14)
        cb.ax.tick_params(labelsize=14)
    ax.set_xticks(range(13), LABELS, rotation=45, ha='right', fontsize=14)
    ax.set_yticks(range(13), LABELS, fontsize=14)
    ax.set_title(title, fontsize=20)
    for i in range(13):
        for j in range(13):
            value = matrix[i, j]
            value = 0.0 if abs(value) < .005 else value
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=14,
                    color='white' if abs(value) > .7 else 'black')
    fig.tight_layout()
    save(fig, output, stem)


def rank_pair(output, stem, panels, colorbar_label):
    """Two 13x13 matrices with all 338 values, preserving publication layout."""
    width, height, side = 6.3, 3.78, 2.60
    style = {'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
             'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
             'axes.unicode_minus': False, 'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(width, height), dpi=200)
        annotations = []
        for x, (matrix, title) in zip((.42, 3.60), panels):
            ax = fig.add_axes([x/width, .80/height, side/width, side/height])
            mesh = ax.pcolormesh(np.arange(14), np.arange(14), matrix,
                                cmap='coolwarm', norm=Normalize(-1, 1),
                                shading='flat', edgecolors='white', linewidth=.30)
            ax.set_xlim(0, 13)
            ax.set_ylim(13, 0)
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(13)+.5, LABELS, rotation=90)
            ax.set_yticks(np.arange(13)+.5, LABELS)
            ax.tick_params(axis='both', which='both', length=0, pad=2.3, labelsize=7)
            ax.set_title(title, fontsize=8.3, pad=7)
            for spine in ax.spines.values():
                spine.set_linewidth(.35)
                spine.set_edgecolor('#b5b5b5')
            texts = []
            for i in range(13):
                for j in range(13):
                    value = f'{matrix[i,j]:.2f}'
                    if value == '-0.00':
                        value = '0.00'
                    texts.append(ax.text(j+.5, i+.5, value, ha='center', va='center',
                                         fontsize=6.1, color='white' if abs(matrix[i,j]) >= .66 else '#161616'))
            annotations.append((ax, texts))
        cax = fig.add_axes([1.62/width, .205/height, 4.07/width, .095/height])
        cb = fig.colorbar(mesh, cax=cax, orientation='horizontal',
                          ticks=[-1, -.5, 0, .5, 1], format='%.1f')
        cb.ax.tick_params(labelsize=7, length=2, width=.4, pad=2)
        cb.outline.set_linewidth(.4)
        fig.text(.42/width, .2525/height, colorbar_label, fontsize=7.4, ha='left', va='center')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, texts in annotations:
            box = ax.get_window_extent(renderer)
            for text in texts:
                bounds = text.get_window_extent(renderer)
                assert bounds.width < box.width/13 and bounds.height < box.height/13
            for artist in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.title]:
                bounds = artist.get_window_extent(renderer)
                assert bounds.x0 >= 0 and bounds.y0 >= 0
                assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
        save(fig, output, stem, tight=False)
    return {'annotations': 338, 'all_annotations_fit_cells': True,
            'all_axis_labels_inside_page': True, 'figure_size_inches': [width, height]}


def potential_histograms(data, output):
    records = {}
    for stem, limits, title in [('hist_V', (0, 2000), 'Histogram of $V$'),
                                ('hist_V_zoomup', (0, 5e-5), 'Histogram of $V$ (zoomed-in)')]:
        values = data.loc[data.FunctionValue.between(*limits), 'FunctionValue']
        fig, ax = plt.subplots(figsize=(8, 5))
        heights, edges, _ = ax.hist(values, bins=50)
        ax.set(xlim=limits, yscale='log')
        ax.set_xlabel('$V$', fontsize=18)
        ax.set_ylabel('count (log scale)', fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        fig.tight_layout()
        assert int(heights.sum()) == len(values)
        # Keep the laid-out canvas: PDF tight-bbox handling can clip the
        # y-label when the x-axis offset is expressed in scientific notation.
        save(fig, output, stem, tight=False)
        pd.DataFrame({'left_edge': edges[:-1], 'right_edge': edges[1:],
                      'count': heights.astype(int)}).to_csv(output/f'{stem}_bins.csv', index=False)
        records[stem] = {'displayed': len(values), 'outside': len(data)-len(values),
                         'selection_range': limits, 'bins': 50,
                         'bin_definition': 'Equal width across selected minimum and maximum, as in manuscript'}
    return records


def density(frame, output, stem, x, y, xlim, ylim, labelmap, susy=False):
    selected = frame[x].between(*xlim) & frame[y].between(*ylim)
    sub = frame.loc[selected]
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(sub[x], sub[y], gridsize=150, bins='log')
    ax.set_xlabel(labelmap[x], fontsize=14)
    ax.set_ylabel(labelmap[y], fontsize=14)
    ax.set_title(('SUSY ' if susy else '')+labelmap[y]+' vs '+labelmap[x], fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('count (log scale)', fontsize=14)
    fig.tight_layout()
    assert int(hb.get_array().sum()) == len(sub)
    bins = pd.DataFrame(hb.get_offsets(), columns=['x', 'y'])
    bins['count'] = np.asarray(hb.get_array(), dtype=int)
    bins.to_csv(output/f'{stem}_bins.csv', index=False)
    axes_limits = [list(ax.get_xlim()), list(ax.get_ylim())]
    save(fig, output, stem)
    return {'total': len(frame), 'displayed': len(sub), 'outside': len(frame)-len(sub),
            'outside_percent': 100*(len(frame)-len(sub))/len(frame),
            'x': x, 'y': y, 'x_selection': xlim, 'y_selection': ylim,
            'axes_limits': axes_limits, 'gridsize': 150, 'bins': 'log'}


def instanton_histogram(ratio, output):
    percent = 100*ratio
    edges = np.linspace(0, .05, 51)
    counts, _ = np.histogram(percent, bins=edges)
    if counts.sum() != len(ratio):
        raise ValueError('Some instanton ratios lie outside the manuscript [0,0.05]% window')
    median, maximum = float(np.median(percent)), float(np.max(percent))
    style = {'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 200}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.7, 4.7), layout='constrained')
        ax.stairs(counts, edges, fill=True, color='#516CB0', alpha=.78,
                  linewidth=.6, edgecolor='#324E8A')
        ax.axvline(median, color='#C85E23', linewidth=1.5, linestyle='--', label=f'Median: {median:.5f}%')
        ax.set(xlim=(0, .05), yscale='log', ylim=(.8, max(counts)*2.3),
               xlabel=r'$|F_{\mathrm{inst}}|/|F_{\mathrm{poly}}|$ (%)',
               ylabel='Number of solutions (log scale)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title('Relative instanton correction to the prepotential', loc='left', pad=36, fontsize=14)
        ax.text(0, 1.035, f'All {len(ratio):,} reliable solutions  |  Total q-degree '+r'$\leq 7$',
                transform=ax.transAxes, fontsize=10, color='#45515E')
        ax.text(.97, .96, f'Maximum: {maximum:.5f}%\n50 equal-width bins',
                transform=ax.transAxes, ha='right', va='top', fontsize=10, color='#45515E')
        ax.legend(loc='upper right', bbox_to_anchor=(1, .84), frameon=False, fontsize=10)
        ax.grid(axis='y', which='major', alpha=.16, zorder=0)
        ax.set_axisbelow(True)
        save(fig, output, 'F_correction_histogram', tight=False)
    pd.DataFrame({'left_edge_percent': edges[:-1], 'right_edge_percent': edges[1:],
                  'solution_count': counts, 'fraction': counts/len(ratio)}).to_csv(
        output/'F_correction_histogram_bins.csv', index=False)
    return {'median_ratio': float(np.median(ratio)), 'maximum_ratio': float(np.max(ratio)),
            'median_percent': median, 'maximum_percent': maximum,
            'zero_float64_ratios': int((ratio == 0).sum()), 'rows_in_bins': int(counts.sum()),
            'bins': 50, 'range_percent': [0, .05],
            'definition': '|F_inst^(total q degree <=7)|/|F_poly including zeta(3)|',
            'method': 'Direct evaluation from saved coordinates and integer instanton numbers with exact rational multicover coefficients',
            'limitations': 'No bound on degree>7 terms, derivatives, Hessian or vacuum shifts; tiny float64 terms can underflow to zero'}
