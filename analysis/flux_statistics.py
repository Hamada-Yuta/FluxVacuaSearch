"""Potential, flux-component rates and exact identity groups of selected solutions.

Run: python -m analysis.flux_statistics --data data/selected_solutions.parquet
     --output results/flux_statistics
Every saved row has equal weight; the input is not averaged or reordered.
"""
from pathlib import Path
import json
import os
import sys
import time

OUT = None
DATA = None
from ._ml_io import parse_cli, environment, check_sample
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'

import numpy as np
import pandas as pd
import pyarrow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

pyarrow.set_cpu_count(2)
pyarrow.set_io_thread_count(2)
COLS = ['FunctionValue','taur','taui','Z1r','Z1i','Z2r','Z2i','m1','m2','m3','n1','n2','n3']
FLUX = COLS[-6:]
INV = ['omega12','omega13','omega23','G11','G12','G13','G22','G23','G33']
THRESHOLD = 10**(-4.5)
QUANTILES = [0,.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,.999,1]
# External scan denominators reported in the paper, not inferred from the
# selected sample (which contains no row for an orbit without retained solutions).
SCANNED_FLUX_CHOICES = 30_770_178
SCANNED_Z4_ORBITS = 27_330_992


def quantiles(x):
    return {str(q):float(v) for q,v in zip(QUANTILES,np.quantile(x,QUANTILES))}


def invariants(raw):
    m,n=raw[:,:3],raw[:,3:]
    return np.column_stack([m[:,a]*n[:,b]-m[:,b]*n[:,a] for a,b in [(0,1),(0,2),(1,2)]]+
                           [m[:,a]*m[:,b]+n[:,a]*n[:,b] for a,b in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]])


def canonical_z4(raw):
    result=raw.copy()
    m,n=raw[:,:3],raw[:,3:]
    indices=np.arange(len(raw))
    for candidate in [np.column_stack([-n,m]),-raw,np.column_stack([n,-m])]:
        different=candidate!=result
        first=np.argmax(different,axis=1)
        smaller=np.any(different,axis=1)&(candidate[indices,first]<result[indices,first])
        result[smaller]=candidate[smaller]
    return result


def summarize_groups(counts,lowcounts):
    mixed=(lowcounts>0)&(lowcounts<counts)
    multiple=counts>1
    return {'unique_groups':len(counts),'groups_with_multiple_rows':int(multiple.sum()),
            'rows_in_multiple_row_groups':int(counts[multiple].sum()),
            'maximum_row_multiplicity':int(counts.max()),
            'mixed_low_nonlow_groups':int(mixed.sum()),
            'rows_in_mixed_groups':int(counts[mixed].sum()),
            'low_rows_in_mixed_groups':int(lowcounts[mixed].sum())}


def plot_figures(values,low,q01,q99,rates):
    plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':12,
                         'figure.dpi':150,'savefig.dpi':180,'pdf.fonttype':42,
                         'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    logs=np.log10(values[values>0])
    start=np.floor(logs.min()*4)/4
    stop=np.ceil(logs.max()*4)/4+.25
    edges=np.arange(start,stop+.125,.25)
    fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
    ax.hist(logs,bins=edges,color='#355D85',edgecolor='white',linewidth=.25,
            label='Retained solutions')
    if q01>0:
        ax.axvspan(np.log10(q01),np.log10(q99),color='#F2C66D',alpha=.24,
                   label=r'$1\%$--$99\%$ range of $V$')
    ax.axvline(-4.5,color='#B84937',ls='--',lw=1.8,label=r'$V_*=10^{-4.5}$')
    ax.set(yscale='log',xlabel=r'$\log_{10}V$',ylabel='Number of retained solutions per bin',
           title='Potential distribution of the retained solutions')
    ax.grid(axis='y',alpha=.18)
    ax.legend(loc='upper left',fontsize=9,framealpha=.95)
    ax.text(.99,.98,f'$N={len(values):,}$\nLow-$V$: {low.sum():,} ({100*low.mean():.2f}%)',
            transform=ax.transAxes,ha='right',va='top',fontsize=10,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
    fig.supxlabel('Each saved solution has equal weight; bin width = 0.25 dex',fontsize=10)
    for extension in ['pdf','png']:
        fig.savefig(OUT/f'ml_hist_logV.{extension}')
    plt.close(fig)
    hist_counts,_=np.histogram(logs,edges)
    pd.DataFrame({'log10V_left':edges[:-1],'log10V_right':edges[1:],'rows':hist_counts}).to_csv(
        OUT/'histogram_logV.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(10.5,6),sharey=True,layout='constrained')
    ymax=min(100,max(100*rates.low_fraction.max()*1.12,100*low.mean()*1.5))
    for ax,flux in zip(axes.ravel(),FLUX):
        sub=rates.loc[rates.flux==flux]
        ax.plot(sub.value,100*sub.low_fraction,'o-',ms=3.4,lw=1.2,color='#355D85')
        ax.axhline(100*low.mean(),ls='--',lw=1.3,color='#B84937')
        ax.set_title('$'+flux[0]+'_'+flux[1]+'$')
        ax.set_ylim(0,ymax)
        lo,hi=int(sub.value.min()),int(sub.value.max())
        ax.set_xlim(lo-.75,hi+.75)
        ax.set_xticks(np.arange((lo//4)*4,hi+1,4))
        ax.grid(axis='y',alpha=.2)
        ax.set_xlabel('Stored flux component')
    axes[0,0].set_ylabel('Low-$V$ solutions (%)')
    axes[1,0].set_ylabel('Low-$V$ solutions (%)')
    fig.suptitle(r'Fraction with $V<10^{-4.5}$ among retained solutions',fontsize=14)
    fig.supxlabel(f'Equal weight per saved solution; dashed line = overall rate ({100*low.mean():.2f}%)',fontsize=10)
    for extension in ['pdf','png']:
        fig.savefig(OUT/f'ml_vacuum_rate.{extension}')
    plt.close(fig)


def main():
    global OUT, DATA
    args = parse_cli(__doc__, predictions=False)
    OUT, DATA = args.output, args.data
    began=time.monotonic()
    frame=pd.read_parquet(DATA,columns=COLS)
    check_sample(frame, COLS)
    assert list(frame.columns)==COLS
    values=frame.FunctionValue.to_numpy()
    low=values<THRESHOLD
    susy=low
    assert int(susy.sum())==127241
    finite=np.isfinite(frame.to_numpy())
    assert finite.all()
    lcs=(frame.Z1i.to_numpy()>=1)&(frame.Z2i.to_numpy()>=1)
    weak=frame.taui.to_numpy()>1
    q01,q99=np.quantile(values,[.01,.99])
    trimmed=(values>=q01)&(values<=q99)
    summary={'source':str(DATA),'rows':len(frame),'columns':COLS,
             'weighting':'Every saved solution/row has equal weight; this is not one row or equal weight per distinct flux choice.',
             'lowV':{'condition':'FunctionValue < 10**(-4.5)','threshold':THRESHOLD,'rows':int(low.sum()),'fraction':float(low.mean())},
             'SUSY':{'condition':'FunctionValue < 10**(-4.5)','rows':int(susy.sum()),'fraction':float(susy.mean())},
             'V':{'finite':int(np.isfinite(values).sum()),'nonfinite':int((~np.isfinite(values)).sum()),
                  'zero':int((values==0).sum()),'negative':int((values<0).sum()),'positive':int((values>0).sum()),
                  'min':float(values.min()),'max':float(values.max()),'quantiles':quantiles(values)},
             'log10V_positive':{'rows':int((values>0).sum()),'excluded_nonpositive':int((values<=0).sum()),
                               'min':float(np.log10(values[values>0]).min()),'max':float(np.log10(values[values>0]).max()),
                               'quantiles':quantiles(np.log10(values[values>0]))},
             'log10_max_V_1e_minus16':{'rows_below_floor':int((values<1e-16).sum()),
                                    'quantiles':quantiles(np.log10(np.maximum(values,1e-16)))},
             'nonfinite_by_column':{col:int((~finite[:,i]).sum()) for i,col in enumerate(COLS)},
             'physical_cuts':{},
             'V_quantile_trim':{'lower_q01':float(q01),'upper_q99':float(q99),'inclusive':True,
                                'rows_retained':int(trimmed.sum()),'lowV_retained':int((trimmed&low).sum()),
                                'lowV_removed':int((~trimmed&low).sum()),'SUSY_retained':int((trimmed&susy).sum())}}
    for name,mask in [('all',np.ones(len(frame),dtype=bool)),('lowV',low),('SUSY',susy)]:
        summary['physical_cuts'][name]={'rows':int(mask.sum()),'LCS_Z1i_and_Z2i_ge1':int((mask&lcs).sum()),
                                        'stored_taui_gt1':int((mask&weak).sum()),'both':int((mask&lcs&weak).sum()),
                                        'stored_taui_positive':int((mask&(frame.taui.to_numpy()>0)).sum())}
    print(f'Source verified; low V {low.sum():,}; SUSY {susy.sum():,}.',flush=True)
    raw=frame[FLUX].to_numpy(np.int32)
    raw_unique,raw_first,rawid,rawcounts=np.unique(raw,axis=0,return_index=True,return_inverse=True,return_counts=True)
    rawid=np.asarray(rawid).reshape(-1)
    rawlow=np.bincount(rawid,weights=low,minlength=len(raw_unique)).astype(np.int64)
    canonical=canonical_z4(raw_unique)
    orbit_unique,orbit_first_raw,raw_to_orbit=np.unique(canonical,axis=0,return_index=True,return_inverse=True)
    raw_to_orbit=np.asarray(raw_to_orbit).reshape(-1)
    orbitid=raw_to_orbit[rawid]
    orbitcounts=np.bincount(orbitid,minlength=len(orbit_unique))
    orbitlow=np.bincount(orbitid,weights=low,minlength=len(orbit_unique)).astype(np.int64)
    assert orbitcounts.min() >= 1 and orbitcounts.max() <= 40
    found = np.arange(1, 41)
    amounts = np.bincount(orbitcounts, minlength=41)[1:41]
    frequency = pd.DataFrame({'N_found': found, 'n_orbits': amounts})
    frequency['active_orbit_fraction_percent'] = 100 * amounts / len(orbit_unique)
    frequency['scanned_orbit_fraction_percent'] = 100 * amounts / SCANNED_Z4_ORBITS
    frequency['n_endpoints'] = found * amounts
    frequency.to_csv(OUT/'solutions_per_orbit.csv', index=False)
    without = SCANNED_Z4_ORBITS - len(orbit_unique)
    with_zero = pd.concat([pd.DataFrame([{'N_found': 0, 'n_orbits': without,
        'active_orbit_fraction_percent': np.nan,
        'scanned_orbit_fraction_percent': 100 * without / SCANNED_Z4_ORBITS,
        'n_endpoints': 0}]), frequency], ignore_index=True)
    with_zero.to_csv(OUT/'solutions_per_orbit_including_zero.csv', index=False)
    assert int(frequency.n_endpoints.sum()) == len(frame)
    assert int(with_zero.n_orbits.sum()) == SCANNED_Z4_ORBITS
    summary['N_found'] = frequency.to_dict(orient='records')
    summary['N_found_definition'] = 'Retained solution rows per canonical Z4 flux orbit; fractions use active or scanned orbits as named.'
    summary['scan_denominators'] = {
        'scanned_flux_choices': SCANNED_FLUX_CHOICES,
        'scanned_Z4_orbits': SCANNED_Z4_ORBITS,
        'orbits_with_retained_solutions': len(orbit_unique),
        'orbits_without_retained_solutions': without,
        'without_retained_solutions_percent': 100 * without / SCANNED_Z4_ORBITS,
        'source': 'Fixed scan counts reported in the paper; not reconstructed from this selected Parquet.',
        'interpretation': 'No retained solution under the finite search and selection; not proof of absence of a vacuum.'}
    rawinv=invariants(raw_unique)
    assert np.array_equal(rawinv,invariants(canonical))
    inv_unique,inv_first_raw,raw_to_inv=np.unique(rawinv,axis=0,return_index=True,return_inverse=True)
    raw_to_inv=np.asarray(raw_to_inv).reshape(-1)
    invid=raw_to_inv[rawid]
    invcounts=np.bincount(invid,minlength=len(inv_unique))
    invlow=np.bincount(invid,weights=low,minlength=len(inv_unique)).astype(np.int64)
    invraw=np.bincount(raw_to_inv,minlength=len(inv_unique))
    orbit_to_inv=raw_to_inv[orbit_first_raw]
    invorbit=np.bincount(orbit_to_inv,minlength=len(inv_unique))
    assert np.array_equal(raw_to_inv,orbit_to_inv[raw_to_orbit])
    assert invcounts.sum()==len(frame) and invraw.sum()==len(raw_unique) and invorbit.sum()==len(orbit_unique)
    identities={'raw_flux':summarize_groups(rawcounts,rawlow),'Z4_orbit':summarize_groups(orbitcounts,orbitlow),
                'invariant_tuple':summarize_groups(invcounts,invlow),
                'invariant_order':INV,'Z4_canonical_rule':'Lexicographic minimum of (m,n),(-n,m),(-m,-n),(n,-m)',
                'omega_definition':'omegaAB=mA*nB-mB*nA','G_definition':'GAB=mA*mB+nA*nB'}
    for name,multiplicity in [('rows',invcounts),('raw_fluxes',invraw),('Z4_orbits',invorbit)]:
        many=multiplicity>1
        identities['invariant_tuple']['multiple_'+name]={'tuples':int(many.sum()),'member_rows':int(invcounts[many].sum()),
                                                        'member_lowV_rows':int(invlow[many].sum()),'maximum':int(multiplicity.max())}
        counts=np.bincount(multiplicity)
        mult=np.flatnonzero(counts)
        pd.DataFrame({'multiplicity':mult,'number_of_invariant_tuples':counts[mult]}).to_csv(
            OUT/f'tuple_{name}_multiplicity_distribution.csv',index=False)
    mixed=(invlow>0)&(invlow<invcounts)
    identities['invariant_tuple']['mixed_single_orbit_tuples']=int((mixed&(invorbit==1)).sum())
    identities['invariant_tuple']['mixed_multiple_orbit_tuples']=int((mixed&(invorbit>1)).sum())
    summary['identity']=identities
    tuple_table=pd.DataFrame(inv_unique,columns=INV)
    tuple_table['invariant_id']=np.arange(len(inv_unique))
    tuple_table['n_rows']=invcounts
    tuple_table['n_raw_fluxes']=invraw
    tuple_table['n_Z4_orbits']=invorbit
    tuple_table['n_lowV']=invlow
    tuple_table['mixed_label']=mixed
    tuple_table.to_parquet(OUT/'tuple_multiplicity.parquet',index=False)
    raw_table=pd.DataFrame(raw_unique,columns=FLUX)
    raw_table['raw_flux_id']=np.arange(len(raw_unique))
    raw_table['orbit_id']=raw_to_orbit
    raw_table['invariant_id']=raw_to_inv
    raw_table['n_rows']=rawcounts
    raw_table['n_lowV']=rawlow
    raw_table.to_parquet(OUT/'raw_flux_groups.parquet',index=False)
    pd.DataFrame({'source_row':np.arange(len(frame)), 'raw_flux_id':rawid,'orbit_id':orbitid,
                  'invariant_id':invid,'lowV':low}).to_parquet(OUT/'row_identity.parquet',index=False)
    print(f'Unique raw/orbit/invariant: {len(raw_unique):,}/{len(orbit_unique):,}/{len(inv_unique):,}.',flush=True)
    mixed_raw=np.flatnonzero((rawlow>0)&(rawlow<rawcounts))
    ranked=mixed_raw[np.argsort(-rawcounts[mixed_raw],kind='stable')[:5]]
    examples=[]
    for rid in ranked:
        rows=np.flatnonzero(rawid==rid)
        minrow=rows[np.argmin(values[rows])]
        maxrow=rows[np.argmax(values[rows])]
        assert low[minrow] and not low[maxrow]
        def record(i):
            return {'source_row':int(i),'lowV':bool(low[i]),
                    **{c:(int(frame[c].iloc[i]) if c in FLUX else float(frame[c].iloc[i])) for c in COLS}}
        examples.append({'raw_flux_id':int(rid),'raw_flux':raw_unique[rid].tolist(),
                         'orbit_id':int(raw_to_orbit[rid]),'invariant_id':int(raw_to_inv[rid]),
                         'invariant_tuple':rawinv[rid].tolist(),'n_rows':int(rawcounts[rid]),'n_lowV':int(rawlow[rid]),
                         'low_record':record(minrow),'nonlow_record':record(maxrow)})
    summary['mixed_same_flux_examples']=examples
    (OUT/'mixed_same_flux_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    train,test=train_test_split(np.arange(len(frame)),test_size=.25,stratify=low,random_state=0)
    np.savez_compressed(OUT/'classification_split.npz',train=train,test=test)
    overlap={}
    for name,ids,nkeys in [('raw_flux',rawid,len(raw_unique)),('Z4_orbit',orbitid,len(orbit_unique)),
                           ('invariant_tuple',invid,len(inv_unique))]:
        train_seen=np.zeros(nkeys,dtype=bool)
        train_seen[ids[train]]=True
        shared=train_seen[ids[test]]
        overlap[name]={'test_rows_with_training_match':int(shared.sum()),'test_fraction':float(shared.mean()),
                       'lowV_test_rows_with_training_match':int((shared&low[test]).sum()),
                       'lowV_test_fraction':float(shared[low[test]].mean()),
                       'nonlowV_test_rows_with_training_match':int((shared&~low[test]).sum()),
                       'nonlowV_test_fraction':float(shared[~low[test]].mean()),
                       'training_unique_keys':int(train_seen.sum()),'test_unique_keys':int(np.unique(ids[test]).size),
                       'test_unique_keys_with_training_match':int(np.unique(ids[test][shared]).size)}
    summary['random_split_identity_overlap']={'split':'Stored source row order; 75/25 stratified by lowV, random_state=0',
                                             'train_rows':len(train),'test_rows':len(test),
                                             'train_lowV':int(low[train].sum()),'test_lowV':int(low[test].sum()),
                                             'by_identity':overlap}
    rates=[]
    for j,name in enumerate(FLUX):
        for value in np.unique(raw[:,j]):
            mask=raw[:,j]==value
            total=int(mask.sum())
            positives=int((mask&low).sum())
            rates.append({'flux':name,'value':int(value),'rows':total,'lowV_rows':positives,'low_fraction':positives/total})
    rates=pd.DataFrame(rates)
    rates.to_csv(OUT/'flux_lowV_rates.csv',index=False)
    summary['individual_flux_rates']={'min':float(rates.low_fraction.min()),'max':float(rates.low_fraction.max()),
                                      'minimum_rows_per_flux_bin':int(rates.rows.min()),
                                      'equal_solution_weight':True}
    plot_figures(values,low,q01,q99,rates)
    summary['figures']={'histogram':'Positive V only, no clipping, bin width 0.25 dex, logarithmic count axis.',
                        'flux_rate':'Fraction lowV among saved solution rows having each flux component; multiple rows per flux contribute separately.',
                        'PDF_PNG':['ml_hist_logV','ml_vacuum_rate']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    footnote={'rows':len(frame),'raw_flux_choices':len(raw_unique),'Z4_flux_orbits':len(orbit_unique),
              'invariant_tuples':len(inv_unique),'lowV_rows':int(low.sum()),'lowV_fraction':float(low.mean()),
              'SUSY_rows':int(susy.sum()),'mixed_raw_flux_choices':identities['raw_flux']['mixed_low_nonlow_groups'],
              'mixed_invariant_tuples':identities['invariant_tuple']['mixed_low_nonlow_groups'],
              'test_row_training_match_fraction':{name:value['test_fraction'] for name,value in overlap.items()},
              'weighting':summary['weighting'],'examples':examples[:2]}
    (OUT/'footnote_summary.json').write_text(json.dumps(footnote,indent=2)+'\n')
    settings={'source':str(DATA), 'environment':environment([np,pd,pyarrow,matplotlib,sklearn]),
              'source_row_convention':'Zero-based physical Parquet row position; no reordering before splitting',
              'threshold':THRESHOLD,'weighting':summary['weighting'], 'cpu_limit':2,
              'classification_split':{'test_size':.25,'stratify':'lowV','random_state':0},
              'elapsed_seconds':time.monotonic()-began}
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(json.dumps(footnote,indent=2),flush=True)
    print(f'Completed in {time.monotonic()-began:.1f}s',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
