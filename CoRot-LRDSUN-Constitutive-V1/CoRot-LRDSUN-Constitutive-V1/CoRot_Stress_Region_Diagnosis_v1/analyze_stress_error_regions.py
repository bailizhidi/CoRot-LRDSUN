#!/usr/bin/env python3
"""
Region-wise stress-error diagnosis for bent tube VTU/NPZ results.

Primary use case:
  - error field: Outer_SPOS_S_Mises_AbsError_MPa
  - X_ref: initial/reference coordinates
  - X_def: current/deformed coordinates (or VTU points as fallback)

Outputs per file:
  ring_profile.csv
  region_metrics.csv
  exclude_end_rings.csv
  summary.json
  ring_error_profile.png
  region_mae.png

Also writes aggregate CSVs when multiple files are processed.
"""
from __future__ import annotations
import argparse, csv, json, math, os, re
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np


def _read_vtu(path: str):
    errors=[]
    try:
        import pyvista as pv
        m = pv.read(path)
        return np.asarray(m.points), {k:np.asarray(m.point_data[k]) for k in m.point_data.keys()}, 'pyvista'
    except Exception as e:
        errors.append(f'pyvista: {e}')
    try:
        import meshio
        m = meshio.read(path)
        return np.asarray(m.points), {k:np.asarray(v) for k,v in m.point_data.items()}, 'meshio'
    except Exception as e:
        errors.append(f'meshio: {e}')
    raise RuntimeError('Cannot read VTU; install pyvista or meshio. ' + ' | '.join(errors))


def _find_key(keys, exact: Optional[str], candidates: List[str], require_all_tokens: Optional[List[str]]=None):
    if exact:
        if exact in keys:
            return exact
        # case-insensitive exact
        low={k.lower():k for k in keys}
        if exact.lower() in low:
            return low[exact.lower()]
        raise KeyError(f'Requested field {exact!r} not found. Available: {list(keys)}')
    low={k.lower():k for k in keys}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    if require_all_tokens:
        hits=[]
        for k in keys:
            kl=k.lower()
            if all(tok.lower() in kl for tok in require_all_tokens):
                hits.append(k)
        if len(hits)==1:
            return hits[0]
        if len(hits)>1:
            # prefer outer/SPOS if requested by tokens
            hits=sorted(hits, key=lambda s:(0 if 'outer' in s.lower() or 'spos' in s.lower() else 1, len(s)))
            return hits[0]
    return None


def load_any(path: Path, error_field=None, ref_field=None, def_field=None):
    if path.suffix.lower()=='.vtu':
        points, pd, backend = _read_vtu(str(path))
        keys=list(pd.keys())
        rk=_find_key(keys, ref_field, ['X_ref','x_ref','mesh_pos','ref_pos','reference_pos'])
        dk=_find_key(keys, def_field, ['X_def','x_def','world_pos','current_pos','deformed_pos'])
        ek=_find_key(keys, error_field,
                     ['Outer_SPOS_S_Mises_AbsError_MPa','S_Mises_AbsError_MPa','S_mises_abs_error_mpa'],
                     require_all_tokens=['mises','error'])
        if rk is None:
            raise KeyError(f'Could not auto-detect reference coordinates in VTU. Available point fields: {keys}')
        X_ref=np.asarray(pd[rk], dtype=float)
        X_def=np.asarray(pd[dk], dtype=float) if dk is not None else np.asarray(points,dtype=float)
        if ek is None:
            # Try compute from true/pred
            truek=_find_key(keys,None,['Outer_SPOS_S_Mises_True_MPa','Outer_SPOS_S_Mises_GT_MPa','S_Mises_True_MPa'],require_all_tokens=['mises','true'])
            predk=_find_key(keys,None,['Outer_SPOS_S_Mises_Pred_MPa','S_Mises_Pred_MPa'],require_all_tokens=['mises','pred'])
            if truek and predk:
                err=np.abs(np.asarray(pd[predk],dtype=float).reshape(-1)-np.asarray(pd[truek],dtype=float).reshape(-1))
                ek=f'computed_abs({predk}-{truek})'
            else:
                raise KeyError(f'Could not auto-detect Mises abs-error field. Available point fields: {keys}')
        else:
            err=np.asarray(pd[ek],dtype=float).reshape(-1)
        return X_ref, X_def, err, {'backend':backend,'ref_field':rk,'def_field':dk or '<VTU points>','error_field':ek}

    if path.suffix.lower()=='.npz':
        z=np.load(path,allow_pickle=True)
        keys=list(z.files)
        rk=_find_key(keys, ref_field, ['X_ref','x_ref','mesh_pos','ref_pos','reference_pos'])
        dk=_find_key(keys, def_field, ['X_def','x_def','world_pos','current_pos','deformed_pos'])
        ek=_find_key(keys, error_field,
                     ['Outer_SPOS_S_Mises_AbsError_MPa','S_Mises_AbsError_MPa','S_mises_abs_error_mpa'],
                     require_all_tokens=['mises','error'])
        if rk is None or dk is None:
            raise KeyError(f'NPZ needs reference and deformed coordinates. Available keys: {keys}')
        X_ref=np.asarray(z[rk],dtype=float)
        X_def=np.asarray(z[dk],dtype=float)
        # Accept leading time dimension only if singleton; user should choose one frame otherwise.
        while X_ref.ndim>2 and X_ref.shape[0]==1: X_ref=X_ref[0]
        while X_def.ndim>2 and X_def.shape[0]==1: X_def=X_def[0]
        if X_ref.ndim!=2 or X_ref.shape[1]!=3 or X_def.ndim!=2 or X_def.shape[1]!=3:
            raise ValueError(f'Expected X_ref/X_def [N,3], got {X_ref.shape}, {X_def.shape}. Use a per-frame NPZ/VTU.')
        if ek is None:
            truek=_find_key(keys,None,['Outer_SPOS_S_Mises_True_MPa','Outer_SPOS_S_Mises_GT_MPa','S_Mises_True_MPa'],require_all_tokens=['mises','true'])
            predk=_find_key(keys,None,['Outer_SPOS_S_Mises_Pred_MPa','S_Mises_Pred_MPa'],require_all_tokens=['mises','pred'])
            if truek and predk:
                err=np.abs(np.asarray(z[predk],dtype=float).reshape(-1)-np.asarray(z[truek],dtype=float).reshape(-1))
                ek=f'computed_abs({predk}-{truek})'
            else:
                raise KeyError(f'Could not auto-detect error or true/pred Mises fields. Available keys: {keys}')
        else:
            err=np.asarray(z[ek],dtype=float).reshape(-1)
        return X_ref, X_def, err, {'backend':'numpy','ref_field':rk,'def_field':dk,'error_field':ek}
    raise ValueError(f'Unsupported file type: {path}')


def pca_axis(X):
    Xc=X-X.mean(axis=0,keepdims=True)
    _,_,vt=np.linalg.svd(Xc,full_matrices=False)
    a=vt[0]
    # deterministic sign: prefer positive Z, then Y, then X
    for idx in [2,1,0]:
        if abs(a[idx])>1e-8:
            if a[idx]<0: a=-a
            break
    return a/a.dot(a)**0.5


def make_rings(s_raw, L):
    # Quantize projection to absorb tiny floating noise. For ~100 mm tubes, this is ~1e-3 mm.
    tol=max(L*1e-5, 1e-6)
    q=np.round(s_raw/tol).astype(np.int64)
    uq=np.unique(q)
    # If too many unique values, fallback to equal bins (~1 mm scale or 100 bins max)
    if len(uq)>500:
        nbin=min(160,max(40,int(round(L))))
        edges=np.linspace(s_raw.min()-1e-12,s_raw.max()+1e-12,nbin+1)
        rid=np.clip(np.digitize(s_raw,edges)-1,0,nbin-1)
        centers=np.array([s_raw[rid==i].mean() if np.any(rid==i) else np.nan for i in range(nbin)])
        valid=np.isfinite(centers)
        remap={old:new for new,old in enumerate(np.where(valid)[0])}
        rid=np.array([remap[i] for i in rid if i in remap]) if not np.all(valid) else rid
        centers=centers[valid]
        return rid,centers,'equal_bins'
    # exact-ish reference rings
    order=np.argsort(uq)
    uq=uq[order]
    mapq={int(v):i for i,v in enumerate(uq)}
    rid=np.array([mapq[int(v)] for v in q],dtype=int)
    centers=np.array([s_raw[rid==i].mean() for i in range(len(uq))])
    return rid,centers,'reference_rings'


def moving_average(P, w=3):
    if len(P)<3 or w<=1: return P.copy()
    out=P.copy()
    h=w//2
    for i in range(len(P)):
        a=max(0,i-h); b=min(len(P),i+h+1)
        out[i]=P[a:b].mean(axis=0)
    return out


def turning_angles_deg(C):
    n=len(C); ang=np.zeros(n,dtype=float)
    if n<3: return ang
    C=moving_average(C,3)
    for i in range(1,n-1):
        v1=C[i]-C[i-1]; v2=C[i+1]-C[i]
        n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
        if n1<1e-12 or n2<1e-12: continue
        c=float(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))
        ang[i]=math.degrees(math.acos(c))
    return ang


def largest_contiguous(mask):
    best=None; cur=None
    for i,v in enumerate(mask):
        if v and cur is None: cur=[i,i]
        elif v and cur is not None: cur[1]=i
        elif (not v) and cur is not None:
            if best is None or cur[1]-cur[0]>best[1]-best[0]: best=cur
            cur=None
    if cur is not None and (best is None or cur[1]-cur[0]>best[1]-best[0]): best=cur
    return best


def detect_regions(ring_centers_norm, C, end_rings=1, bend_s_min=None, bend_s_max=None):
    nr=len(ring_centers_norm)
    turn=turning_angles_deg(C)
    if bend_s_min is not None and bend_s_max is not None:
        bend=np.where((ring_centers_norm>=bend_s_min)&(ring_centers_norm<=bend_s_max))[0]
        if len(bend)==0: raise ValueError('Manual bend range selects zero rings.')
        b0,b1=int(bend[0]),int(bend[-1])
        mode='manual_s_range'
    else:
        usable=np.ones(nr,dtype=bool)
        usable[:end_rings+1]=False; usable[max(0,nr-end_rings-1):]=False
        vals=turn[usable]
        mx=float(vals.max()) if vals.size else 0.0
        thr=max(0.25,0.20*mx)
        mask=(turn>=thr)&usable
        seg=largest_contiguous(mask)
        if seg is None:
            # fallback center 50% of arc length
            b0=int(round(0.25*(nr-1))); b1=int(round(0.75*(nr-1)))
            mode='fallback_center50'
        else:
            b0=max(end_rings+1,seg[0]-2)
            b1=min(nr-end_rings-2,seg[1]+2)
            mode=f'auto_turning_angle_thr_{thr:.4g}deg'
    reg=np.empty(nr,dtype=object)
    for i in range(nr):
        if i<end_rings: reg[i]='end_left'
        elif i>=nr-end_rings: reg[i]='end_right'
        elif i<b0: reg[i]='straight_left'
        elif i<=b1: reg[i]='bend'
        else: reg[i]='straight_right'
    return reg,turn,{'bend_ring_start':b0,'bend_ring_end':b1,'mode':mode}


def metrics(err):
    err=np.asarray(err,dtype=float)
    err=err[np.isfinite(err)]
    if len(err)==0:
        return dict(n=0,mae=np.nan,rmse=np.nan,p95=np.nan,max=np.nan,sum_abs=0.0)
    return dict(n=int(len(err)),mae=float(np.mean(np.abs(err))),rmse=float(np.sqrt(np.mean(err*err))),
                p95=float(np.percentile(np.abs(err),95)),max=float(np.max(np.abs(err))),sum_abs=float(np.sum(np.abs(err))))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    if fieldnames is None:
        fieldnames=list(rows[0].keys())
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def analyze_one(path:Path,out_root:Path,error_field,ref_field,def_field,end_rings,bend_s_min,bend_s_max,make_plots=True):
    Xr,Xd,err,meta=load_any(path,error_field,ref_field,def_field)
    if len(Xr)!=len(err) or len(Xd)!=len(err):
        raise ValueError(f'Point count mismatch: X_ref={len(Xr)}, X_def={len(Xd)}, error={len(err)}')
    axis=pca_axis(Xr)
    s=np.dot(Xr-Xr.mean(axis=0),axis)
    s0=float(s.min()); s1=float(s.max()); L=s1-s0
    if L<=0: raise ValueError('Degenerate reference axial length.')
    sn=(s-s0)/L
    rid,ring_s_raw,ring_mode=make_rings(s,L)
    nr=len(ring_s_raw)
    ring_sn=np.array([(sn[rid==i].mean()) for i in range(nr)])
    C=np.array([Xd[rid==i].mean(axis=0) for i in range(nr)])
    reg,turn,regmeta=detect_regions(ring_sn,C,end_rings,bend_s_min,bend_s_max)

    sample_dir=out_root/path.stem
    sample_dir.mkdir(parents=True,exist_ok=True)
    ring_rows=[]
    for i in range(nr):
        m=metrics(err[rid==i])
        ring_rows.append({
            'file':path.name,'ring_id':i,'s_over_L':float(ring_sn[i]),'region':reg[i],
            'turn_angle_deg':float(turn[i]),'n_nodes':m['n'],'mae_MPa':m['mae'],'rmse_MPa':m['rmse'],
            'p95_MPa':m['p95'],'max_MPa':m['max']
        })
    write_csv(sample_dir/'ring_profile.csv',ring_rows)

    total_abs=float(np.sum(np.abs(err[np.isfinite(err)])))
    region_rows=[]
    order=['end_left','straight_left','bend','straight_right','end_right']
    for name in order:
        rings=np.where(reg==name)[0]
        mask=np.isin(rid,rings)
        m=metrics(err[mask])
        region_rows.append({
            'file':path.name,'region':name,'n_nodes':m['n'],'node_fraction':m['n']/len(err),
            'mae_MPa':m['mae'],'rmse_MPa':m['rmse'],'p95_MPa':m['p95'],'max_MPa':m['max'],
            'abs_error_contribution_fraction':(m['sum_abs']/total_abs if total_abs>0 else np.nan)
        })
    write_csv(sample_dir/'region_metrics.csv',region_rows)

    excl=[]
    for k in range(0,min(6,nr//2)):
        keep=np.ones(len(err),dtype=bool)
        if k>0:
            keep &= (rid>=k)&(rid<nr-k)
        m=metrics(err[keep])
        excl.append({'file':path.name,'exclude_rings_each_end':k,'n_nodes':m['n'],'mae_MPa':m['mae'],'rmse_MPa':m['rmse'],'p95_MPa':m['p95'],'max_MPa':m['max']})
    write_csv(sample_dir/'exclude_end_rings.csv',excl)

    summary={
        'file':str(path),'n_points':len(err),'axis':axis.tolist(),'reference_length':L,'ring_count':nr,'ring_detection':ring_mode,
        'fields':meta,'region_detection':regmeta,'end_rings_each_side':end_rings,
        'global':metrics(err),
        'region_metrics':region_rows,
        'exclude_end_rings':excl,
    }
    with open(sample_dir/'summary.json','w',encoding='utf-8') as f: json.dump(summary,f,indent=2,ensure_ascii=False)

    if make_plots:
        try:
            import matplotlib.pyplot as plt
            x=np.array([r['s_over_L'] for r in ring_rows]); y=np.array([r['mae_MPa'] for r in ring_rows])
            fig,ax=plt.subplots(figsize=(8,4.5))
            ax.plot(x,y,marker='o',markersize=2,linewidth=1)
            # region labels via spans
            for name in order:
                rr=np.where(reg==name)[0]
                if len(rr):
                    ax.axvspan(ring_sn[rr[0]],ring_sn[rr[-1]],alpha=0.08,label=name)
            ax.set_xlabel('Reference axial coordinate s/L')
            ax.set_ylabel('Ring mean |S_mises error| (MPa)')
            ax.set_title(path.stem)
            ax.grid(True,alpha=.25)
            # unique legend entries
            h,l=ax.get_legend_handles_labels(); seen={};
            for hh,ll in zip(h,l): seen.setdefault(ll,hh)
            ax.legend(seen.values(),seen.keys(),fontsize=8,ncol=2)
            fig.tight_layout(); fig.savefig(sample_dir/'ring_error_profile.png',dpi=180); plt.close(fig)

            fig,ax=plt.subplots(figsize=(7,4.5))
            names=[r['region'] for r in region_rows]; vals=[r['mae_MPa'] for r in region_rows]
            ax.bar(names,vals)
            ax.set_ylabel('Mean |S_mises error| (MPa)'); ax.set_title(path.stem)
            ax.tick_params(axis='x',rotation=20); ax.grid(True,axis='y',alpha=.25)
            fig.tight_layout(); fig.savefig(sample_dir/'region_mae.png',dpi=180); plt.close(fig)
        except Exception as e:
            print(f'[WARN] plot failed for {path.name}: {e}')

    print(f'[OK] {path.name}: rings={nr}, global_MAE={summary["global"]["mae"]:.6g} MPa, out={sample_dir}')
    return region_rows,excl,ring_rows,summary


def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--input-file',type=Path)
    g.add_argument('--input-dir',type=Path)
    ap.add_argument('--glob',default='*.vtu',help='Used with --input-dir, e.g. "*final*.vtu" or "*.vtu"')
    ap.add_argument('--out-dir',type=Path,default=Path('stress_region_diagnosis'))
    ap.add_argument('--error-field',default='Outer_SPOS_S_Mises_AbsError_MPa')
    ap.add_argument('--ref-field',default='X_ref')
    ap.add_argument('--def-field',default='X_def')
    ap.add_argument('--end-rings',type=int,default=1,help='Number of outermost axial rings on each end treated as end-ring region.')
    ap.add_argument('--bend-s-min',type=float,default=None,help='Optional manual bend-zone start in s/L.')
    ap.add_argument('--bend-s-max',type=float,default=None,help='Optional manual bend-zone end in s/L.')
    ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args()
    if (args.bend_s_min is None)!=(args.bend_s_max is None):
        ap.error('--bend-s-min and --bend-s-max must be supplied together')
    if args.input_file:
        files=[args.input_file]
    else:
        files=sorted(args.input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError('No input files matched.')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    all_regions=[]; all_excl=[]; all_rings=[]; summaries=[]
    for p in files:
        try:
            rr,ee,rg,ss=analyze_one(p,args.out_dir,args.error_field,args.ref_field,args.def_field,args.end_rings,args.bend_s_min,args.bend_s_max,not args.no_plots)
            all_regions+=rr; all_excl+=ee; all_rings+=rg; summaries.append(ss)
        except Exception as e:
            print(f'[FAIL] {p}: {type(e).__name__}: {e}')
    write_csv(args.out_dir/'ALL_region_metrics.csv',all_regions)
    write_csv(args.out_dir/'ALL_exclude_end_rings.csv',all_excl)
    write_csv(args.out_dir/'ALL_ring_profiles.csv',all_rings)
    with open(args.out_dir/'ALL_summary.json','w',encoding='utf-8') as f: json.dump(summaries,f,indent=2,ensure_ascii=False)
    print('DONE')
    print('Processed:',len(summaries),'/',len(files))
    print('Output:',args.out_dir)

if __name__=='__main__':
    main()
