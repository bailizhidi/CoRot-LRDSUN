#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from corot_lrdsun.data import PreparedSample
from corot_lrdsun.io import load_manifest, split_records, save_json
from corot_lrdsun.metrics import MetricAccumulator
from corot_lrdsun.model import CoRotLRDSUN
from corot_lrdsun.normalization import load_stats
from corot_lrdsun.runtime import predict_batch


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--split',choices=['val','test'],default='test'); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=1024); ap.add_argument('--max-samples',type=int,default=0); ap.add_argument('--no-amp',action='store_true'); ap.add_argument('--soft-gate',action='store_true'); ap.add_argument('--gate-threshold',type=float,default=0.5); ap.add_argument('--save-predictions',action='store_true')
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(args.checkpoint,map_location=device); stats=ck.get('stats') or load_stats(args.cache_dir/'stats.json'); cfg=ck.get('config',{}); model=CoRotLRDSUN(int(cfg.get('hidden_dim',192)),float(cfg.get('dropout',0.0))).to(device); model.load_state_dict(ck['model']); model.eval()
    recs=split_records(load_manifest(args.cache_dir),args.split); recs=recs[:args.max_samples] if args.max_samples>0 else recs
    global_m=MetricAccumulator(); summaries=[]; t0=time.time()
    for ir,rec in enumerate(recs,1):
        s=PreparedSample(rec); pred=np.stack([np.asarray(s.state_outer[0]).copy(),np.asarray(s.state_inner[0]).copy()],axis=0).astype(np.float32); mm=MetricAccumulator(); frames=[]
        save_state=np.empty((s.T,2,s.N,9),np.float32) if args.save_predictions else None
        save_le=np.empty((s.T,2,s.N,4),np.float32) if args.save_predictions else None
        if save_state is not None: save_state[0]=pred
        for t in range(s.T-1):
            frame_m=MetricAccumulator(); next_pred=np.empty_like(pred)
            for surface in (0,1):
                for a in range(0,s.N,args.batch_size):
                    c=np.arange(a,min(a+args.batch_size,s.N),dtype=np.int64); surf=np.full(len(c),surface,np.int64); override=pred[surface,c]
                    b=s.make_batch(t,c,surf,state_override=override)
                    pstate,ple,gate,_,_=predict_batch(model,b,stats,device,amp=not args.no_amp,hard_gate=not args.soft_gate,gate_threshold=args.gate_threshold)
                    ps=pstate.cpu().numpy(); pl=ple.cpu().numpy(); next_pred[surface,c]=ps
                    gp=gate.cpu().numpy(); frame_m.update(ps,b['next_state'],pl,b['le_next'],gp,b['delta_peeq'],stats.get('plastic_threshold',1e-10)); mm.update(ps,b['next_state'],pl,b['le_next'],gp,b['delta_peeq'],stats.get('plastic_threshold',1e-10)); global_m.update(ps,b['next_state'],pl,b['le_next'],gp,b['delta_peeq'],stats.get('plastic_threshold',1e-10))
                    if save_le is not None: save_le[t+1,surface,c]=pl
            pred=next_pred
            if save_state is not None: save_state[t+1]=pred
            frames.append({'frame':t+1}|frame_m.as_dict())
            if (t+1)%30==0 or t==s.T-2: print(f"ROLLOUT sample={rec['sample_id']} frame={t+1}/{s.T-1} vm_rmse={frames[-1]['rmse_S_mises_MPa']:.4f}",flush=True)
        sr={'sample_id':int(rec['sample_id']),'metrics':mm.as_dict(),'final_frame':frames[-1],'frames':frames}; summaries.append(sr)
        save_json(args.output_dir/f"sample_{int(rec['sample_id']):04d}_summary.json",sr)
        if args.save_predictions:
            np.savez(args.output_dir/f"sample_{int(rec['sample_id']):04d}_rollout.npz",state_pred=save_state,LE_pred=save_le)
        print(f"ROLLOUT {ir}/{len(recs)} sample={rec['sample_id']} metrics={json.dumps(mm.as_dict())}",flush=True)
    out={'mode':'true_autoregressive_rollout','split':args.split,'checkpoint':str(args.checkpoint),'samples':summaries,'global':global_m.as_dict(),'seconds':time.time()-t0}; save_json(args.output_dir/'rollout_summary.json',out); print(json.dumps(out['global'],indent=2))

if __name__=='__main__': main()
