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
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--split',choices=['val','test'],default='test'); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=1024); ap.add_argument('--transition-stride',type=int,default=1); ap.add_argument('--max-samples',type=int,default=0); ap.add_argument('--no-amp',action='store_true'); ap.add_argument('--soft-gate',action='store_true'); ap.add_argument('--gate-threshold',type=float,default=0.5)
    args=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(args.checkpoint,map_location=device); stats=ck.get('stats') or load_stats(args.cache_dir/'stats.json'); cfg=ck.get('config',{}); model=CoRotLRDSUN(int(cfg.get('hidden_dim',192)),float(cfg.get('dropout',0.0))).to(device); model.load_state_dict(ck['model']); model.eval()
    recs=split_records(load_manifest(args.cache_dir),args.split); recs=recs[:args.max_samples] if args.max_samples>0 else recs
    global_m=MetricAccumulator(); per=[]; t0=time.time()
    for ir,rec in enumerate(recs,1):
        s=PreparedSample(rec); mm=MetricAccumulator(); ns=0
        for t in range(0,s.T-1,args.transition_stride):
            for surface in (0,1):
                for a in range(0,s.N,args.batch_size):
                    c=np.arange(a,min(a+args.batch_size,s.N),dtype=np.int64); surf=np.full(len(c),surface,np.int64); b=s.make_batch(t,c,surf)
                    pred_state,pred_le,gate,_,_=predict_batch(model,b,stats,device,amp=not args.no_amp,hard_gate=not args.soft_gate,gate_threshold=args.gate_threshold)
                    ps=pred_state.cpu().numpy(); pl=pred_le.cpu().numpy(); gp=gate.cpu().numpy(); mm.update(ps,b['next_state'],pl,b['le_next'],gp,b['delta_peeq'],stats.get('plastic_threshold',1e-10)); global_m.update(ps,b['next_state'],pl,b['le_next'],gp,b['delta_peeq'],stats.get('plastic_threshold',1e-10)); ns+=len(c)
        r={'sample_id':int(rec['sample_id']),'num_predictions':ns}|mm.as_dict(); per.append(r); print(f"TF {ir}/{len(recs)} sample={rec['sample_id']} {json.dumps(r)}",flush=True)
    out={'mode':'teacher_forcing','split':args.split,'checkpoint':str(args.checkpoint),'transition_stride':args.transition_stride,'samples':per,'global':global_m.as_dict(),'seconds':time.time()-t0}; save_json(args.output,out); print(json.dumps(out['global'],indent=2))

if __name__=='__main__': main()
